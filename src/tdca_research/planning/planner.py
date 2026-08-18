from __future__ import annotations

from typing import Any

from ..budget import Budget
from ..llm import BaseLLM
from ..models import QAExample, ReasoningPlan, ReasoningSlot, VariableBinding
from .dag import PlanValidationError, validate_plan
from ..utils import estimate_message_tokens


SYSTEM = """You decompose multi-hop questions into a small dependency DAG. Return JSON only.
Never answer the original question. Never use hidden labels. Use generic entity variables such as $bridge_1.
The object schema is {plan_type, slots}. Each slot has slot_id, subquestion_template, answer_type,
dependencies, variable_bindings [{variable, source_slot}], output_variable, terminal, confidence.
Use 1-6 slots and explicit dependencies. Every variable in a template must be produced by a dependency."""


class Planner:
    def __init__(self, llm: BaseLLM, budget: Budget, max_tokens: int, temperature: float = 0.0) -> None:
        self.llm = llm
        self.budget = budget
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def from_oracle(example: QAExample) -> ReasoningPlan:
        if not example.oracle_decomposition:
            raise PlanValidationError("oracle decomposition unavailable")
        slots: list[ReasoningSlot] = []
        output_by_original_id: dict[str, str] = {}
        for index, step in enumerate(example.oracle_decomposition, start=1):
            original_id = str(step.get("id", index))
            slot_id = f"slot_{index}"
            output_by_original_id[original_id] = f"$bridge_{index}" if index < len(example.oracle_decomposition) else "$answer"
            question = str(step.get("question", "")).strip()
            answer_type = Planner._infer_answer_type(question, str(step.get("answer", "")))
            dependencies: list[str] = []
            bindings: list[VariableBinding] = []
            for previous_index in range(1, index):
                marker = f"#{previous_index}"
                if marker in question:
                    dependency = f"slot_{previous_index}"
                    variable = slots[previous_index - 1].output_variable
                    question = question.replace(marker, variable)
                    dependencies.append(dependency)
                    bindings.append(VariableBinding(variable=variable, source_slot=dependency))
            slots.append(ReasoningSlot(
                slot_id=slot_id,
                subquestion_template=question,
                dependencies=dependencies,
                variable_bindings=bindings,
                output_variable=output_by_original_id[original_id],
                answer_type=answer_type,
                terminal=index == len(example.oracle_decomposition),
                confidence=1.0,
            ))
        return validate_plan(ReasoningPlan(example.question, slots, source="oracle"))

    @staticmethod
    def _infer_answer_type(question: str, answer: str = "") -> str:
        """Dataset-agnostic wh-word/type inference for oracle annotations."""
        import re

        normalized = question.strip().lower()
        if normalized.startswith(("what year", "which year")):
            return "year"
        if normalized.startswith("when "):
            return "date"
        if normalized.startswith(("where ", "what place", "which place", "what country", "which country")):
            return "location"
        # "Who did TEAM beat?" can answer an organization, team or country;
        # entity is deliberately broader than person and avoids false rejection.
        if normalized.startswith(("who ", "whom ", "whose ")):
            return "entity"
        if normalized.startswith(("how many", "how much")):
            return "number"
        if normalized.startswith(("is ", "are ", "was ", "were ", "did ", "does ", "do ")):
            return "yes_no"
        if re.fullmatch(r"(1\d{3}|20\d{2})", answer.strip()):
            return "year"
        return "entity"

    def create(self, example: QAExample, oracle: bool = False) -> ReasoningPlan:
        if oracle:
            return self.from_oracle(example)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": example.question}]
        self.budget.require(self.max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages))
        data, generation = self.llm.generate_json(
            messages,
            schema_name="reasoning_plan_v1",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self.budget.record_generation(generation)
        try:
            plan = self._parse(example.question, data)
            pruned = self._prune_noncontributing_slots(plan)
            if len(pruned.slots) < len(plan.slots):
                pruned.source = "model_pruned"
            return validate_plan(pruned)
        except (PlanValidationError, TypeError, ValueError):
            # Dataset-agnostic safe fallback. It preserves execution and makes the
            # planning failure visible without inventing benchmark-specific hops.
            return validate_plan(ReasoningPlan(
                question=example.question,
                slots=[ReasoningSlot(
                    slot_id="slot_root",
                    subquestion_template=example.question,
                    answer_type="entity",
                    output_variable="$answer",
                    terminal=True,
                    confidence=0.1,
                )],
                plan_type="direct_fallback",
                source="generic_fallback",
            ))

    @staticmethod
    def _prune_noncontributing_slots(plan: ReasoningPlan) -> ReasoningPlan:
        """Drop side-work outside the ancestor closure of terminal slots.

        No dependency or binding is invented or rewritten. Unknown dependencies,
        cycles, missing bindings and non-sink terminals still fail strict validation.
        """
        by_id = plan.by_id()
        terminal_ids = {slot.slot_id for slot in plan.slots if slot.terminal}
        if not terminal_ids:
            return plan
        required = set(terminal_ids)
        frontier = list(terminal_ids)
        while frontier:
            current = frontier.pop()
            slot = by_id.get(current)
            if slot is None:
                continue
            for dependency in slot.dependencies:
                if dependency not in required:
                    required.add(dependency)
                    frontier.append(dependency)
        return ReasoningPlan(
            question=plan.question,
            slots=[slot for slot in plan.slots if slot.slot_id in required],
            plan_type=plan.plan_type,
            source=plan.source,
        )

    @staticmethod
    def _parse(question: str, data: dict[str, Any]) -> ReasoningPlan:
        slots: list[ReasoningSlot] = []
        for raw in data.get("slots", []):
            if not isinstance(raw, dict):
                continue
            bindings = [
                VariableBinding(variable=str(item.get("variable", "")), source_slot=str(item.get("source_slot", "")))
                for item in raw.get("variable_bindings", []) if isinstance(item, dict)
            ]
            slots.append(ReasoningSlot(
                slot_id=str(raw.get("slot_id", "")),
                subquestion_template=str(raw.get("subquestion_template", "")),
                answer_type=str(raw.get("answer_type", "entity")),
                dependencies=[str(value) for value in raw.get("dependencies", [])],
                variable_bindings=bindings,
                output_variable=str(raw.get("output_variable", "$answer")),
                terminal=bool(raw.get("terminal", False)),
                confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.5)))),
            ))
        return ReasoningPlan(question=question, slots=slots, plan_type=str(data.get("plan_type", "chain")), source="model")
