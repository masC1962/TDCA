from __future__ import annotations

from copy import deepcopy
import re

from ..budget import Budget
from ..dynamic.graph import (
    BranchState,
    ClaimNode,
    GraphInvariantError,
    GraphOperation,
    OperationType,
    SubgoalNode,
)
from ..llm import BaseLLM
from ..utils import estimate_message_tokens, normalize_text
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2


EDITOR_SYSTEM = """You are an event-triggered reasoning-graph editor, not an answer generator.
Return JSON only as {operations:[...]}. You may propose at most one EXPAND operation that adds one missing
subgoal needed to connect the current typed claims to the target subgoal. Use only supplied node IDs and
state. The subgoal must contain question_template, answer_type, dependencies, and variable_bindings.
Dependencies and binding values must be existing subgoal IDs. Never invent evidence, produce an answer,
special-case the question, duplicate an existing subgoal, or modify graph state. Return [] when no generic,
evidence-grounded structural edit is justified."""


class EventTriggeredGraphEditorV2:
    """Proposes structural edits only after an explicit graph-state event."""

    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicV2ResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config
        self.last_diagnostics: dict = {}

    def propose(
        self,
        graph: DynamicReasoningHypergraphV2,
        event: str,
        branch: BranchState,
        operation_id: str,
        target_id: str,
        token_budget: int,
    ) -> GraphOperation | None:
        messages = [
            {"role": "system", "content": EDITOR_SYSTEM},
            {"role": "user", "content": (
                f"Event: {event}\nTarget subgoal: {target_id}\nRoot question: {graph.question}\n"
                f"Graph state:\n{_summary(graph, branch)}"
            )},
        ]
        max_tokens = max(128, min(int(token_budget), self.config.graph_editor_max_tokens))
        self.budget.require(max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages))
        data, generation = self.llm.generate_json(
            messages, "dynamic_v2_event_graph_editor_v1", max_tokens, self.config.temperature,
        )
        self.budget.record_generation(generation)
        rows = data.get("operations", [])
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            self.last_diagnostics = {"accepted": False, "reason": "no_structural_edit"}
            return None
        row = rows[0]
        if str(row.get("operation", row.get("type", ""))).upper() != "EXPAND":
            self.last_diagnostics = {"accepted": False, "reason": "non_expand_edit"}
            return None
        local = row.get("subgoal", row)
        if not isinstance(local, dict):
            self.last_diagnostics = {"accepted": False, "reason": "invalid_subgoal"}
            return None
        known = {node.node_id for node in graph.subgoals()}
        target = graph.node(target_id, SubgoalNode)
        dependencies = [
            str(value) for value in local.get("dependencies", [])
            if str(value) in known and str(value) != target_id
        ]
        if not dependencies:
            dependencies = list(target.dependencies)
        bindings = {
            str(key): str(value)
            for key, value in (local.get("variable_bindings", {}) or {}).items()
            if str(value) in dependencies
        }
        question = str(local.get("question_template", "")).strip()
        variables = set(re.findall(r"\$[A-Za-z][A-Za-z0-9_]*", question))
        existing = {normalize_text(node.question_template) for node in graph.subgoals()}
        if not question or not variables.issubset(bindings) or normalize_text(question) in existing:
            self.last_diagnostics = {"accepted": False, "reason": "unsafe_or_duplicate_subgoal"}
            return None
        node_id = f"subgoal_dynamic_v2_{graph.step + 1}"
        # The model may select a currently known node that is downstream of the
        # target as a dependency.  The proposed node is then valid in isolation,
        # but attaching target -> proposed -> downstream -> target would create
        # an execution cycle.  Preflight the complete edit on an isolated DAG so
        # an untrusted editor proposal is rejected as a no-op, never surfaced as
        # an infrastructure/invariant failure.
        execution = deepcopy(graph.execution_graph)
        try:
            execution.add_node(node_id, dependencies)
            if target.variable_bindings:
                target_dependencies = list(dict.fromkeys([node_id] + target.dependencies))
            else:
                target_dependencies = [node_id]
            execution.replace_dependencies(target_id, target_dependencies)
        except GraphInvariantError:
            self.last_diagnostics = {"accepted": False, "reason": "unsafe_execution_cycle"}
            return None
        self.last_diagnostics = {"accepted": True, "event": event, "node_id": node_id}
        return GraphOperation(
            operation_id, OperationType.EXPAND, node_id, dependencies, branch.branch_id,
            {
                "subgoals": [{
                    "node_id": node_id,
                    "question_template": question,
                    "instantiated_question": question,
                    "dependencies": dependencies,
                    "variable_bindings": bindings,
                    "answer_type": str(local.get("answer_type", "entity")),
                    "terminal": False,
                    "confidence": 0.1,
                    "uncertainty": 0.9,
                }],
                "attach_target": target_id,
                "attach_node": node_id,
                "event": event,
            },
            f"event_triggered:{event}", "event_triggered_graph_editor_v2",
            {
                "llm_calls": 1.0,
                "tokens": float(generation.prompt_tokens + generation.completion_tokens),
            },
        )


def _summary(graph: DynamicReasoningHypergraphV2, branch: BranchState) -> str:
    lines = [f"branch={branch.branch_id}; assignments={branch.assignments}"]
    for node in graph.subgoals():
        state = graph.belief_states.get(node.node_id)
        lines.append(
            f"SUBGOAL {node.node_id}: deps={node.dependencies}; type={node.answer_type}; "
            f"question={node.question_template!r}; heat={getattr(state, 'computation_heat', 0.0):.3f}"
        )
    for node in graph.claims(branch_id=branch.branch_id):
        semantics = graph.claim_semantics.get(node.node_id)
        lines.append(
            f"CLAIM {node.node_id}: ({node.subject}, {node.relation}, {node.value}); "
            f"types=({getattr(semantics, 'subject_type', 'entity')}, "
            f"{getattr(semantics, 'value_type', 'entity')}); status={node.status.value}; "
            f"support={node.score.absolute_support:.3f}"
        )
    return "\n".join(lines)[:12000]
