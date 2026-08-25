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
Return JSON only as {operations:[...]}, with at most one operation. Normally use EXPAND to add one missing
subgoal needed to connect current typed claims to the target. If and only if the terminal target question
has accidentally stopped at an intermediate relation and visibly omits a remaining relation from the Root
question, use REPAIR_ROOT with root_question_template and bridge_variable. The repaired template must ask
the original final objective, replace the already-computed intermediate with that one $bridge variable,
and contain no answer. REPAIR_ROOT atomically demotes the current target to an intermediate subgoal.
For REPAIR_ROOT, bridge_variable MUST begin with '$' and the exact same literal MUST occur exactly once in
root_question_template. Replace the full phrase computed by the old terminal target with this variable;
never return the unchanged Root question as the template.
Decide this by the outermost requested relation: if the Root question's final requested relation differs
from the terminal target's requested relation and the target already has a grounded claim, you MUST use
REPAIR_ROOT. Upstream entity details do not count as a different outermost relation.
For EXPAND, provide subgoal with question_template, answer_type, dependencies, and variable_bindings;
dependencies and binding values must be existing subgoal IDs. Never invent evidence, produce an answer,
special-case the question, duplicate a subgoal, or modify graph state. Return [] when no generic,
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
                f"Structural objective audit: {_objective_audit(graph, target_id)}\n"
                f"Graph state:\n{_summary(graph, branch)}"
            )},
        ]
        max_tokens = max(128, min(int(token_budget), self.config.graph_editor_max_tokens))
        self.budget.require(max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages))
        data, generation = self.llm.generate_json(
            messages, "dynamic_v2_event_graph_editor_v5_literal_bridge_contract", max_tokens, self.config.temperature,
        )
        self.budget.record_generation(generation)
        rows = data.get("operations", [])
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            self.last_diagnostics = {"accepted": False, "reason": "no_structural_edit"}
            return None
        row = rows[0]
        operation_kind = str(row.get("operation", row.get("type", ""))).upper()
        if operation_kind == "REPAIR_ROOT":
            return self._root_repair(
                graph, row, event, branch, operation_id, target_id, generation,
            )
        if operation_kind != "EXPAND":
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


    def _root_repair(
        self, graph, row, event, branch, operation_id, target_id, generation,
    ) -> GraphOperation | None:
        target = graph.node(target_id, SubgoalNode)
        if not target.terminal:
            self.last_diagnostics = {"accepted": False, "reason": "root_repair_requires_terminal"}
            return None
        question = str(row.get("root_question_template", "")).strip()
        variables = set(re.findall(r"\$[A-Za-z][A-Za-z0-9_]*", question))
        if len(variables) != 1:
            self.last_diagnostics = {"accepted": False, "reason": "unsafe_root_rewrite_binding"}
            return None
        # The template is the authoritative executable object. Derive its sole
        # binding instead of trusting a redundant model field that may omit the
        # leading '$' while expressing the same variable.
        variable = next(iter(variables))
        old_tokens = set(normalize_text(target.question_template).split())
        root_tokens = set(normalize_text(graph.question).split())
        rewrite_tokens = set(normalize_text(question.replace(variable, "")).split())
        residual = root_tokens - old_tokens - _ROOT_REPAIR_STOPWORDS
        if (
            normalize_text(question) == normalize_text(target.question_template)
            or not residual
            or not (residual & rewrite_tokens)
        ):
            self.last_diagnostics = {"accepted": False, "reason": "root_rewrite_has_no_residual_objective"}
            return None
        node_id = f"subgoal_dynamic_v2_{graph.step + 1}"
        execution = deepcopy(graph.execution_graph)
        try:
            execution.add_node(node_id, list(target.dependencies))
            execution.replace_dependencies(target_id, [node_id])
        except GraphInvariantError:
            self.last_diagnostics = {"accepted": False, "reason": "unsafe_execution_cycle"}
            return None
        self.last_diagnostics = {
            "accepted": True, "event": event, "node_id": node_id,
            "mode": "repair_underdecomposed_root",
        }
        return GraphOperation(
            operation_id, OperationType.EXPAND, node_id, list(target.dependencies), branch.branch_id,
            {
                "subgoals": [{
                    "node_id": node_id,
                    "question_template": target.question_template,
                    "instantiated_question": target.instantiated_question,
                    "dependencies": list(target.dependencies),
                    "variable_bindings": dict(target.variable_bindings),
                    "answer_type": target.answer_type,
                    "terminal": False,
                    "confidence": target.confidence,
                    "uncertainty": target.uncertainty,
                }],
                "attach_target": target_id,
                "attach_node": node_id,
                "target_rewrite": {
                    "question_template": question,
                    "variable_bindings": {variable: node_id},
                    "dependencies": [node_id],
                    "answer_type": target.answer_type,
                },
                "event": event,
            },
            f"event_triggered:{event}", "event_triggered_graph_editor_v2",
            {
                "llm_calls": 1.0,
                "tokens": float(generation.prompt_tokens + generation.completion_tokens),
            },
        )


def editor_preallocation_preflight(
    graph: DynamicReasoningHypergraphV2, event: str, target_id: str,
) -> dict:
    """Reject editor allocation unless an executable diff is known in advance.

    The current editor asks a model to invent both the missing node and its
    bindings, so even a plausible structural event cannot guarantee a non-empty
    diff before spending a call.  v2.4.1 therefore routes recoverable belief
    gaps to concrete RETRIEVE/BRANCH/MERGE actions and records this explicit
    rejection.  Legacy versions remain free to call ``propose`` directly.
    """
    target = graph.node(target_id, SubgoalNode)
    return {
        "allowed": False,
        "reason_code": "model_dependent_diff_not_preflightable",
        "event": str(event),
        "target_terminal": bool(target.terminal),
        "candidate_diff_count": 0,
    }


_ROOT_REPAIR_STOPWORDS = {
    "a", "an", "are", "did", "do", "does", "for", "from", "in", "is", "of", "on",
    "the", "to", "was", "were", "what", "when", "where", "which", "who", "whose",
}


def _objective_audit(graph: DynamicReasoningHypergraphV2, target_id: str) -> str:
    target = graph.node(target_id, SubgoalNode)
    root_tokens = set(normalize_text(graph.question).split())
    target_tokens = set(normalize_text(target.question_template).split())
    residual = sorted(root_tokens - target_tokens - _ROOT_REPAIR_STOPWORDS)
    grounded = sum(
        1 for claim in graph.claims(target_id)
        if claim.score.absolute_support >= 0.5
    )
    return (
        f"target_terminal={target.terminal}; target_question={target.question_template!r}; "
        f"target_grounded_claim_count={grounded}; root_terms_absent_from_target={residual}. "
        "Compare the outermost requested relation of Root question and target_question first. "
        "Use REPAIR_ROOT when those relations differ and grounded target state already computes the "
        "intermediate; otherwise do not treat upstream entity terms as a missing final relation."
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
