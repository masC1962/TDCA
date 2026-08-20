from __future__ import annotations

import re
from typing import Any

from ..budget import Budget
from ..llm import BaseLLM
from ..utils import estimate_message_tokens
from .config import DynamicResearchConfig
from .graph import (
    BranchState,
    ClaimNode,
    DynamicReasoningHypergraph,
    GraphOperation,
    OperationType,
    SubgoalNode,
)


SYSTEM = """You are a graph-operation proposer, not a graph mutator. Return JSON only as {operations:[...]}.
Use only node IDs and evidence summaries supplied by the controller. Allowed proposals are EXPAND, MERGE,
and REVISE. EXPAND adds a missing subgoal with question_template, answer_type, dependencies and
variable_bindings. MERGE names genuinely equivalent claim IDs. REVISE may reopen a committed claim or replace
execution dependencies. Never answer the root question, invent evidence, reference unknown IDs, create cycles,
or use hidden labels. For a high-uncertainty or missing-terminal-path event, prefer one concrete EXPAND that
asks for the missing relation immediately before the named target. Return at most two operations; return []
only when no evidence-grounded structural edit is possible. Use exactly
{operations:[{operation:"EXPAND",subgoal:{question_template,answer_type,dependencies,variable_bindings}}]} for
an expansion; dependency and binding values are existing subgoal IDs."""


class EventTriggeredGraphEditor:
    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config

    def propose(
        self, graph: DynamicReasoningHypergraph, event: str, branch: BranchState,
        operation_prefix: str, target_id: str | None = None,
    ) -> list[GraphOperation]:
        summary = _graph_summary(graph, branch)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Event: {event}\nTarget subgoal: {target_id or '(none)'}\nRoot question: {graph.question}\nGraph state:\n{summary}"},
        ]
        self.budget.require(
            self.config.graph_editor_max_tokens,
            estimated_prompt_tokens=estimate_message_tokens(messages),
        )
        data, generation = self.llm.generate_json(
            messages, "dynamic_graph_editor_v1", self.config.graph_editor_max_tokens,
            self.config.temperature,
        )
        self.budget.record_generation(generation)
        proposals = []
        known_subgoals = {node.node_id for node in graph.subgoals()}
        known_claims = {node.node_id for node in graph.claims()}
        existing_questions = {
            _normalized_question(node.question_template) for node in graph.subgoals()
        }
        for index, row in enumerate(data.get("operations", [])[:2], start=1):
            if not isinstance(row, dict):
                continue
            kind = str(row.get("operation") or row.get("type") or "").upper()
            operation_id = f"{operation_prefix}_{index}"
            if kind == "EXPAND":
                local = row.get("subgoal", {}) if isinstance(row.get("subgoal"), dict) else row
                node_id = f"subgoal_dynamic_{graph.step + index}"
                target = graph.node(target_id, SubgoalNode) if target_id in known_subgoals else None
                dependencies = [
                    str(value) for value in local.get("dependencies", [])
                    if str(value) in known_subgoals and str(value) != target_id
                ]
                if target is not None and not dependencies:
                    dependencies = list(target.dependencies)
                value_to_subgoal = {
                    graph.node(claim_id, ClaimNode).value.casefold(): subgoal_id
                    for subgoal_id, claim_id in branch.assignments.items()
                }
                bindings = {}
                for key, value in (local.get("variable_bindings", {}) or {}).items():
                    source = str(value)
                    if source not in dependencies:
                        source = value_to_subgoal.get(source.casefold(), "")
                    if source in dependencies:
                        bindings[str(key)] = source
                question = str(local.get("question_template", "")).strip()
                variables = set(re.findall(r"\$[A-Za-z][A-Za-z0-9_]*", question))
                if (
                    not question or not variables.issubset(bindings)
                    or _normalized_question(question) in existing_questions
                ):
                    continue
                proposals.append(GraphOperation(
                    operation_id, OperationType.EXPAND, node_id, dependencies, branch.branch_id,
                    {"subgoals": [{
                        "node_id": node_id, "question_template": question,
                        "instantiated_question": question, "dependencies": dependencies,
                        "variable_bindings": bindings,
                        "answer_type": str(local.get("answer_type", "entity")),
                        "terminal": bool(local.get("terminal", False)),
                        "confidence": 0.1, "uncertainty": 0.9,
                    }], **({
                        "attach_target": target.node_id, "attach_node": node_id,
                    } if target is not None else {})},
                    f"event_triggered:{event}", "llm_graph_editor_v1",
                    {"llm_calls": 1.0 / max(1, len(data.get("operations", []))),
                     "tokens": float(generation.prompt_tokens + generation.completion_tokens)},
                ))
            elif kind == "MERGE":
                keep = str(row.get("keep_id", ""))
                merge = [str(value) for value in row.get("merge_ids", [])]
                if keep not in known_claims or not merge or any(value not in known_claims for value in merge):
                    continue
                target = graph.node(keep, ClaimNode).target_subgoal
                proposals.append(GraphOperation(
                    operation_id, OperationType.MERGE, target, [keep] + merge,
                    branch.branch_id, {"keep_id": keep, "merge_ids": merge},
                    f"event_triggered:{event}", "llm_graph_editor_v1",
                ))
            elif kind == "REVISE":
                action = str(row.get("action", ""))
                target = str(row.get("target_id", ""))
                if action == "reopen" and target in known_claims:
                    proposals.append(GraphOperation(
                        operation_id, OperationType.REVISE,
                        graph.node(target, ClaimNode).target_subgoal, [target], branch.branch_id,
                        {"action": "reopen", "claim_id": target},
                        f"event_triggered:{event}", "llm_graph_editor_v1",
                    ))
                elif action == "dependencies" and target in known_subgoals:
                    dependencies = [str(value) for value in row.get("dependencies", []) if str(value) in known_subgoals]
                    proposals.append(GraphOperation(
                        operation_id, OperationType.REVISE, target, dependencies, branch.branch_id,
                        {"action": "dependencies", "dependencies": dependencies,
                         "variable_bindings": row.get("variable_bindings", {}) or {}},
                        f"event_triggered:{event}", "llm_graph_editor_v1",
                    ))
        return proposals


def _graph_summary(graph: DynamicReasoningHypergraph, branch: BranchState) -> str:
    lines = [f"branch={branch.branch_id}; assignments={branch.assignments}; score={branch.score:.4f}"]
    for node in graph.subgoals():
        lines.append(
            f"SUBGOAL {node.node_id}: status={node.status.value}; deps={node.dependencies}; "
            f"q={node.question_template!r}; terminal={node.terminal}; uncertainty={node.uncertainty:.3f}"
        )
    for node in graph.claims():
        lines.append(
            f"CLAIM {node.node_id}: target={node.target_subgoal}; status={node.status.value}; "
            f"value={node.value!r}; support={node.score.absolute_support:.3f}; "
            f"contradictions={node.contradiction_links}"
        )
    return "\n".join(lines)[:12000]


def _normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
