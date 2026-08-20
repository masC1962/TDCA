from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from ..budget import Budget
from ..llm import BaseLLM
from ..utils import estimate_message_tokens
from ..utils import normalize_text
from ..verification import normalize_typed_answer
from .config import DynamicResearchConfig
from .graph import (
    BranchState,
    ClaimNode,
    DynamicReasoningHypergraph,
    EvidenceNode,
    GraphOperation,
    OperationType,
    SubgoalNode,
)


SYSTEM = """Derive short answer candidates only from the supplied branch claims and evidence. Return JSON only
as {answers:[...]}. Each answer has branch_id, candidate_answer, answer_type, supporting_claim_ids,
supporting_evidence_ids, inference_type, confidence, answer_type_consistency, contradiction_risk. Use an
explicit multi-premise inference when comparison, aggregation or conjunction is required. Never introduce a
new fact, claim ID or evidence ID. Do not write a free-form reasoning chain. Return no candidate for a branch
whose claims do not answer the root question."""


class GraphGroundedTerminalReasoner:
    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config

    def direct_operations(
        self, graph: DynamicReasoningHypergraph, branches: list[BranchState], operation_prefix: str,
    ) -> tuple[list[GraphOperation], list[BranchState]]:
        root = next((node for node in graph.subgoals() if node.node_id == "subgoal_root"), None)
        if root is None:
            return [], branches
        operations = []
        unresolved = []
        for branch in branches:
            claim_id = branch.assignments.get(root.node_id)
            if not claim_id:
                claim_id = _promotable_editor_dependency(graph, root, branch)
                if not claim_id:
                    unresolved.append(branch)
                    continue
            terminal = graph.node(claim_id, ClaimNode)
            claim_ids = list(dict.fromkeys(list(branch.assignments.values()) + [claim_id]))
            if not self.config.enable_hyperedges:
                # A1/A2 retain a unary provenance edge needed by the AnswerNode
                # schema, while A3 activates genuine multi-premise hyperedges.
                claim_ids = [claim_id]
            evidence_ids = list(dict.fromkeys(
                evidence_id for value in claim_ids
                for evidence_id in graph.node(value, ClaimNode).evidence_refs
            ))
            answer = normalize_typed_answer(terminal.value, root.answer_type, graph.question)
            confidence = min(
                terminal.score.absolute_support,
                sum(graph.node(value, ClaimNode).score.absolute_support for value in claim_ids) / len(claim_ids),
            )
            operations.append(_answer_operation(
                graph, branch, answer, root.answer_type, claim_ids, evidence_ids,
                (
                    (
                        "event_triggered_dependency_completion"
                        if graph.node(terminal.target_subgoal, SubgoalNode).provenance.source == "llm_graph_editor_v1"
                        else "root_alias_dependency_completion"
                    )
                    if terminal.target_subgoal != root.node_id
                    else "direct_terminal_claim" if len(claim_ids) == 1
                    else "dependency_grounded_terminal_claim"
                ),
                confidence, 1.0, terminal.score.raw.contradiction_risk,
                f"{operation_prefix}_{len(operations) + 1}",
            ))
        return operations, unresolved

    def derive_operations(
        self, graph: DynamicReasoningHypergraph, branches: list[BranchState], operation_prefix: str,
    ) -> list[GraphOperation]:
        if not self.config.enable_hyperedges:
            return []
        root = graph.node("subgoal_root", SubgoalNode)
        eligible = [
            branch for branch in branches
            if branch.assignments
            and all(dependency in branch.assignments for dependency in root.dependencies)
        ]
        if not eligible:
            return []
        blocks = []
        allowed_claims: dict[str, set[str]] = {}
        allowed_evidence: dict[str, set[str]] = {}
        for branch in eligible:
            claim_ids = set(branch.assignments.values())
            evidence_ids = {
                evidence_id for claim_id in claim_ids
                for evidence_id in graph.node(claim_id, ClaimNode).evidence_refs
            }
            allowed_claims[branch.branch_id] = claim_ids
            allowed_evidence[branch.branch_id] = evidence_ids
            claim_text = "\n".join(
                f"  [{claim_id}] ({graph.node(claim_id, ClaimNode).subject}, "
                f"{graph.node(claim_id, ClaimNode).relation}, {graph.node(claim_id, ClaimNode).value})"
                for claim_id in sorted(claim_ids)
            )
            evidence_text = "\n".join(
                f"  [{evidence_id}] {graph.node(evidence_id, EvidenceNode).source_span}"
                for evidence_id in sorted(evidence_ids)
            )
            blocks.append(f"Branch {branch.branch_id}:\nClaims:\n{claim_text}\nEvidence:\n{evidence_text}")
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Root question: {graph.question}\n\n" + "\n\n".join(blocks)},
        ]
        self.budget.require(
            self.config.terminal_derivation_max_tokens,
            estimated_prompt_tokens=estimate_message_tokens(messages), final=True,
        )
        data, generation = self.llm.generate_json(
            messages, "dynamic_terminal_derivation_v1", self.config.terminal_derivation_max_tokens,
            self.config.temperature,
        )
        self.budget.record_generation(generation)
        by_branch: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        branch_map = {branch.branch_id: branch for branch in eligible}
        for row in data.get("answers", []):
            if not isinstance(row, dict):
                continue
            branch_id = str(row.get("branch_id", ""))
            if branch_id not in branch_map:
                continue
            claim_ids = [str(value) for value in row.get("supporting_claim_ids", [])]
            evidence_ids = [str(value) for value in row.get("supporting_evidence_ids", [])]
            if not claim_ids or not evidence_ids:
                continue
            if not set(claim_ids).issubset(allowed_claims[branch_id]) or not set(evidence_ids).issubset(allowed_evidence[branch_id]):
                continue
            raw_confidence = _unit(row.get("confidence"))
            if not _compatible_answer_types(str(row.get("answer_type", "entity")), root.answer_type):
                continue
            type_score = _unit(row.get("answer_type_consistency"))
            contradiction = _unit(row.get("contradiction_risk"))
            claim_support = sum(graph.node(value, ClaimNode).score.absolute_support for value in claim_ids) / len(claim_ids)
            confidence = raw_confidence * type_score * (1.0 - contradiction) * claim_support
            by_branch[branch_id].append((confidence, row | {
                "claim_ids": claim_ids, "evidence_ids": evidence_ids,
                "type_score": type_score, "contradiction": contradiction,
            }))
        operations = []
        for branch_id, candidates in sorted(by_branch.items()):
            confidence, row = max(candidates, key=lambda value: (value[0], str(value[1].get("candidate_answer", ""))))
            answer = normalize_typed_answer(
                str(row.get("candidate_answer", "")), str(row.get("answer_type", root.answer_type)), graph.question,
            )
            if not answer:
                continue
            matching_evidence = [
                evidence_id for evidence_id in sorted(allowed_evidence[branch_id])
                if normalize_text(answer) in normalize_text(graph.node(evidence_id, EvidenceNode).source_span)
            ]
            if not matching_evidence:
                continue
            row["evidence_ids"] = list(dict.fromkeys(row["evidence_ids"] + matching_evidence))
            operations.append(_answer_operation(
                graph, branch_map[branch_id], answer,
                str(row.get("answer_type", root.answer_type)), row["claim_ids"], row["evidence_ids"],
                str(row.get("inference_type", "multi_premise_terminal_derivation")),
                confidence, row["type_score"], row["contradiction"],
                f"{operation_prefix}_{len(operations) + 1}",
            ))
        return operations


def _answer_operation(
    graph: DynamicReasoningHypergraph, branch: BranchState, answer: str, answer_type: str,
    claim_ids: list[str], evidence_ids: list[str], inference_type: str,
    confidence: float, type_score: float, contradiction: float, operation_id: str,
) -> GraphOperation:
    suffix = operation_id.replace("op_", "").replace(".", "_")
    answer_id = f"answer_{suffix}"
    edge_id = f"hyperedge_{suffix}"
    return GraphOperation(
        operation_id, OperationType.COMMIT, "subgoal_root", claim_ids,
        branch.branch_id,
        {"mode": "answer", "answer": {
            "node_id": answer_id,
            "candidate_answer": answer,
            "answer_type": answer_type,
            "supporting_claims": claim_ids,
            "supporting_evidence": evidence_ids,
            "derivation_edge": edge_id,
            "confidence": _unit(confidence),
            "answer_type_consistency": _unit(type_score),
            "contradiction_risk": _unit(contradiction),
            "inference_type": inference_type,
            "status": "accepted",
        }},
        "graph_grounded_terminal_answer", "deterministic_terminal_controller",
    )


def _promotable_editor_dependency(
    graph: DynamicReasoningHypergraph, root: SubgoalNode, branch: BranchState,
) -> str | None:
    """Promote only an editor-inserted final relation explicitly attached to root."""
    if len(root.dependencies) != 1:
        return None
    dependency_id = root.dependencies[0]
    dependency = graph.node(dependency_id, SubgoalNode)
    editor_completion = dependency.provenance.source == "llm_graph_editor_v1"
    alias_completion = _template_signature(dependency.question_template) == _template_signature(
        root.question_template,
    )
    if not editor_completion and not alias_completion:
        return None
    claim_id = branch.assignments.get(dependency_id)
    if not claim_id:
        return None
    claim = graph.node(claim_id, ClaimNode)
    if not _compatible_answer_types(claim.answer_type, root.answer_type):
        return None
    return claim_id


def _template_signature(value: str) -> str:
    normalized = re.sub(r"\$[A-Za-z][A-Za-z0-9_]*", " variable ", value.casefold())
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if tokens and tokens[0] in {"what", "which"}:
        tokens[0] = "wh"
    return " ".join(tokens)


def _unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _canonical_type(value: str) -> str:
    normalized = value.lower().strip().replace("-", "_")
    aliases = {
        "human": "person", "people": "person", "individual": "person",
        "nation": "country", "state": "location", "city": "location",
        "place": "location", "geographic_entity": "location",
        "year": "date", "time": "date",
        "count": "number", "quantity": "number", "percentage": "number",
    }
    canonical = aliases.get(normalized, normalized or "entity")
    known = {"entity", "person", "country", "location", "date", "number", "boolean", "list", "set", "collection"}
    return canonical if canonical in known else "entity"


def _compatible_answer_types(proposed: str, expected: str) -> bool:
    left, right = _canonical_type(proposed), _canonical_type(expected)
    return "entity" in {left, right} or left == right or {left, right} <= {"country", "location"}
