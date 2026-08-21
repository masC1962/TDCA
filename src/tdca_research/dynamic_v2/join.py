from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..budget import Budget
from ..dynamic.graph import CandidateStatus, ClaimNode, EvidenceNode, GraphOperation, OperationType
from ..llm import BaseLLM
from ..utils import estimate_message_tokens, normalize_text, stable_hash
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2


JOIN_SYSTEM = """Validate one proposed relational join using only the supplied premise claims and their
evidence. Return JSON only with valid, reason_codes, and derived_claim. derived_claim must contain subject,
relation, value, subject_type, value_type, derivation_confidence, type_match, dependency_consistency, and
qualifiers. The derived claim must follow from all premises jointly; copying one premise is invalid. Do not
use prior knowledge or question-specific rules. The sole exception is a named variable-binding projection
premise: its tuple may be preserved when the other premise establishes its dependency binding. Return
valid=false when the shared binding does not license the composition."""


@dataclass(frozen=True)
class JoinCandidate:
    premise_ids: tuple[str, ...]
    binding: str
    target_subgoal: str
    signature: str
    join_depth: int
    orientation: str = "value_subject"
    open_endpoints: tuple[str, str] = ("", "")
    projection_premise_id: str = ""


class MultiHopJoinEngine:
    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicV2ResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config
        self.last_diagnostics: dict[str, Any] = {}

    def discover(
        self, graph: DynamicReasoningHypergraphV2, branch_id: str, target_subgoal: str,
    ) -> list[JoinCandidate]:
        claims = _accessible_claims(graph, branch_id)
        existing = {
            semantics.join_signature for semantics in graph.claim_semantics.values()
            if semantics.join_signature
        }
        existing_endpoints = {
            (
                semantics.normalized_subject,
                semantics.normalized_value,
                graph.node(node_id, ClaimNode).target_subgoal,
            )
            for node_id, semantics in graph.claim_semantics.items()
            if semantics.join_depth > 0 and node_id in graph.nodes
        }
        rows: list[JoinCandidate] = []
        for left_index, left in enumerate(claims):
            left_semantics = graph.claim_semantics[left.node_id]
            for right in claims[left_index + 1:]:
                if _depends_on(graph, left.node_id, right.node_id) or _depends_on(
                    graph, right.node_id, left.node_id,
                ):
                    continue
                right_semantics = graph.claim_semantics[right.node_id]
                for first, second, orientation, binding, endpoints in _endpoint_unifications(
                    left, left_semantics, right, right_semantics,
                ):
                    projection_premise_id = _projection_premise(
                        graph, first, second, target_subgoal,
                    )
                    normalized_endpoints = tuple(normalize_text(value) for value in endpoints)
                    if (
                        normalized_endpoints[0], normalized_endpoints[1], target_subgoal,
                    ) in existing_endpoints:
                        continue
                    depth = max(left_semantics.join_depth, right_semantics.join_depth) + 1
                    if depth > self.config.max_join_depth:
                        continue
                    signature = stable_hash({
                        "premises": [first.node_id, second.node_id],
                        "binding": binding,
                        "orientation": orientation,
                        "open_endpoints": normalized_endpoints,
                        "projection_premise_id": projection_premise_id,
                        "target": target_subgoal,
                    })
                    if signature in existing:
                        continue
                    rows.append(JoinCandidate(
                        premise_ids=(first.node_id, second.node_id),
                        binding=binding,
                        target_subgoal=target_subgoal,
                        signature=signature,
                        join_depth=depth,
                        orientation=orientation,
                        open_endpoints=endpoints,
                        projection_premise_id=projection_premise_id,
                    ))
        unique = {row.signature: row for row in rows}
        return sorted(
            unique.values(),
            key=lambda row: (row.join_depth, row.premise_ids),
        )

    def propose(
        self,
        graph: DynamicReasoningHypergraphV2,
        candidate: JoinCandidate,
        operation_id: str,
        token_budget: int | None = None,
    ) -> GraphOperation | None:
        premises = [graph.node(node_id, ClaimNode) for node_id in candidate.premise_ids]
        projection = next((
            row for row in premises if row.node_id == candidate.projection_premise_id
        ), None)
        if projection is not None:
            other_support = min(
                row.score.absolute_support for row in premises if row.node_id != projection.node_id
            )
            if (
                projection.score.absolute_support >= self.config.commit_support_threshold
                and other_support >= self.config.commit_support_threshold
                and projection.score.raw.dependency_consistency
                >= self.config.commit_support_threshold
                and projection.score.raw.type_match >= self.config.commit_support_threshold
            ):
                self.last_diagnostics = {
                    "accepted": True,
                    "reason_codes": [
                        "exact_dependency_binding",
                        "independent_verifier_projection",
                        "no_additional_generation",
                    ],
                }
                semantics = graph.claim_semantics[projection.node_id]
                derived = {
                    "subject": projection.subject,
                    "relation": projection.relation,
                    "value": projection.value,
                    "subject_type": semantics.subject_type,
                    "value_type": semantics.value_type,
                    "derivation_confidence": min(
                        row.score.absolute_support for row in premises
                    ),
                    "type_match": projection.score.raw.type_match,
                    "dependency_consistency": projection.score.raw.dependency_consistency,
                    "qualifiers": {
                        "projection_premise_id": projection.node_id,
                        "validation": "independent_raw_scoring",
                    },
                }
                return self._operation(
                    graph, candidate, derived, operation_id,
                    "independently_verified_variable_binding_projection",
                    "deterministic_projection_join_v2",
                    {"llm_calls": 0.0, "tokens": 0.0},
                )
        blocks = []
        for claim in premises:
            evidence = [graph.node(node_id, EvidenceNode) for node_id in claim.evidence_refs]
            source_spans = [
                str(value).strip()
                for value in claim.provenance.metadata.get("source_spans", [])
                if str(value).strip()
            ]
            if not source_spans:
                source_spans = [node.source_span[:500].strip() for node in evidence]
            source_spans = list(dict.fromkeys(source_spans))
            span_budget = min(2000, max(500, self.config.evidence_char_budget // len(premises)))
            grounded = "\n".join(
                f"Grounded span {index}: {value}"
                for index, value in enumerate(source_spans, start=1)
            )[:span_budget]
            blocks.append(
                f"Premise [{claim.node_id}] ({claim.subject}, {claim.relation}, {claim.value}); "
                f"types=({graph.claim_semantics[claim.node_id].subject_type}, "
                f"{graph.claim_semantics[claim.node_id].value_type})\n"
                + f"Evidence IDs: {[node.node_id for node in evidence]}\n{grounded}"
            )
        messages = [
            {"role": "system", "content": JOIN_SYSTEM},
            {"role": "user", "content": (
                f"Root question: {graph.question}\nShared binding: {candidate.binding}\n"
                f"Join orientation: {candidate.orientation}\n"
                f"Required open endpoints: {candidate.open_endpoints[0]} <-> "
                f"{candidate.open_endpoints[1]}\n"
                f"Allowed variable-binding projection premise: "
                f"{candidate.projection_premise_id or 'none'}\n"
                f"Target subgoal: {candidate.target_subgoal}\n\n" + "\n\n".join(blocks)
            )},
        ]
        max_tokens = max(128, min(
            int(token_budget or self.config.join_validation_max_tokens),
            self.config.join_validation_max_tokens,
        ))
        self.budget.require(max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages))
        data, generation = self.llm.generate_json(
            messages, "dynamic_v2_typed_join_validation_v1", max_tokens, self.config.temperature,
        )
        self.budget.record_generation(generation)
        derived = data.get("derived_claim", {})
        valid = bool(data.get("valid", False)) and isinstance(derived, dict)
        normalized_derived = {
            **derived,
            "subject": _scalar_text(derived.get("subject")),
            "relation": _scalar_text(derived.get("relation")),
            "value": _scalar_text(derived.get("value")),
        } if isinstance(derived, dict) else {}
        if not valid or not all(normalized_derived.get(key) for key in ("subject", "relation", "value")):
            self.last_diagnostics = {
                "accepted": False,
                "reason_codes": [str(value) for value in data.get("reason_codes", [])][:5],
            }
            return None
        derived = normalized_derived
        derived_endpoints = {
            normalize_text(derived["subject"]), normalize_text(derived["value"]),
        }
        required_endpoints = {normalize_text(value) for value in candidate.open_endpoints}
        projection = next((
            row for row in premises if row.node_id == candidate.projection_premise_id
        ), None)
        projection_endpoints = {
            normalize_text(projection.subject), normalize_text(projection.value),
        } if projection is not None else set()
        projection_match = bool(projection_endpoints) and derived_endpoints == projection_endpoints
        premise_triples = {
            (normalize_text(row.subject), normalize_text(row.relation), normalize_text(row.value))
            for row in premises
        }
        derived_triple = (
            normalize_text(derived["subject"]),
            normalize_text(derived["relation"]),
            normalize_text(derived["value"]),
        )
        if derived_endpoints != required_endpoints and not projection_match:
            self.last_diagnostics = {"accepted": False, "reason_codes": ["join_endpoint_mismatch"]}
            return None
        if derived_triple in premise_triples and not projection_match:
            self.last_diagnostics = {"accepted": False, "reason_codes": ["degenerate_premise_copy"]}
            return None
        self.last_diagnostics = {
            "accepted": True,
            "reason_codes": [str(value) for value in data.get("reason_codes", [])][:5],
        }
        node_id = f"join_claim_{graph.step + 1}_{candidate.signature[:12]}"
        return self._operation(
            graph, candidate, derived, operation_id,
            "validated_typed_multi_hop_join", "multi_hop_join_engine_v2",
            {
                "llm_calls": 1.0,
                "tokens": float(generation.prompt_tokens + generation.completion_tokens),
            },
        )

    def _operation(
        self, graph, candidate, derived, operation_id, reason, proposed_by, estimated_cost,
    ) -> GraphOperation:
        premises = [graph.node(node_id, ClaimNode) for node_id in candidate.premise_ids]
        node_id = f"join_claim_{graph.step + 1}_{candidate.signature[:12]}"
        edge_id = f"join_edge_{graph.step + 1}_{candidate.signature[:12]}"
        return GraphOperation(
            operation_id=operation_id,
            operation_type=OperationType.MERGE,
            target_id=candidate.target_subgoal,
            source_ids=list(candidate.premise_ids),
            branch_id=premises[0].branch_id,
            payload={
                "mode": "derive_join",
                "binding": candidate.binding,
                "join_signature": candidate.signature,
                "validation": {
                    "valid": True,
                    "reason_codes": self.last_diagnostics["reason_codes"],
                },
                "claim": {
                    "node_id": node_id,
                    "edge_id": edge_id,
                    "target_subgoal": candidate.target_subgoal,
                    **derived,
                },
            },
            reason=reason,
            proposed_by=proposed_by,
            estimated_cost=estimated_cost,
        )
    def deterministic_operation(
        self,
        graph: DynamicReasoningHypergraphV2,
        candidate: JoinCandidate,
        derived_claim: dict[str, Any],
        operation_id: str,
    ) -> GraphOperation:
        """Offline-test constructor; production joins always use `propose`."""
        signature = candidate.signature
        return GraphOperation(
            operation_id, OperationType.MERGE, candidate.target_subgoal,
            list(candidate.premise_ids), graph.node(candidate.premise_ids[0], ClaimNode).branch_id,
            {
                "mode": "derive_join", "binding": candidate.binding,
                "join_signature": signature,
                "validation": {"valid": True, "reason_codes": ["offline_fixture"]},
                "claim": {
                    "node_id": f"join_claim_{graph.step + 1}_{signature[:12]}",
                    "edge_id": f"join_edge_{graph.step + 1}_{signature[:12]}",
                    "target_subgoal": candidate.target_subgoal,
                    **derived_claim,
                },
            },
            "validated_typed_multi_hop_join", "multi_hop_join_engine_v2",
            {"llm_calls": 0.0, "tokens": 0.0},
        )


def _scalar_text(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return ""
    return str(value).strip()


def _compatible(left: str, right: str) -> bool:
    left, right = left.lower(), right.lower()
    return left == right or "entity" in {left, right} or {left, right} <= {"country", "location"}


def _accessible_claims(
    graph: DynamicReasoningHypergraphV2, branch_id: str,
) -> list[ClaimNode]:
    """Return branch-local claims plus the assigned dependency lineage."""
    local = {
        claim.node_id: claim for claim in graph.claims(branch_id=branch_id)
        if claim.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
        and claim.node_id in graph.claim_semantics
    }
    branch = graph.branches.get(branch_id)
    queue = list((branch.assignments if branch else {}).values())
    seen = set(queue)
    while queue:
        node_id = queue.pop()
        node = graph.nodes.get(node_id)
        if not isinstance(node, ClaimNode):
            continue
        if (
            node.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
            and node.node_id in graph.claim_semantics
        ):
            local[node.node_id] = node
        for dependency_id in node.dependency_claim_ids:
            if dependency_id not in seen:
                seen.add(dependency_id)
                queue.append(dependency_id)
    return sorted(local.values(), key=lambda value: value.node_id)


def _endpoint_unifications(left, left_semantics, right, right_semantics):
    """Enumerate valid length-two relational paths without relation-name rules."""
    rows = []
    if (
        left_semantics.normalized_value == right_semantics.normalized_subject
        and _compatible(left_semantics.value_type, right_semantics.subject_type)
    ):
        rows.append((
            left, right, "value_subject", left_semantics.normalized_value,
            (left.subject, right.value),
        ))
    if (
        right_semantics.normalized_value == left_semantics.normalized_subject
        and _compatible(right_semantics.value_type, left_semantics.subject_type)
    ):
        rows.append((
            right, left, "value_subject", right_semantics.normalized_value,
            (right.subject, left.value),
        ))
    if (
        left_semantics.normalized_subject == right_semantics.normalized_subject
        and _compatible(left_semantics.subject_type, right_semantics.subject_type)
    ):
        rows.append((
            left, right, "shared_subject", left_semantics.normalized_subject,
            (left.value, right.value),
        ))
    if (
        left_semantics.normalized_value == right_semantics.normalized_value
        and _compatible(left_semantics.value_type, right_semantics.value_type)
    ):
        rows.append((
            left, right, "shared_value", left_semantics.normalized_value,
            (left.subject, right.subject),
        ))
    return [row for row in rows if row[3] and normalize_text(row[4][0]) != normalize_text(row[4][1])]


def _depends_on(
    graph: DynamicReasoningHypergraphV2, claim_id: str, possible_ancestor_id: str,
) -> bool:
    semantics = graph.claim_semantics.get(claim_id)
    if semantics is None or semantics.join_depth <= 0:
        return False
    queue = list(graph.node(claim_id, ClaimNode).dependency_claim_ids)
    seen = set(queue)
    while queue:
        node_id = queue.pop()
        if node_id == possible_ancestor_id:
            return True
        node = graph.nodes.get(node_id)
        if not isinstance(node, ClaimNode):
            continue
        for dependency_id in node.dependency_claim_ids:
            if dependency_id not in seen:
                seen.add(dependency_id)
                queue.append(dependency_id)
    return False


def _projection_premise(
    graph: DynamicReasoningHypergraphV2,
    left: ClaimNode,
    right: ClaimNode,
    target_subgoal: str,
) -> str:
    for candidate, other in ((left, right), (right, left)):
        if candidate.target_subgoal != target_subgoal or not candidate.dependency_claim_ids:
            continue
        if set(candidate.dependency_claim_ids) & _dependency_closure(graph, other.node_id):
            return candidate.node_id
    return ""


def _dependency_closure(
    graph: DynamicReasoningHypergraphV2, claim_id: str,
) -> set[str]:
    closure = {claim_id}
    queue = [claim_id]
    while queue:
        node = graph.nodes.get(queue.pop())
        if not isinstance(node, ClaimNode):
            continue
        for dependency_id in node.dependency_claim_ids:
            if dependency_id not in closure:
                closure.add(dependency_id)
                queue.append(dependency_id)
    return closure
