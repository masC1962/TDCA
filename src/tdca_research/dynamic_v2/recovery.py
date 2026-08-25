from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

from ..dynamic.graph import CandidateStatus, ClaimNode, EvidenceNode, SubgoalNode
from ..utils import normalize_text
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2
from .proof import audit_graph_proof, claim_closure


@dataclass(frozen=True)
class ProofUsability:
    """Gold-free verdict for using one claim to close a subgoal.

    Semantic projection alone is deliberately insufficient: the claim must also
    retain independent support, grounding, type consistency, evidence lineage,
    and (for composed targets) dependency/hyperedge closure.
    """

    usable: bool
    reason_codes: tuple[str, ...]
    absolute_support: float
    grounding: float
    type_match: float
    evidence_gap: float
    contradiction_risk: float
    dependency_coverage: float
    proof_connected: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProofGapDiagnosis:
    reason_code: str
    reason_codes: tuple[str, ...]
    target_claim_ids: tuple[str, ...]
    proof_gap_reducibility: float
    feasibility_unlock: float

    def to_payload(self) -> dict:
        return {
            "proof_gap_reason": self.reason_code,
            "proof_gap_reason_codes": list(self.reason_codes),
            "recovery_target_claim_ids": list(self.target_claim_ids),
            "proof_gap_reducibility": self.proof_gap_reducibility,
            "feasibility_unlock": self.feasibility_unlock,
            "recovery_policy": "proof_gap_recovery_v1",
        }


_VERIFIED_STATUSES = {
    CandidateStatus.SCORED,
    CandidateStatus.RETAINED,
    CandidateStatus.REVISED,
    CandidateStatus.COMMITTED,
}


def proof_usable_target_claim(
    graph: DynamicReasoningHypergraphV2,
    claim: ClaimNode,
    subgoal: SubgoalNode,
    config: DynamicV2ResearchConfig,
    *,
    projects_target: bool,
) -> ProofUsability:
    """Apply a conjunctive, inference-time-only proof usability test."""
    reasons: set[str] = set()
    raw = claim.score.raw
    semantics = graph.claim_semantics.get(claim.node_id)
    support_floor = (
        config.terminal_min_absolute_support
        if subgoal.terminal else config.join_min_premise_support
    )
    type_floor = (
        config.terminal_min_type_consistency
        if subgoal.terminal else config.join_min_premise_support
    )
    if not projects_target:
        reasons.add("claim_does_not_project_target")
    if claim.status not in _VERIFIED_STATUSES:
        reasons.add("claim_not_independently_verified")
    if claim.score.absolute_support < support_floor:
        reasons.add("absolute_support_below_proof_floor")
    if raw.grounding < config.join_min_premise_support:
        reasons.add("grounding_below_proof_floor")
    if raw.type_match < type_floor:
        reasons.add("type_match_below_proof_floor")
    if claim.score.evidence_gap > config.terminal_max_evidence_gap:
        reasons.add("evidence_gap_above_proof_ceiling")
    if raw.contradiction_risk > config.terminal_max_contradiction:
        reasons.add("contradiction_above_proof_ceiling")
    if not claim.evidence_refs or any(
        not isinstance(graph.nodes.get(evidence_id), EvidenceNode)
        for evidence_id in claim.evidence_refs
    ):
        reasons.add("missing_grounded_evidence_lineage")

    audit = audit_graph_proof(
        graph, subgoal.node_id, claim.branch_id, [claim.node_id],
    )
    if subgoal.dependencies and audit.dependency_coverage < 1.0:
        reasons.add("missing_dependency_closure")
    if semantics is not None and semantics.join_depth > 0 and not audit.proof_connected:
        reasons.add("joined_claim_lacks_connected_hyperedge")
    return ProofUsability(
        usable=not reasons,
        reason_codes=tuple(sorted(reasons)),
        absolute_support=float(claim.score.absolute_support),
        grounding=float(raw.grounding),
        type_match=float(raw.type_match),
        evidence_gap=float(claim.score.evidence_gap),
        contradiction_risk=float(raw.contradiction_risk),
        dependency_coverage=float(audit.dependency_coverage),
        proof_connected=bool(audit.proof_connected),
    )


def diagnose_proof_gap(
    graph: DynamicReasoningHypergraphV2,
    subgoal: SubgoalNode,
    claims: Iterable[ClaimNode],
    semantic_target_claims: Iterable[ClaimNode],
    verdicts: dict[str, ProofUsability],
    dependency_claim_ids: Iterable[str],
) -> ProofGapDiagnosis:
    """Select one auditable recovery target without answers or oracle labels."""
    claims = list(claims)
    targets = list(semantic_target_claims)
    dependency_ids = set(dependency_claim_ids)
    reasons = {
        reason
        for verdict in verdicts.values()
        for reason in verdict.reason_codes
    }
    if not claims:
        primary = "no_extracted_claim"
        reducibility = 1.0
    elif not targets:
        primary = "no_target_projection"
        reducibility = 0.95
    elif "missing_dependency_closure" in reasons:
        primary = "missing_dependency_closure"
        reducibility = 0.90
    elif "missing_grounded_evidence_lineage" in reasons:
        primary = "missing_grounded_evidence_lineage"
        reducibility = 0.90
    elif "evidence_gap_above_proof_ceiling" in reasons:
        primary = "evidence_gap_above_proof_ceiling"
        reducibility = 0.85
    elif "absolute_support_below_proof_floor" in reasons:
        primary = "absolute_support_below_proof_floor"
        reducibility = 0.80
    elif "grounding_below_proof_floor" in reasons:
        primary = "grounding_below_proof_floor"
        reducibility = 0.80
    elif "type_match_below_proof_floor" in reasons:
        primary = "type_match_below_proof_floor"
        reducibility = 0.70
    elif "joined_claim_lacks_connected_hyperedge" in reasons:
        primary = "joined_claim_lacks_connected_hyperedge"
        reducibility = 0.65
    else:
        primary = sorted(reasons)[0] if reasons else "unresolved_proof_gap"
        reducibility = 0.60

    raw_local = {
        claim.node_id for claim in claims
        if graph.claim_semantics.get(claim.node_id) is not None
        and graph.claim_semantics[claim.node_id].join_depth == 0
        and claim.evidence_refs
    }
    feasibility_unlock = 1.0 if dependency_ids and raw_local else (
        0.75 if dependency_ids or raw_local else 0.50
    )
    return ProofGapDiagnosis(
        reason_code=primary,
        reason_codes=tuple(sorted(reasons or {primary})),
        target_claim_ids=tuple(sorted(claim.node_id for claim in targets)),
        proof_gap_reducibility=reducibility,
        feasibility_unlock=feasibility_unlock,
    )


def proof_gap_recovery_query(
    graph: DynamicReasoningHypergraphV2,
    subgoal: SubgoalNode,
    question: str,
    dependency_claim_ids: Iterable[str],
    claims: Iterable[ClaimNode],
    diagnosis: ProofGapDiagnosis,
    attempted_queries: Iterable[str] = (),
) -> str:
    """Create a deterministic novel query from unresolved graph state only."""
    existing = {normalize_text(value) for value in attempted_queries}
    dependencies = [
        graph.nodes.get(str(claim_id)) for claim_id in dependency_claim_ids
    ]
    anchors = list(dict.fromkeys(
        value
        for claim in dependencies
        if isinstance(claim, ClaimNode)
        for value in (claim.subject, claim.value)
        if value
    ))[:3]
    frontier = sorted(
        claims,
        key=lambda claim: (
            -int(claim.node_id in set(diagnosis.target_claim_ids)),
            -float(claim.score.evidence_gap),
            float(claim.score.absolute_support),
            claim.node_id,
        ),
    )
    anchors.extend(
        value
        for claim in frontier[:2]
        for value in (claim.subject, claim.value)
        if value and value not in anchors
    )
    anchor_text = " ; ".join(anchors[:5])
    candidates = [
        f"Independent source for the missing {subgoal.answer_type} relation"
        + (f" connecting {anchor_text}" if anchor_text else "")
        + f". Gap: {diagnosis.reason_code.replace('_', ' ')}.",
        f"Find independent evidence for the unresolved {subgoal.answer_type} output"
        + (f" from {anchor_text}" if anchor_text else "") + ".",
        f"{question} Verify an independent missing relation.",
    ]
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if normalized not in existing and all(
            _query_overlap(normalized, previous) <= 0.80 for previous in existing
        ):
            return candidate
    return ""


def _query_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(left)))
    right_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
