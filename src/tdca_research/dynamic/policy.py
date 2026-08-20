from __future__ import annotations

from dataclasses import dataclass

from .config import DynamicResearchConfig
from .graph import CandidateStatus, ClaimNode
from .scoring import CandidateSetSummary


@dataclass(frozen=True)
class CandidateDecision:
    action: str
    candidate_ids: list[str]
    reason: str


def decide_candidate_set(
    candidates: list[ClaimNode], summary: CandidateSetSummary,
    config: DynamicResearchConfig, *, budget_fallback: bool = False,
) -> CandidateDecision:
    viable = sorted(
        [
            candidate for candidate in candidates
            if candidate.status not in {CandidateStatus.ARCHIVED, CandidateStatus.INVALID}
            and candidate.score.absolute_support >= config.retain_support_threshold
        ],
        key=lambda candidate: (-candidate.score.relative_weight, -candidate.score.absolute_support, candidate.node_id),
    )
    if not viable:
        return CandidateDecision("retrieve_or_expand", [], "no_candidate_above_retain_support")
    top = viable[0]
    if budget_fallback:
        return CandidateDecision("commit", [top.node_id], "deterministic_budget_fallback")
    if (
        top.score.absolute_support >= config.commit_support_threshold
        and summary.top_margin >= config.commit_margin_threshold
        and summary.entropy <= config.commit_entropy_threshold
    ):
        return CandidateDecision("commit", [top.node_id], "high_support_low_uncertainty")
    should_branch = (
        len(viable) >= 2
        and (
            summary.top_margin <= config.branch_margin_threshold
            or summary.entropy >= config.branch_entropy_threshold
        )
    )
    if should_branch:
        return CandidateDecision(
            "branch", [candidate.node_id for candidate in viable[:config.max_active_branches]],
            "lazy_branching_high_uncertainty",
        )
    return CandidateDecision("retain", [top.node_id], "plausible_but_not_committable")


def candidate_prune_value(candidate: ClaimNode) -> float:
    raw = candidate.score.raw
    uncertainty_resolution = 1.0 - candidate.score.evidence_gap
    dependency_unlock = raw.dependency_consistency
    evidence_novelty = raw.retrieval_support
    answer_impact = raw.type_match
    return max(0.0, min(1.0, (
        candidate.score.absolute_support
        + uncertainty_resolution
        + dependency_unlock
        + evidence_novelty
        + answer_impact
    ) / 5.0))


def may_reopen(
    committed: ClaimNode, alternative: ClaimNode, config: DynamicResearchConfig,
    *, contradiction_risk: float, steps_since_revision: int,
) -> bool:
    if steps_since_revision < config.revision_cooldown_steps:
        return False
    stronger = alternative.score.absolute_support - committed.score.absolute_support >= config.reopen_score_delta
    contradicted = contradiction_risk >= config.contradiction_threshold
    return stronger or contradicted
