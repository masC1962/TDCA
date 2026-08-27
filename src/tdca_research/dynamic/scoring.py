from __future__ import annotations

import math
from dataclasses import dataclass

from .config import DynamicResearchConfig
from .graph import CandidateScoreProfile, VerificationSignals


@dataclass(frozen=True)
class CandidateSetSummary:
    entropy: float
    top_margin: float
    top_candidate_id: str | None
    candidate_count: int


def fuse_candidate_scores(
    raw_by_id: dict[str, VerificationSignals],
    config: DynamicResearchConfig,
) -> tuple[dict[str, CandidateScoreProfile], CandidateSetSummary]:
    """Fuse independent raw scores while preserving every component.

    Absolute support and evidence gap remain separate. Relative weights and entropy
    are candidate-set quantities and are never fed back into the raw evidence scores.
    """
    if not raw_by_id:
        return {}, CandidateSetSummary(1.0, 0.0, None, 0)
    positive_weights = {
        "grounding": config.score_weight_grounding,
        "entailment": config.score_weight_entailment,
        "type_match": config.score_weight_type_match,
        "dependency_consistency": config.score_weight_dependency,
        "retrieval_support": config.score_weight_retrieval,
    }
    denominator = sum(positive_weights.values())
    absolute: dict[str, float] = {}
    evidence_gaps: dict[str, float] = {}
    for candidate_id, raw in raw_by_id.items():
        weighted = sum(getattr(raw, name) * weight for name, weight in positive_weights.items()) / denominator
        support = weighted - config.score_weight_contradiction * raw.contradiction_risk
        if getattr(config, "grounding_conjunctive_absolute_support", False):
            # A claim with no evidence-local grounding cannot recover high
            # absolute support by accumulating unrelated additive channels.
            # The raw components and evidence gap remain separately visible.
            support = min(support, raw.grounding)
        absolute[candidate_id] = _unit(support)
        # Evidence insufficiency remains inspectable rather than being hidden in
        # the fused support score.
        evidence_gaps[candidate_id] = _unit(1.0 - (raw.grounding + raw.retrieval_support) / 2.0)
    peak = max(absolute.values())
    exponents = {
        candidate_id: math.exp((score - peak) / config.candidate_temperature)
        for candidate_id, score in absolute.items()
    }
    total = sum(exponents.values()) or 1.0
    relative = {candidate_id: value / total for candidate_id, value in exponents.items()}
    if len(relative) <= 1:
        entropy = 0.0
    else:
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in relative.values()) / math.log(len(relative))
    ranked = sorted(relative, key=lambda key: (-relative[key], -absolute[key], key))
    margin = relative[ranked[0]] - relative[ranked[1]] if len(ranked) > 1 else 1.0
    profiles = {
        candidate_id: CandidateScoreProfile(
            raw=raw_by_id[candidate_id],
            absolute_support=absolute[candidate_id],
            relative_weight=relative[candidate_id],
            set_entropy=_unit(entropy),
            evidence_gap=evidence_gaps[candidate_id],
        )
        for candidate_id in raw_by_id
    }
    return profiles, CandidateSetSummary(
        entropy=_unit(entropy), top_margin=_unit(margin),
        top_candidate_id=ranked[0], candidate_count=len(ranked),
    )


def retrieval_support_from_ranks(ranks: list[int]) -> float:
    """Query-local, retriever-agnostic rank support; never compares raw BM25 scores."""
    valid = [rank for rank in ranks if rank > 0]
    if not valid:
        return 0.0
    return max(1.0 / rank for rank in valid)


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
