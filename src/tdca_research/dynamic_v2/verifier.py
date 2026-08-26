from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean
from typing import Any

from ..budget import Budget, BudgetExceeded
from ..dynamic.candidates import (
    _blend,
    _dependency_consistency,
    _deterministic_raw,
    _unit,
)
from ..dynamic.graph import (
    CandidateScoreProfile,
    CandidateStatus,
    ClaimNode,
    EvidenceNode,
    GraphOperation,
    OperationType,
    SubgoalNode,
    VerificationSignals,
)
from ..dynamic.scoring import fuse_candidate_scores
from ..llm import BaseLLM, ProviderRefusalError, StructuredOutputError
from ..utils import bounded_context, estimate_message_tokens, normalize_text
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2


V2_VERIFY_SYSTEM = """Independently score each Candidate against only its referenced evidence and stated
dependencies. Return JSON only as {scores:[...]}, one compact row per Candidate and no rows for Existing
comparison claims. Each row contains exactly candidate_id, grounding, entailment, type_match,
dependency_consistency, contradiction_risk, raw_model_confidence, answer_position, and
contradiction_candidate_ids. Scores are in [0,1]. answer_position is subject, value, or none; use none unless
that exact field directly answers the subgoal. Score a true but question-irrelevant tuple low on entailment.
List contradiction IDs only for logically mutually exclusive claims, never merely different values of a
multi-valued relation. Do not rank, normalize, omit candidates, explain, or add reasons."""


class MultiSampleIndependentVerifier:
    """Independent raw scoring passes followed by one deterministic fusion."""

    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicV2ResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config
        self.last_diagnostics: dict[str, Any] = {}

    def propose(
        self,
        graph: DynamicReasoningHypergraphV2,
        subgoal_id: str,
        branch_id: str,
        instantiated_question: str,
        operation_id: str,
        samples: int,
        token_budget: int,
    ) -> GraphOperation | None:
        candidates = [
            claim for claim in graph.claims(subgoal_id, branch_id)
            if claim.status == CandidateStatus.PROPOSED
        ]
        if not candidates:
            return None
        comparison_candidates = [
            claim for claim in graph.claims(subgoal_id, branch_id)
            if claim.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
        ]
        dependency_ids = list(dict.fromkeys(
            dependency for claim in candidates for dependency in claim.dependency_claim_ids
        ))
        blocks = []
        evidence_by_id: dict[str, EvidenceNode] = {}
        proposed_ids = {candidate.node_id for candidate in candidates}
        for candidate in comparison_candidates:
            evidence = [graph.node(value, EvidenceNode) for value in candidate.evidence_refs]
            evidence_by_id.update({node.node_id: node for node in evidence})
            blocks.append(
                f"{'Candidate' if candidate.node_id in proposed_ids else 'Existing comparison claim'} "
                f"{candidate.node_id}: ({candidate.subject}, {candidate.relation}, {candidate.value}); "
                f"types=({graph.claim_semantics[candidate.node_id].subject_type}, "
                f"{graph.claim_semantics[candidate.node_id].value_type}); "
                f"evidence_ids={[node.node_id for node in evidence]}"
            )
        evidence_text = bounded_context([
            f"Evidence [{node_id}] {node.source_span}"
            for node_id, node in sorted(evidence_by_id.items())
        ], self.config.evidence_char_budget)
        dependency_text = "\n".join(
            f"[{node_id}] ({graph.node(node_id, ClaimNode).subject}, "
            f"{graph.node(node_id, ClaimNode).relation}, {graph.node(node_id, ClaimNode).value})"
            for node_id in dependency_ids
        ) or "(none)"
        raw_samples: dict[str, list[VerificationSignals]] = defaultdict(list)
        audits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        projection_votes: dict[str, list[str]] = defaultdict(list)
        contradiction_votes: dict[str, list[str]] = defaultdict(list)
        total_prompt = total_completion = calls = 0
        sample_count = max(1, min(int(samples), self.config.max_independent_verifications))
        per_call_tokens = max(128, min(int(token_budget), self.config.soft_verifier_max_tokens))
        for sample_index in range(sample_count):
            messages = [
                {"role": "system", "content": V2_VERIFY_SYSTEM},
                {"role": "user", "content": (
                    f"Independent scoring pass: {sample_index + 1}/{sample_count}. Do not infer scores from "
                    f"another pass.\nRoot question: {graph.question}\nSubgoal: {instantiated_question}\n"
                    f"Dependency claims:\n{dependency_text}\n\nCandidate claims:\n"
                    + "\n".join(blocks) + f"\n\nShared evidence:\n{evidence_text}"
                )},
            ]
            self.budget.require(per_call_tokens, estimated_prompt_tokens=estimate_message_tokens(messages))
            try:
                data, generation = self.llm.generate_json(
                    messages,
                    f"dynamic_v2_independent_verification_v1_pass_{sample_index + 1}",
                    per_call_tokens,
                    self.config.temperature,
                )
            except (ProviderRefusalError, StructuredOutputError) as exc:
                if sample_index == 0:
                    raise
                if isinstance(exc, ProviderRefusalError):
                    self.budget.record_provider_failure(exc)
                else:
                    try:
                        self.budget.record_generation(exc.generation)
                    except BudgetExceeded:
                        pass
                self.last_diagnostics = {
                    "partial_failure": type(exc).__name__,
                    "failed_pass": sample_index + 1,
                    "finish_reason": getattr(getattr(exc, "generation", None), "finish_reason", ""),
                }
                break
            self.budget.record_generation(generation)
            calls += 1
            total_prompt += generation.prompt_tokens
            total_completion += generation.completion_tokens
            returned = {
                str(row.get("candidate_id")): row
                for row in data.get("scores", []) if isinstance(row, dict)
            }
            for candidate in candidates:
                deterministic = _deterministic_raw(candidate, graph)
                row = returned.get(candidate.node_id)
                if row is None:
                    raw = deterministic
                    mode = "deterministic_missing_row_fallback"
                    answer_position = (
                        "value" if candidate.provenance.metadata.get("answers_subgoal", False) else "none"
                    )
                else:
                    model_contradiction = _unit(row.get("contradiction_risk"))
                    model = VerificationSignals(
                        grounding=_unit(row.get("grounding")),
                        entailment=_unit(row.get("entailment")),
                        type_match=_unit(row.get("type_match")),
                        dependency_consistency=_unit(row.get("dependency_consistency")),
                        retrieval_support=deterministic.retrieval_support,
                        contradiction_risk=model_contradiction,
                        raw_model_confidence=_unit(row.get("raw_model_confidence")),
                        reasons=[str(value) for value in row.get("reasons", [])][:5],
                    )
                    weight = self.config.soft_verifier_model_weight
                    raw = VerificationSignals(
                        grounding=min(deterministic.grounding, model.grounding),
                        entailment=_blend(deterministic.entailment, model.entailment, weight),
                        type_match=_blend(deterministic.type_match, model.type_match, weight),
                        dependency_consistency=_dependency_consistency(
                            candidate, dependency_ids, graph, instantiated_question,
                            _blend(deterministic.dependency_consistency, model.dependency_consistency, weight),
                        ),
                        retrieval_support=deterministic.retrieval_support,
                        contradiction_risk=_blend(0.0, model.contradiction_risk, weight),
                        raw_model_confidence=_blend(
                            deterministic.raw_model_confidence, model.raw_model_confidence, weight,
                        ),
                        reasons=model.reasons,
                    )
                    mode = "deterministic_prior_plus_independent_model_residual"
                    answer_position = str(row.get("answer_position", "")).strip().lower()
                    if answer_position not in {"subject", "value", "none"}:
                        answer_position = (
                            "value" if candidate.provenance.metadata.get("answers_subgoal", False) else "none"
                        )
                    if model_contradiction >= self.config.contradiction_threshold:
                        known_ids = {value.node_id for value in comparison_candidates}
                        contradiction_votes[candidate.node_id].extend(
                            str(value) for value in row.get("contradiction_candidate_ids", [])
                            if str(value) in known_ids and str(value) != candidate.node_id
                        )
                raw_samples[candidate.node_id].append(raw)
                projection_votes[candidate.node_id].append(answer_position)
                audits[candidate.node_id].append({
                    "pass": sample_index + 1, "mode": mode, "raw": raw.__dict__,
                })
        if calls == 0:
            return None
        self.last_diagnostics = {
            **self.last_diagnostics,
            "independent_passes_requested": sample_count,
            "independent_passes_completed": calls,
        }
        aggregated = {
            candidate.node_id: _average(raw_samples[candidate.node_id]) for candidate in candidates
        }
        for candidate in comparison_candidates:
            aggregated.setdefault(candidate.node_id, candidate.score.raw)
        expected_type = graph.node(subgoal_id, SubgoalNode).answer_type
        projection_by_id = {}
        for candidate in comparison_candidates:
            position = _projection_vote(
                projection_votes[candidate.node_id],
                _preserved_projection(candidate),
            )
            structural_position = _structural_projection(
                graph, subgoal_id, candidate, expected_type,
                numeric_aliases=self.config.numeric_output_type_normalization,
            )
            if structural_position != "none" and projection_votes[candidate.node_id]:
                # Query-graph bindings are a harder constraint than an LLM's
                # answer-position label.  A missing model projection can be
                # recovered deterministically, but two explicit, conflicting
                # endpoint decisions are not silently collapsed into one.  The
                # latter is projection uncertainty and must stay off the answer
                # frontier while preserving the independently scored claim.
                # Comparison-only rows have no fresh vote, so their previously
                # verified projection is preserved without reinterpretation.
                position = (
                    structural_position
                    if position in {"none", structural_position}
                    else "none"
                )
            semantics = graph.claim_semantics[candidate.node_id]
            projection_by_id[candidate.node_id] = _type_corrected_projection(
                position, semantics.subject_type, semantics.value_type, expected_type,
                numeric_aliases=self.config.numeric_output_type_normalization,
            )
        answer_ids = {node_id for node_id, value in projection_by_id.items() if value != "none"}
        profiles = _semantic_group_profiles(
            aggregated, comparison_candidates, projection_by_id, answer_ids, self.config,
        )
        scores = {}
        for candidate in comparison_candidates:
            profile = profiles[candidate.node_id]
            contradictions = sorted(set(contradiction_votes[candidate.node_id]) | {
                source_id for source_id, targets in contradiction_votes.items()
                if candidate.node_id in targets
            })
            scores[candidate.node_id] = {
                **profile.raw.__dict__,
                "absolute_support": profile.absolute_support,
                "relative_weight": profile.relative_weight,
                "set_entropy": profile.set_entropy,
                "evidence_gap": profile.evidence_gap,
                "status": "scored" if candidate.node_id in proposed_ids else candidate.status.value,
                "contradiction_links": contradictions,
                "answer_position": projection_by_id[candidate.node_id],
                "scoring_audit": {
                    "independent_passes_requested": sample_count,
                    "independent_passes_completed": calls,
                    "aggregation": "componentwise_arithmetic_mean_before_fusion",
                    "comparison_group": (
                        "slot_answer_alternatives"
                        if candidate.node_id in answer_ids else "non_answer_relational_claims"
                    ),
                    "passes": audits[candidate.node_id] or [{
                        "mode": "preserved_prior_independent_raw_scores",
                        "raw": candidate.score.raw.__dict__,
                    }],
                },
            }
        return GraphOperation(
            operation_id, OperationType.VERIFY, subgoal_id,
            [candidate.node_id for candidate in comparison_candidates], branch_id,
            {"scores": scores}, "multi_sample_independent_raw_scoring",
            "independent_verifier_v2",
            {
                "llm_calls": float(calls),
                "tokens": float(total_prompt + total_completion),
                "prompt_tokens": float(total_prompt),
                "completion_tokens": float(total_completion),
            },
        )


def _average(rows: list[VerificationSignals]) -> VerificationSignals:
    names = (
        "grounding", "entailment", "type_match", "dependency_consistency",
        "retrieval_support", "contradiction_risk", "raw_model_confidence",
    )
    return VerificationSignals(
        **{name: mean(float(getattr(row, name)) for row in rows) for name in names},
        reasons=list(dict.fromkeys(reason for row in rows for reason in row.reasons))[:8],
    )


def _projection_vote(rows: list[str], fallback: str) -> str:
    if not rows:
        return fallback
    counts = {value: rows.count(value) for value in {"subject", "value", "none"}}
    return max(counts, key=lambda value: (counts[value], value == fallback, value == "none"))


def _preserved_projection(candidate: ClaimNode) -> str:
    """Preserve an independent projection decision across comparison-only passes."""
    verified = str(
        candidate.provenance.metadata.get("verified_answer_position", "")
    ).strip().lower()
    if verified in {"subject", "value", "none"}:
        return verified
    # Extraction canonicalizes a subject answer into the stored value endpoint,
    # so the pre-verification fallback is value rather than the raw source label.
    return "value" if candidate.provenance.metadata.get("answers_subgoal", False) else "none"


def _type_corrected_projection(
    position: str, subject_type: str, value_type: str, expected_type: str,
    *, numeric_aliases: bool = False,
) -> str:
    if position == "none":
        return position
    selected = subject_type if position == "subject" else value_type
    if _projection_type_compatible(
        selected, expected_type, numeric_aliases=numeric_aliases,
    ):
        return position
    alternative_position = "value" if position == "subject" else "subject"
    alternative = value_type if position == "subject" else subject_type
    return alternative_position if _projection_type_compatible(
        alternative, expected_type, numeric_aliases=numeric_aliases,
    ) else "none"


def _projection_type_compatible(
    proposed: str, expected: str, *, numeric_aliases: bool = False,
) -> bool:
    aliases = {
        "human": "person", "actor": "person", "actress": "person", "individual": "person",
        "city": "location", "county": "location", "country": "location", "nation": "location",
        "province": "location", "state": "location", "district": "location",
        "administrative_district": "location", "region": "location",
        "geographic_entity": "location", "body_of_water": "location", "place": "location",
        "year": "date", "time": "date",
        "count": "number", "quantity": "number", "percentage": "number",
        "company": "organization", "institution": "organization", "division": "organization",
        "phrase": "textual", "text": "textual", "string": "textual",
        "acronym_expansion": "textual", "definition": "textual", "meaning": "textual",
    }
    if numeric_aliases:
        aliases.update({
            "numerical": "number", "numeric": "number",
            "fraction": "number", "percentage": "number",
            "percent": "number", "decimal": "number", "ratio": "number",
        })
    def options(value: str) -> set[str]:
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        collection = re.fullmatch(r"(?:list|set|collection)\[(.+)\]", normalized)
        if collection:
            normalized = collection.group(1)
        return {
            aliases.get(item, item) for item in normalized.split("_or_") if item
        }

    left = options(proposed)
    right = options(expected)
    return bool(right & {"", "entity", "thing"}) or bool(left & right)


def _structural_projection(
    graph: DynamicReasoningHypergraphV2,
    subgoal_id: str,
    candidate: ClaimNode,
    expected_type: str,
    *,
    numeric_aliases: bool = False,
) -> str:
    """Project the unbound endpoint of a query-constrained tuple.

    The rule is label-free: anchors come only from the compiled query graph and
    controller-owned dependency assignments.  It deliberately abstains unless
    exactly one endpoint is bound and the other has the requested type.
    """
    anchors = {
        normalize_text(str(value))
        for row in graph.query_graph.get("constraints", [])
        if str(row.get("subgoal_id")) == subgoal_id
        for value in row.get("known_entities", [])
        if normalize_text(str(value))
    }
    branch = graph.branches.get(candidate.branch_id)
    subgoal = graph.node(subgoal_id, SubgoalNode)
    if branch is not None:
        for dependency_id in subgoal.dependencies:
            claim_id = branch.assignments.get(dependency_id)
            dependency = graph.nodes.get(str(claim_id))
            if isinstance(dependency, ClaimNode):
                value = normalize_text(dependency.value)
                if value:
                    anchors.add(value)
    subject_bound = normalize_text(candidate.subject) in anchors
    value_bound = normalize_text(candidate.value) in anchors
    if subject_bound == value_bound:
        return "none"
    semantics = graph.claim_semantics[candidate.node_id]
    if subject_bound and _projection_type_compatible(
        semantics.value_type, expected_type, numeric_aliases=numeric_aliases,
    ):
        return "value"
    if value_bound and _projection_type_compatible(
        semantics.subject_type, expected_type, numeric_aliases=numeric_aliases,
    ):
        return "subject"
    return "none"


def _semantic_group_profiles(
    aggregated: dict[str, VerificationSignals],
    candidates: list[ClaimNode],
    projection_by_id: dict[str, str],
    answer_ids: set[str],
    config: DynamicV2ResearchConfig,
) -> dict[str, CandidateScoreProfile]:
    profiles: dict[str, CandidateScoreProfile] = {}
    answer_groups: dict[str, list[str]] = defaultdict(list)
    by_id = {candidate.node_id: candidate for candidate in candidates}
    for node_id in answer_ids:
        candidate = by_id[node_id]
        projected = candidate.subject if projection_by_id[node_id] == "subject" else candidate.value
        answer_groups[normalize_text(projected)].append(node_id)
    if answer_groups:
        group_raw = {
            key: _max_signals([aggregated[node_id] for node_id in node_ids])
            for key, node_ids in answer_groups.items()
        }
        group_profiles, _ = fuse_candidate_scores(group_raw, config)
        for key, node_ids in answer_groups.items():
            for node_id in node_ids:
                singleton, _ = fuse_candidate_scores({node_id: aggregated[node_id]}, config)
                local = singleton[node_id]
                profiles[node_id] = CandidateScoreProfile(
                    raw=local.raw,
                    absolute_support=local.absolute_support,
                    relative_weight=group_profiles[key].relative_weight,
                    set_entropy=group_profiles[key].set_entropy,
                    evidence_gap=local.evidence_gap,
                    scoring_version="dh-v2-semantic-answer-group-v1",
                )
    nonanswer = set(aggregated) - answer_ids
    if nonanswer:
        local, _ = fuse_candidate_scores(
            {node_id: aggregated[node_id] for node_id in nonanswer}, config,
        )
        profiles.update(local)
    return profiles


def _max_signals(rows: list[VerificationSignals]) -> VerificationSignals:
    names = (
        "grounding", "entailment", "type_match", "dependency_consistency",
        "retrieval_support", "raw_model_confidence",
    )
    return VerificationSignals(
        **{name: max(float(getattr(row, name)) for row in rows) for name in names},
        contradiction_risk=max(float(row.contradiction_risk) for row in rows),
        reasons=list(dict.fromkeys(reason for row in rows for reason in row.reasons))[:8],
    )
