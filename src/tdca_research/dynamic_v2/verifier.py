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


HARA_ALIGNMENT_SYSTEM = """Judge only whether each Candidate satisfies the compiled query constraint. Do not
judge evidence truth, grounding, entailment, confidence, or contradictions in this pass. Return JSON only as
{scores:[...]}, one compact row per Candidate and no rows for Existing comparison claims. Each row contains
exactly candidate_id, relation_target_alignment, subject_binding_coverage, dependency_binding_coverage,
qualifier_coverage, output_slot_coverage, and answer_position. Every score is in [0,1].
relation_target_alignment asks whether the tuple expresses the requested relation. subject_binding_coverage
asks whether it is about the entity fixed by the current subgoal/branch. dependency_binding_coverage asks
whether every declared bridge output is correctly used. qualifier_coverage asks whether all explicit
temporal/comparison/cardinality/set/negation constraints are covered; use 1 when none are declared.
output_slot_coverage asks whether the selected endpoint has the requested semantic role and type.
answer_position is subject, value, or none. Do not rank, normalize, omit candidates, or explain."""


ALIGNMENT_FIELDS = (
    "relation_target_alignment",
    "subject_binding_coverage",
    "dependency_binding_coverage",
    "qualifier_coverage",
    "output_slot_coverage",
)


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
        query_constraint = next((
            dict(row) for row in graph.query_graph.get("constraints", [])
            if str(row.get("subgoal_id")) == subgoal_id
        ), {})
        constraint_text = (
            f"description={query_constraint.get('description', instantiated_question)!r}; "
            f"known_entities={query_constraint.get('known_entities', [])}; "
            f"input_variables={query_constraint.get('input_variables', [])}; "
            f"output_variable={query_constraint.get('output_variable', '')!r}; "
            f"expected_output_type={graph.node(subgoal_id, SubgoalNode).answer_type!r}; "
            f"required_qualifiers={query_constraint.get('required_qualifiers', [])}"
        )
        raw_samples: dict[str, list[VerificationSignals]] = defaultdict(list)
        audits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        projection_votes: dict[str, list[str]] = defaultdict(list)
        alignment_samples: dict[str, list[dict[str, float]]] = defaultdict(list)
        alignment_audits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        alignment_projection_votes: dict[str, list[str]] = defaultdict(list)
        contradiction_votes: dict[str, list[str]] = defaultdict(list)
        total_prompt = total_completion = calls = evidence_calls = alignment_calls = 0
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
            evidence_calls += 1
            total_prompt += generation.prompt_tokens
            total_completion += generation.completion_tokens
            returned = {
                str(row.get("candidate_id")): row
                for row in data.get("scores", []) if isinstance(row, dict)
            }
            for candidate in candidates:
                deterministic = _deterministic_raw(candidate, graph)
                if self.config.evidence_endpoint_grounding:
                    deterministic.grounding = min(
                        deterministic.grounding,
                        _evidence_endpoint_grounding(candidate, graph),
                    )
                    deterministic.reasons.append(
                        "evidence_local_tuple_endpoint_grounding"
                    )
                elif self.config.generic_evidence_endpoint_grounding:
                    deterministic.grounding = min(
                        deterministic.grounding,
                        _generic_evidence_endpoint_grounding(candidate, graph),
                    )
                    deterministic.reasons.append(
                        "generic_evidence_entity_endpoint_grounding"
                    )
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
        if evidence_calls == 0:
            return None
        aggregated = {
            candidate.node_id: _average(raw_samples[candidate.node_id])
            for candidate in candidates
        }
        evidence_profiles = {
            candidate.node_id: aggregated.get(candidate.node_id, candidate.score.raw)
            for candidate in comparison_candidates
        }
        controller_alignment = bool(
            self.config.query_conditioned_semantic_alignment
            and self.config.controller_query_alignment_certificates
        )
        if controller_alignment:
            certificates = _controller_query_alignment_certificates(
                comparison_candidates, graph, subgoal_id, evidence_profiles,
            )
            for candidate in candidates:
                certificate = certificates[candidate.node_id]
                alignment_samples[candidate.node_id].append({
                    name: float(certificate[name]) for name in ALIGNMENT_FIELDS
                })
                alignment_projection_votes[candidate.node_id].append(
                    str(certificate["answer_position"])
                )
                alignment_audits[candidate.node_id].append({
                    "pass": 0,
                    "mode": "controller_query_graph_certificate",
                    "raw": {
                        name: float(certificate[name]) for name in ALIGNMENT_FIELDS
                    },
                    "answer_position": str(certificate["answer_position"]),
                    "certificate": certificate["certificate"],
                })
        elif self.config.query_conditioned_semantic_alignment:
            for sample_index in range(sample_count):
                messages = [
                    {"role": "system", "content": HARA_ALIGNMENT_SYSTEM},
                    {"role": "user", "content": (
                        f"Independent alignment pass: {sample_index + 1}/{sample_count}. "
                        f"Do not infer evidence scores.\nRoot question: {graph.question}\n"
                        f"Subgoal: {instantiated_question}\n"
                        f"Compiled query constraint: {constraint_text}\n"
                        f"Dependency claims:\n{dependency_text}\n\nCandidate claims:\n"
                        + "\n".join(blocks)
                    )},
                ]
                self.budget.require(
                    per_call_tokens,
                    estimated_prompt_tokens=estimate_message_tokens(messages),
                )
                try:
                    data, generation = self.llm.generate_json(
                        messages,
                        f"hara_v24319_independent_query_alignment_pass_{sample_index + 1}",
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
                        **self.last_diagnostics,
                        "query_alignment_partial_failure": type(exc).__name__,
                        "query_alignment_failed_pass": sample_index + 1,
                    }
                    break
                self.budget.record_generation(generation)
                calls += 1
                alignment_calls += 1
                total_prompt += generation.prompt_tokens
                total_completion += generation.completion_tokens
                returned = {
                    str(row.get("candidate_id")): row
                    for row in data.get("scores", []) if isinstance(row, dict)
                }
                for candidate in candidates:
                    row = returned.get(candidate.node_id, {})
                    alignment = {
                        name: _unit(row.get(name)) for name in ALIGNMENT_FIELDS
                    }
                    alignment_samples[candidate.node_id].append(alignment)
                    answer_position = str(row.get("answer_position", "none")).strip().lower()
                    if answer_position not in {"subject", "value", "none"}:
                        answer_position = "none"
                    alignment_projection_votes[candidate.node_id].append(answer_position)
                    alignment_audits[candidate.node_id].append({
                        "pass": sample_index + 1,
                        "mode": "independent_query_constraint_only",
                        "raw": alignment,
                        "answer_position": answer_position,
                    })
        self.last_diagnostics = {
            **self.last_diagnostics,
            "independent_passes_requested": sample_count,
            "independent_passes_completed": evidence_calls,
            "query_alignment_passes_requested": (
                sample_count if self.config.query_conditioned_semantic_alignment else 0
            ),
            "query_alignment_passes_completed": alignment_calls,
            "query_alignment_certificates_completed": (
                1 if controller_alignment else 0
            ),
        }
        if self.config.query_conditioned_semantic_alignment:
            for candidate in candidates:
                base = aggregated[candidate.node_id]
                alignment = {
                    name: mean(row[name] for row in alignment_samples[candidate.node_id])
                    for name in ALIGNMENT_FIELDS
                }
                alignment_position = _projection_vote(
                    alignment_projection_votes[candidate.node_id],
                    _projection_vote(
                        projection_votes[candidate.node_id],
                        _preserved_projection(candidate),
                    ),
                )
                aggregated[candidate.node_id] = _query_conditioned_signals(
                    VerificationSignals(
                        grounding=base.grounding,
                        entailment=base.entailment,
                        type_match=base.type_match,
                        dependency_consistency=base.dependency_consistency,
                        retrieval_support=base.retrieval_support,
                        contradiction_risk=base.contradiction_risk,
                        raw_model_confidence=base.raw_model_confidence,
                        **alignment,
                        reasons=base.reasons,
                    ),
                    candidate, graph, subgoal_id, alignment_position,
                    structural_dependency=self.config.structural_dependency_binding_coverage,
                    controller_certificate=controller_alignment,
                )
        for candidate in comparison_candidates:
            aggregated.setdefault(candidate.node_id, candidate.score.raw)
        expected_type = graph.node(subgoal_id, SubgoalNode).answer_type
        projection_by_id = {}
        for candidate in comparison_candidates:
            position = _projection_vote(
                (
                    alignment_projection_votes[candidate.node_id]
                    if controller_alignment
                    and alignment_projection_votes[candidate.node_id]
                    else projection_votes[candidate.node_id]
                ),
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
                    "independent_passes_completed": evidence_calls,
                    "query_alignment_passes_requested": (
                        sample_count if self.config.query_conditioned_semantic_alignment else 0
                    ),
                    "query_alignment_passes_completed": alignment_calls,
                    "query_alignment_certificates_completed": (
                        1 if controller_alignment else 0
                    ),
                    "aggregation": "componentwise_arithmetic_mean_before_fusion",
                    "comparison_group": (
                        "slot_answer_alternatives"
                        if candidate.node_id in answer_ids else "non_answer_relational_claims"
                    ),
                    "passes": audits[candidate.node_id] or [{
                        "mode": "preserved_prior_independent_raw_scores",
                        "raw": candidate.score.raw.__dict__,
                    }],
                    "query_alignment_passes": alignment_audits[candidate.node_id],
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
        *ALIGNMENT_FIELDS, "full_subgoal_coverage",
    )
    return VerificationSignals(
        **{name: mean(float(getattr(row, name)) for row in rows) for name in names},
        reasons=list(dict.fromkeys(reason for row in rows for reason in row.reasons))[:8],
    )


def _evidence_endpoint_grounding(
    candidate: ClaimNode,
    graph: DynamicReasoningHypergraphV2,
) -> float:
    """Audit an extracted dependent tuple against only its cited evidence.

    This is deliberately a raw grounding channel, not a fused relevance score.
    Claims without dependencies retain the existing exact-span audit.  Derived
    JOIN claims have no extraction span and are governed by proof-leaf closure.
    """
    spans = [
        str(value).strip()
        for value in candidate.provenance.metadata.get("source_spans", [])
        if str(value).strip()
    ]
    if not spans or not candidate.dependency_claim_ids:
        return 1.0
    answer_text = normalize_text(" ".join(spans))
    answer_endpoint = normalize_text(candidate.value)
    if not answer_endpoint or answer_endpoint not in answer_text:
        return 0.0
    cited_evidence = [
        graph.node(node_id, EvidenceNode)
        for node_id in candidate.evidence_refs if node_id in graph.nodes
    ]
    binding_text = normalize_text(" ".join(
        [*spans]
        + [node.title for node in cited_evidence]
        + [node.source_span for node in cited_evidence]
    ))

    return float(_lexical_endpoint_anchored(candidate.subject, binding_text))


def _lexical_endpoint_anchored(endpoint: str, normalized_evidence: str) -> bool:
    normalized = normalize_text(endpoint)
    if not normalized:
        return False
    if normalized in normalized_evidence:
        return True
    generic = {
        "city", "country", "county", "district", "division", "region",
        "state", "province", "territory", "area", "part", "true", "false",
    }
    tokens = {
        token for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 4 and token not in generic
    }
    evidence_tokens = set(re.findall(r"[a-z0-9]+", normalized_evidence))
    return bool(tokens.intersection(evidence_tokens))


def _generic_evidence_endpoint_grounding(
    candidate: ClaimNode,
    graph: DynamicReasoningHypergraphV2,
) -> float:
    spans = [
        str(value).strip()
        for value in candidate.provenance.metadata.get("source_spans", [])
        if str(value).strip()
    ]
    if not spans or not candidate.dependency_claim_ids:
        return 1.0
    generic_cue = re.search(
        r"\b(?:all|any|each|every|generally|typically|usually)\b",
        " ".join(spans),
        re.IGNORECASE,
    )
    if generic_cue is None:
        return 1.0
    return _evidence_endpoint_grounding(candidate, graph)


def _controller_query_alignment_certificates(
    candidates: list[ClaimNode],
    graph: DynamicReasoningHypergraphV2,
    subgoal_id: str,
    evidence_profiles: dict[str, VerificationSignals],
) -> dict[str, dict[str, Any]]:
    """Certify query satisfaction without a second provider judgment.

    The certificate reads only the compiled query constraint, typed tuple
    endpoints, controller-owned dependency lineage and independently produced
    evidence-channel scores.  Evidence scores are used solely to decide which
    *other* tuple edges may carry a binding; they never become an alignment
    value.  Excluding the target tuple from its own reachability proof prevents
    circular subject binding.
    """
    constraint = next((
        row for row in graph.query_graph.get("constraints", [])
        if str(row.get("subgoal_id")) == subgoal_id
    ), {})
    description = str(constraint.get("description") or graph.node(
        subgoal_id, SubgoalNode,
    ).instantiated_question)
    expected_type = graph.node(subgoal_id, SubgoalNode).answer_type
    base_anchors = {
        normalize_text(str(value))
        for value in constraint.get("known_entities", [])
        if normalize_text(str(value))
    }
    subgoal = graph.node(subgoal_id, SubgoalNode)
    branch_ids = {candidate.branch_id for candidate in candidates}
    for branch_id in branch_ids:
        branch = graph.branches.get(branch_id)
        if branch is None:
            continue
        for dependency_id in subgoal.dependencies:
            claim = graph.nodes.get(str(branch.assignments.get(dependency_id, "")))
            if isinstance(claim, ClaimNode) and normalize_text(claim.value):
                base_anchors.add(normalize_text(claim.value))
    for candidate in candidates:
        for dependency_id in candidate.dependency_claim_ids:
            dependency = graph.nodes.get(dependency_id)
            if isinstance(dependency, ClaimNode) and normalize_text(dependency.value):
                base_anchors.add(normalize_text(dependency.value))

    result: dict[str, dict[str, Any]] = {}
    for target in candidates:
        reachable = set(base_anchors)
        target_endpoint_pair = frozenset(filter(None, (
            normalize_text(target.subject), normalize_text(target.value),
        )))
        excluded_parallel_edges = sum(
            edge.node_id != target.node_id
            and frozenset(filter(None, (
                normalize_text(edge.subject), normalize_text(edge.value),
            ))) == target_endpoint_pair
            for edge in candidates
        )
        changed = True
        while changed:
            changed = False
            for edge in candidates:
                edge_endpoint_pair = frozenset(filter(None, (
                    normalize_text(edge.subject), normalize_text(edge.value),
                )))
                if (
                    edge.node_id == target.node_id
                    or edge_endpoint_pair == target_endpoint_pair
                    or not edge.evidence_refs
                ):
                    continue
                raw = evidence_profiles.get(edge.node_id, edge.score.raw)
                if (
                    raw.grounding < 0.70
                    or raw.entailment < 0.70
                    or raw.contradiction_risk >= 0.70
                ):
                    continue
                subject = normalize_text(edge.subject)
                value = normalize_text(edge.value)
                if not subject or not value:
                    continue
                if _endpoint_in_anchors(subject, reachable) and not _endpoint_in_anchors(
                    value, reachable,
                ):
                    reachable.add(value)
                    changed = True
                elif _endpoint_in_anchors(value, reachable) and not _endpoint_in_anchors(
                    subject, reachable,
                ):
                    reachable.add(subject)
                    changed = True

        subject_bound = _endpoint_in_anchors(target.subject, reachable)
        value_bound = _endpoint_in_anchors(target.value, reachable)
        semantics = graph.claim_semantics[target.node_id]
        relation = _relation_target_certificate(
            description, target.relation, expected_type,
            known_entities=constraint.get("known_entities", []),
        )
        answer_position = "none"
        if subject_bound != value_bound:
            if subject_bound and _projection_type_compatible(
                semantics.value_type, expected_type, numeric_aliases=True,
            ):
                answer_position = "value"
            elif value_bound and _projection_type_compatible(
                semantics.subject_type, expected_type, numeric_aliases=True,
            ):
                answer_position = "subject"
        elif (
            subject_bound and value_bound
            and normalize_text(target.subject) == normalize_text(target.value)
            and relation >= 1.0 - 1e-9
            and _projection_type_compatible(
                semantics.value_type, expected_type, numeric_aliases=True,
            )
        ):
            # A reflexive relation may legitimately map a bound input entity to
            # the same entity in the typed output slot.  Endpoint identity plus
            # a certified target relation is sufficient; no extraction label or
            # answer value is consulted.
            answer_position = "value"
        if answer_position == "none" and (
            subject_bound and not value_bound
            and relation >= 1.0 - 1e-9
            and _projection_type_compatible(
                semantics.subject_type, expected_type, numeric_aliases=True,
            )
            and _inverse_bound_output_role_certificate(
                target.relation, target.value, description,
            )
        ):
            # Some relational questions name the output country as a bound
            # descriptor (e.g. an organization *of* that country) and ask for
            # the inverse affiliation.  A typed possession/affiliation edge and
            # an explicit query-role match certify the subject slot without
            # trusting an extraction answer label.
            answer_position = "subject"
        subject_binding = float(
            subject_bound != value_bound or answer_position != "none"
        )
        output_slot = float(answer_position != "none")
        dependency = _structural_dependency_binding_coverage(
            target, graph, subgoal_id,
        )
        dependency_coverage = 0.0 if dependency is None else dependency
        qualifier = _qualifier_certificate(
            tuple(str(value) for value in constraint.get("required_qualifiers", [])),
            target, semantics, answer_position,
        )
        result[target.node_id] = {
            "relation_target_alignment": relation,
            "subject_binding_coverage": subject_binding,
            "dependency_binding_coverage": dependency_coverage,
            "qualifier_coverage": qualifier,
            "output_slot_coverage": output_slot,
            "answer_position": answer_position,
            "certificate": {
                "version": "controller-query-constraint-v1",
                "relation_intent": _relation_intent(description),
                "candidate_relation_concepts": sorted(
                    _candidate_relation_concepts(target.relation)
                ),
                "reachable_anchor_count_excluding_target": len(reachable),
                "excluded_parallel_tuple_edges": excluded_parallel_edges,
                "subject_bound": subject_bound,
                "value_bound": value_bound,
                "required_qualifiers": list(
                    constraint.get("required_qualifiers", [])
                ),
            },
        }
    return result


def _endpoint_in_anchors(endpoint: str, anchors: set[str]) -> bool:
    value = normalize_text(endpoint).removesuffix(" s").strip()
    if not value:
        return False
    for anchor in anchors:
        normalized = normalize_text(anchor).removesuffix(" s").strip()
        if not normalized:
            continue
        if value == normalized:
            return True
        if (
            " " in value and len(value) >= 6
            and normalized == f"{value}s"
        ) or (
            " " in normalized and len(normalized) >= 6
            and value == f"{normalized}s"
        ):
            return True
        # Possessive normalization and harmless title decoration are common in
        # planner anchors.  Containment is allowed only for a multi-token name.
        shorter, longer = sorted((value, normalized), key=len)
        if len(shorter) >= 6 and " " in shorter and re.search(
            rf"\b{re.escape(shorter)}\b", longer,
        ):
            return True
    return False


_RELATION_INTENT_PATTERNS = (
    ("birth_date", r"\b(?:birth date|date of birth)\b|\bwhen\b[^?]*\bborn\b"),
    ("border", r"\b(?:share[sd]? (?:a )?border|border(?:s|ed|ing)? with|adjoin|adjacent)\b"),
    ("empty", r"\b(?:empt(?:y|ies|ied|ying)|flows? into|mouth|discharg(?:e|es|ed))\b"),
    ("capital", r"\b(?:became?|become|made) (?:the )?capital\b|\bcapital of\b"),
    ("location", r"\b(?:located? in|lies? in|contains?|within|part of)\b"),
    ("citizenship", r"\b(?:citizenship|nationality)\b"),
    ("literature", r"\b(?:country|place) of literature\b|\bliterature\b"),
    ("production", r"\b(?:produc(?:e|ed|er|ers|ing|tion)|made)\b"),
    ("proximity", r"\b(?:river is by|river by|nearby river|has (?:a )?river)\b"),
    ("arrival", r"\b(?:came?|come|arriv(?:e|ed|al)|settle[dr]?)\b"),
    ("origin", r"\b(?:is|are|was|were) from\b|\bfrom what (?:country|state|place)\b"),
    ("perform", r"\b(?:perform(?:ed|er|ers|ing|s)?|sang|singer|artist)\b"),
    ("birth_place", r"\b(?:birthplace|place of birth)\b|\bwhere\b[^?]*\bborn\b"),
)


def _relation_intent(description: str) -> str:
    normalized = normalize_text(description)
    for name, pattern in _RELATION_INTENT_PATTERNS:
        if re.search(pattern, normalized):
            return name
    return ""


def _candidate_relation_concepts(relation: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(relation))
    text = normalize_text(re.sub(r"[_:\-/]+", " ", text))
    concepts = set()
    patterns = {
        "birth_date": r"\b(?:birth date|date of birth|born on)\b",
        "birth_place": r"\b(?:birth place|birthplace|born in)\b",
        "border": r"\b(?:border|adjoin|adjacent|neighbor|neighbour)\w*\b",
        "perform": r"\b(?:perform|sing|sang|artist|record)\w*\b",
        "citizenship": r"\b(?:citizen|nationality)\w*\b",
        "literature": r"\bliterature\b",
        "production": r"\b(?:produce|produced|production|made)\b",
        "empty": r"\b(?:empty|empties|flow|mouth|discharge)\w*\b",
        "proximity": r"\b(?:is by|has river|near|beside)\b",
        "capital": r"\bcapital\b",
        "arrival": r"\b(?:came|come|arrive|settle)\w*\b",
        "location": r"\b(?:located|lies|contain|within|part of)\w*\b",
        "origin": r"\b(?:from|origin|belong to|has [a-z0-9 ]*(?:troop|guard))\w*\b",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, text):
            concepts.add(name)
    return concepts


def _relation_target_certificate(
    description: str,
    relation: str,
    expected_type: str,
    *,
    known_entities: Any,
) -> float:
    intent = _relation_intent(description)
    concepts = _candidate_relation_concepts(relation)
    if intent:
        return float(intent in concepts)

    # Auditable lexical fallback for relations outside the compact ontology.
    # Entity and output-type words are removed so a type label cannot satisfy
    # the requested predicate (the failure exposed by the causal diagnostic).
    excluded = {
        token for value in known_entities
        for token in _relation_tokens(str(value))
    }
    excluded.update(_relation_tokens(expected_type))
    excluded.update({
        "what", "which", "who", "where", "when", "how", "the", "a", "an",
        "of", "to", "in", "on", "for", "with", "does", "did", "is", "are",
        "was", "were", "entity", "thing", "person", "country", "city", "state",
    })
    query_tokens = _relation_tokens(description) - excluded
    relation_tokens = _relation_tokens(relation) - {"inverse", "of"}
    return float(bool(query_tokens & relation_tokens))


def _inverse_bound_output_role_certificate(
    relation: str, unbound_endpoint: str, description: str,
) -> bool:
    relation_text = normalize_text(re.sub(r"[_:\-/]+", " ", str(relation)))
    if not (
        relation_text.startswith("has ")
        or " belong " in f" {relation_text} "
        or relation_text.startswith("inverse of ")
    ):
        return False
    endpoint_tokens = _relation_tokens(unbound_endpoint) - {
        "the", "group", "organization", "entity", "gdr",
    }
    question_tokens = _relation_tokens(description)
    return bool(endpoint_tokens & question_tokens)


def _relation_tokens(text: str) -> set[str]:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    value = normalize_text(re.sub(r"[_:\-/]+", " ", value))
    aliases = {
        "performed": "perform", "performer": "perform", "performs": "perform",
        "located": "locate", "location": "locate", "contains": "contain",
        "produced": "produce", "producer": "produce", "empties": "empty",
        "bordering": "border", "borders": "border", "settled": "settle",
        "arrived": "arrive", "literary": "literature",
    }
    return {
        aliases.get(token, token)
        for token in re.findall(r"[a-z0-9]+", value)
        if len(token) >= 3
    }


def _qualifier_certificate(
    required: tuple[str, ...],
    candidate: ClaimNode,
    semantics: Any,
    answer_position: str,
) -> float:
    if not required:
        return 1.0
    selected_type = ""
    if answer_position == "subject":
        selected_type = semantics.subject_type
    elif answer_position == "value":
        selected_type = semantics.value_type
    relation_tokens = _relation_tokens(candidate.relation)
    qualifier_metadata = {
        normalize_text(str(key))
        for key, value in (semantics.qualifiers or {}).items() if value
    }
    checks = []
    for family in required:
        if family == "temporal":
            checks.append(_projection_type_compatible(
                selected_type, "date", numeric_aliases=True,
            ))
        elif family == "cardinality":
            checks.append(_projection_type_compatible(
                selected_type, "number", numeric_aliases=True,
            ))
        elif family == "comparison":
            checks.append(bool(relation_tokens & {
                "more", "less", "greater", "largest", "smallest", "higher",
                "lower", "older", "younger", "compare",
            }) or "comparison" in qualifier_metadata)
        else:
            checks.append(family in qualifier_metadata)
    return float(all(checks))


def _query_conditioned_signals(
    raw: VerificationSignals,
    candidate: ClaimNode,
    graph: DynamicReasoningHypergraphV2,
    subgoal_id: str,
    answer_position: str,
    *,
    structural_dependency: bool,
    controller_certificate: bool = False,
) -> VerificationSignals:
    """Keep evidence truth and query satisfaction as independent raw channels.

    Only controller-provable binding facts override a model residual. Relation
    and qualifier semantics remain independent and never enter absolute support.
    """
    dependency = _structural_dependency_binding_coverage(
        candidate, graph, subgoal_id,
    )
    dependency_coverage = raw.dependency_binding_coverage
    if not controller_certificate:
        dependency_coverage = (
            dependency if dependency is not None
            else raw.dependency_binding_coverage
        )
    dependency_consistency = raw.dependency_consistency
    reasons = list(raw.reasons)
    if (
        structural_dependency and dependency is not None
        and raw.grounding >= 1.0 - 1e-9
    ):
        dependency_consistency = dependency
        reasons.append("controller_structural_dependency_binding")

    subject_binding = raw.subject_binding_coverage
    if not controller_certificate:
        subject_binding = _structural_subject_binding_coverage(
            candidate, graph, subgoal_id, answer_position,
        )
        if subject_binding is None:
            subject_binding = raw.subject_binding_coverage
        else:
            reasons.append("controller_structural_subject_binding")

    expected_type = graph.node(subgoal_id, SubgoalNode).answer_type
    structural_projection = _structural_projection(
        graph, subgoal_id, candidate, expected_type,
        numeric_aliases=True,
    )
    output_slot = raw.output_slot_coverage
    if not controller_certificate and structural_projection != "none":
        output_slot = 1.0 if answer_position in {"none", structural_projection} else 0.0
        reasons.append("controller_structural_output_slot")

    constraint = next((
        row for row in graph.query_graph.get("constraints", [])
        if str(row.get("subgoal_id")) == subgoal_id
    ), {})
    qualifier = raw.qualifier_coverage
    if not controller_certificate and not constraint.get("required_qualifiers"):
        qualifier = 1.0
    components = {
        "relation_target_alignment": raw.relation_target_alignment,
        "subject_binding_coverage": subject_binding,
        "dependency_binding_coverage": dependency_coverage,
        "qualifier_coverage": qualifier,
        "output_slot_coverage": output_slot,
    }
    return VerificationSignals(
        grounding=raw.grounding,
        entailment=raw.entailment,
        type_match=raw.type_match,
        dependency_consistency=_unit(dependency_consistency),
        retrieval_support=raw.retrieval_support,
        contradiction_risk=raw.contradiction_risk,
        raw_model_confidence=raw.raw_model_confidence,
        **{name: _unit(value) for name, value in components.items()},
        full_subgoal_coverage=min(_unit(value) for value in components.values()),
        reasons=list(dict.fromkeys(reasons))[:8],
    )


def _structural_dependency_binding_coverage(
    candidate: ClaimNode,
    graph: DynamicReasoningHypergraphV2,
    subgoal_id: str,
) -> float | None:
    """Prove declared bridge use from controller-owned lineage and endpoints."""
    subgoal = graph.node(subgoal_id, SubgoalNode)
    if not subgoal.dependencies:
        return 1.0
    dependency_claims = [
        graph.nodes.get(node_id) for node_id in candidate.dependency_claim_ids
    ]
    dependency_claims = [
        row for row in dependency_claims if isinstance(row, ClaimNode)
    ]
    if not dependency_claims:
        return 0.0
    endpoints = {normalize_text(candidate.subject), normalize_text(candidate.value)}
    endpoints.discard("")
    covered = []
    for required_subgoal in subgoal.dependencies:
        matching = [
            dependency for dependency in dependency_claims
            if dependency.target_subgoal == required_subgoal
        ]
        covered.append(float(any(
            normalize_text(dependency.value) in endpoints
            for dependency in matching
            if normalize_text(dependency.value)
        )))
    return min(covered) if covered else None


def _structural_subject_binding_coverage(
    candidate: ClaimNode,
    graph: DynamicReasoningHypergraphV2,
    subgoal_id: str,
    answer_position: str,
) -> float | None:
    query_anchors = {
        normalize_text(str(value))
        for row in graph.query_graph.get("constraints", [])
        if str(row.get("subgoal_id")) == subgoal_id
        for value in row.get("known_entities", [])
        if normalize_text(str(value))
    }
    dependency_anchors = set()
    for dependency_id in candidate.dependency_claim_ids:
        dependency = graph.nodes.get(dependency_id)
        if isinstance(dependency, ClaimNode):
            dependency_anchors.update(filter(None, (
                normalize_text(dependency.value),
            )))
    subgoal = graph.node(subgoal_id, SubgoalNode)
    # A dependent subgoal is bound to the bridge output, not merely to any
    # entity mentioned by the original question.
    anchors = dependency_anchors if subgoal.dependencies else query_anchors
    if not anchors:
        return None
    bound_endpoint = (
        candidate.value if answer_position == "subject" else candidate.subject
    )
    if answer_position == "none":
        return float(bool({
            normalize_text(candidate.subject), normalize_text(candidate.value),
        } & anchors))
    return float(normalize_text(bound_endpoint) in anchors)


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
        "administrative_district": "location", "administrative_entity": "location",
        "administrative_territorial_entity": "location", "municipality": "location",
        "region": "location",
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
    # A freshly extracted dependent claim may be verified before the branch
    # assignment transition materializes. Its controller-recorded lineage is
    # already sufficient to expose the bound bridge endpoint.
    for dependency_id in candidate.dependency_claim_ids:
        dependency = graph.nodes.get(dependency_id)
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
        *ALIGNMENT_FIELDS, "full_subgoal_coverage",
    )
    return VerificationSignals(
        **{name: max(float(getattr(row, name)) for row in rows) for name in names},
        contradiction_risk=max(float(row.contradiction_risk) for row in rows),
        reasons=list(dict.fromkeys(reason for row in rows for reason in row.reasons))[:8],
    )
