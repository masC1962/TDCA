from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from ..budget import Budget
from ..llm import BaseLLM
from ..utils import bounded_context, estimate_message_tokens, normalize_text
from .config import DynamicResearchConfig
from .graph import (
    CandidateStatus,
    ClaimNode,
    DynamicReasoningHypergraph,
    EvidenceNode,
    GraphOperation,
    OperationType,
    SubgoalNode,
    VerificationSignals,
)
from .scoring import fuse_candidate_scores, retrieval_support_from_ranks


EXTRACT_SYSTEM = """Generate a small set of independently plausible evidence-grounded candidates for one
subgoal. Return JSON only as {candidates:[...]}. Each candidate has answer, subject, relation, answer_type,
evidence_ids, source_spans, extraction_confidence. Use only supplied evidence. Every source span must be a
short verbatim substring of the referenced evidence. Return at most the requested candidate cap. Do not rank
candidates against each other and do not use prior knowledge. Return an empty list if evidence is insufficient."""


VERIFY_SYSTEM = """Score each supplied candidate independently against only its referenced evidence and the
stated dependency claims. Return JSON only as {scores:[...]}. For every candidate return candidate_id,
grounding, entailment, type_match, dependency_consistency, contradiction_risk, raw_model_confidence, reasons.
All scores are in [0,1]. Do not rank, normalize, compare, prune, commit, or omit candidates. The same raw score
must mean the same degree of support regardless of other candidates in the batch. Reasons must contain at
most three short machine-readable codes, never explanatory prose."""


class DynamicCandidateGenerator:
    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config
        self.last_diagnostics: dict[str, Any] = {}

    def propose(
        self, graph: DynamicReasoningHypergraph, subgoal_id: str, branch_id: str,
        instantiated_question: str, dependency_claim_ids: list[str], operation_id: str,
    ) -> GraphOperation | None:
        evidence = graph.evidence(subgoal_id, branch_id)
        if not evidence:
            self.last_diagnostics = {"raw_candidate_count": 0, "accepted_count": 0, "rejections": {"no_evidence": 1}}
            return None
        context = bounded_context([
            f"[{node.node_id}] doc={node.document_id} title={node.title}\n{node.source_span}"
            for node in evidence
        ], self.config.evidence_char_budget)
        dependency_text = "\n".join(
            f"[{claim_id}] ({graph.node(claim_id, ClaimNode).subject}, "
            f"{graph.node(claim_id, ClaimNode).relation}, {graph.node(claim_id, ClaimNode).value})"
            for claim_id in dependency_claim_ids
        ) or "(none)"
        expected_type = graph.node(subgoal_id, SubgoalNode).answer_type
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": (
                f"Root question: {graph.question}\nSubgoal: {instantiated_question}\n"
                f"Expected answer type: {expected_type}\n"
                f"Candidate cap: {self._candidate_cap()}\n"
                f"Dependency claims:\n{dependency_text}\nEvidence:\n{context}"
            )},
        ]
        self.budget.require(
            self.config.candidate_set_max_tokens,
            estimated_prompt_tokens=estimate_message_tokens(messages),
        )
        data, generation = self.llm.generate_json(
            messages, "dynamic_candidate_set_v1", self.config.candidate_set_max_tokens,
            self.config.temperature,
        )
        self.budget.record_generation(generation)
        available = {node.node_id: node for node in evidence}
        rows = []
        seen_values = set()
        raw_candidates = data.get("candidates", []) if isinstance(data.get("candidates"), list) else []
        rejection_counts: dict[str, int] = defaultdict(int)
        for index, raw in enumerate(raw_candidates[: self._candidate_cap()], start=1):
            if not isinstance(raw, dict):
                rejection_counts["non_object"] += 1
                continue
            answer = str(raw.get("answer", "")).strip()
            normalized = normalize_text(answer)
            evidence_ids = [str(value) for value in raw.get("evidence_ids", []) if str(value) in available]
            spans = [str(value).strip() for value in raw.get("source_spans", []) if str(value).strip()]
            grounded_spans = [
                span for span in spans
                if any(normalize_text(span) in normalize_text(available[evidence_id].source_span) for evidence_id in evidence_ids)
            ]
            if not answer or not normalized:
                rejection_counts["empty_answer"] += 1
                continue
            if normalized in seen_values:
                rejection_counts["duplicate_answer"] += 1
                continue
            if not evidence_ids:
                rejection_counts["unknown_evidence_id"] += 1
                continue
            if not grounded_spans:
                rejection_counts["ungrounded_source_span"] += 1
                continue
            if not _atomic_typed_answer(answer, expected_type):
                rejection_counts["non_atomic_or_type_shape"] += 1
                continue
            if not _compatible_answer_types(str(raw.get("answer_type", "entity")), expected_type):
                rejection_counts["answer_type_mismatch"] += 1
                continue
            seen_values.add(normalized)
            rows.append({
                "node_id": f"claim_{graph.step + 1}_{subgoal_id}_{index}",
                "value": answer,
                "subject": str(raw.get("subject", "")).strip(),
                "relation": str(raw.get("relation", "")).strip(),
                "answer_type": expected_type,
                "evidence_refs": evidence_ids,
                "source_spans": grounded_spans[:3],
                "dependency_claim_ids": dependency_claim_ids,
                "extraction_confidence": _unit(raw.get("extraction_confidence")),
            })
        self.last_diagnostics = {
            "raw_candidate_count": len(raw_candidates),
            "accepted_count": len(rows),
            "rejections": dict(sorted(rejection_counts.items())),
        }
        if not rows:
            return None
        return GraphOperation(
            operation_id, OperationType.BRANCH, subgoal_id,
            [node.node_id for node in evidence] + dependency_claim_ids,
            branch_id, {"mode": "candidates", "candidates": rows},
            "preserve_plausible_candidates", "dynamic_candidate_generator_v1",
            {"llm_calls": 1.0, "tokens": float(generation.prompt_tokens + generation.completion_tokens)},
        )

    def _candidate_cap(self) -> int:
        return self.config.max_candidates_per_subgoal if self.config.enable_candidate_preservation else 1


class SoftCandidateVerifier:
    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config

    def propose(
        self, graph: DynamicReasoningHypergraph, subgoal_id: str, branch_id: str,
        instantiated_question: str, operation_id: str,
    ) -> tuple[GraphOperation | None, Any]:
        candidates = [
            claim for claim in graph.claims(subgoal_id, branch_id)
            if claim.status == CandidateStatus.PROPOSED
        ]
        if not candidates:
            return None, None
        dependency_ids = list(dict.fromkeys(
            dependency for candidate in candidates for dependency in candidate.dependency_claim_ids
        ))
        dependency_text = "\n".join(
            f"[{claim_id}] ({graph.node(claim_id, ClaimNode).subject}, "
            f"{graph.node(claim_id, ClaimNode).relation}, {graph.node(claim_id, ClaimNode).value}); evidence="
            + " | ".join(
                graph.node(evidence_id, EvidenceNode).source_span
                for evidence_id in graph.node(claim_id, ClaimNode).evidence_refs
            )
            for claim_id in dependency_ids
        ) or "(none)"
        blocks = []
        for candidate in candidates:
            evidence = [graph.node(value, EvidenceNode) for value in candidate.evidence_refs]
            spans = candidate.provenance.metadata.get("source_spans", [])
            blocks.append(
                f"Candidate {candidate.node_id}: value={candidate.value!r}; subject={candidate.subject!r}; "
                f"relation={candidate.relation!r}; expected_type={candidate.answer_type}; "
                f"declared_spans={spans}\n" + "\n".join(
                    f"  Evidence [{node.node_id}] {node.source_span}" for node in evidence
                )
            )
        messages = [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": (
                f"Root question: {graph.question}\nSubgoal: {instantiated_question}\n"
                f"Dependency claims:\n{dependency_text}\n\n" + "\n\n".join(blocks)
            )},
        ]
        self.budget.require(
            self.config.soft_verifier_max_tokens,
            estimated_prompt_tokens=estimate_message_tokens(messages),
        )
        data, generation = self.llm.generate_json(
            messages, "dynamic_soft_verification_v1", self.config.soft_verifier_max_tokens,
            self.config.temperature,
        )
        self.budget.record_generation(generation)
        returned = {
            str(row.get("candidate_id")): row
            for row in data.get("scores", []) if isinstance(row, dict)
        }
        raw_by_id: dict[str, VerificationSignals] = {}
        scoring_audit: dict[str, dict[str, Any]] = {}
        contradiction_groups: dict[tuple[str, str], list[ClaimNode]] = defaultdict(list)
        for candidate in candidates:
            deterministic = _deterministic_raw(candidate, graph)
            row = returned.get(candidate.node_id)
            if row is None:
                model = VerificationSignals(reasons=["missing_verifier_candidate_row"])
                fused = deterministic
                mode = "deterministic_missing_row_fallback"
            else:
                model = VerificationSignals(
                    grounding=_unit(row.get("grounding")),
                    entailment=_unit(row.get("entailment")),
                    type_match=_unit(row.get("type_match")),
                    dependency_consistency=_unit(row.get("dependency_consistency")),
                    retrieval_support=deterministic.retrieval_support,
                    contradiction_risk=_unit(row.get("contradiction_risk")),
                    raw_model_confidence=_unit(row.get("raw_model_confidence")),
                    reasons=[str(value) for value in row.get("reasons", [])][:5],
                )
                weight = self.config.soft_verifier_model_weight
                fused = VerificationSignals(
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
            raw_by_id[candidate.node_id] = fused
            scoring_audit[candidate.node_id] = {
                "mode": mode,
                "model_weight": self.config.soft_verifier_model_weight if row is not None else 0.0,
                "deterministic_raw": deterministic.__dict__,
                "model_raw": model.__dict__,
            }
            contradiction_groups[(normalize_text(candidate.subject), normalize_text(candidate.relation))].append(candidate)
        profiles, summary = fuse_candidate_scores(raw_by_id, self.config)
        scores = {}
        for candidate in candidates:
            profile = profiles[candidate.node_id]
            contradictions = [
                other.node_id for other in contradiction_groups[(normalize_text(candidate.subject), normalize_text(candidate.relation))]
                if other.node_id != candidate.node_id and normalize_text(other.value) != normalize_text(candidate.value)
            ] if candidate.subject and candidate.relation else []
            scores[candidate.node_id] = {
                **profile.raw.__dict__,
                "absolute_support": profile.absolute_support,
                "relative_weight": profile.relative_weight,
                "set_entropy": profile.set_entropy,
                "evidence_gap": profile.evidence_gap,
                "status": "scored",
                "contradiction_links": contradictions,
                "scoring_audit": scoring_audit[candidate.node_id],
            }
        return GraphOperation(
            operation_id, OperationType.VERIFY, subgoal_id,
            [candidate.node_id for candidate in candidates], branch_id,
            {"scores": scores}, "independent_raw_evidence_scoring",
            "dynamic_soft_verifier_v1",
            {"llm_calls": 1.0, "tokens": float(generation.prompt_tokens + generation.completion_tokens)},
        ), summary

    def deterministic_propose(
        self, graph: DynamicReasoningHypergraph, subgoal_id: str, branch_id: str,
        operation_id: str,
    ) -> tuple[GraphOperation | None, Any]:
        """A1--A3 verifier-free scoring from extraction and exact provenance.

        This path intentionally makes no verifier model call.  Its raw components
        remain independent and are fused only after every candidate has been
        scored, preserving the same downstream policy interface as A4+.
        """
        candidates = [
            claim for claim in graph.claims(subgoal_id, branch_id)
            if claim.status == CandidateStatus.PROPOSED
        ]
        if not candidates:
            return None, None
        raw_by_id = {candidate.node_id: _deterministic_raw(candidate, graph) for candidate in candidates}
        profiles, summary = fuse_candidate_scores(raw_by_id, self.config)
        scores = {
            candidate.node_id: {
                **profiles[candidate.node_id].raw.__dict__,
                "absolute_support": profiles[candidate.node_id].absolute_support,
                "relative_weight": profiles[candidate.node_id].relative_weight,
                "set_entropy": profiles[candidate.node_id].set_entropy,
                "evidence_gap": profiles[candidate.node_id].evidence_gap,
                "status": "scored",
                "contradiction_links": [],
            }
            for candidate in candidates
        }
        return GraphOperation(
            operation_id, OperationType.VERIFY, subgoal_id,
            [candidate.node_id for candidate in candidates], branch_id,
            {"scores": scores}, "verifier_ablation_deterministic_raw_scoring",
            "deterministic_candidate_scorer_v1", {"llm_calls": 0.0, "tokens": 0.0},
        ), summary


def _unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _blend(prior: float, model: float, model_weight: float) -> float:
    return (1.0 - model_weight) * prior + model_weight * model


def _deterministic_raw(candidate: ClaimNode, graph: DynamicReasoningHypergraph) -> VerificationSignals:
    evidence = [graph.node(value, EvidenceNode) for value in candidate.evidence_refs]
    spans = [str(value) for value in candidate.provenance.metadata.get("source_spans", [])]
    grounding = float(bool(spans) and all(
        any(normalize_text(span) in normalize_text(node.source_span) for node in evidence)
        for span in spans
    ))
    extraction = candidate.score.raw.raw_model_confidence
    return VerificationSignals(
        grounding=grounding,
        entailment=extraction,
        type_match=float(bool(candidate.value)),
        dependency_consistency=1.0,
        retrieval_support=retrieval_support_from_ranks([node.retrieval_rank for node in evidence]),
        contradiction_risk=0.0,
        raw_model_confidence=extraction,
        reasons=["deterministic_exact_grounding_and_extraction_prior"],
    )


def _atomic_typed_answer(answer: str, answer_type: str) -> bool:
    value = " ".join(answer.split())
    kind = answer_type.lower().strip()
    if not value or len(value.split()) > 16:
        return False
    # Reject lists of alternatives and explanatory prose as candidates. Set/list
    # questions may explicitly opt into a collection answer type.
    if kind not in {"list", "set", "collection"} and (value.count(",") >= 2 or ";" in value):
        return False
    if kind == "year" and not re.fullmatch(r"(?:1\d{3}|20\d{2})", value.strip(" .")):
        return False
    if kind in {"number", "count"} and not re.search(r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b", value.lower()):
        return False
    return True


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


def _dependency_consistency(
    candidate: ClaimNode, dependency_ids: list[str], graph: DynamicReasoningHypergraph,
    question: str, model_score: float,
) -> float:
    if not dependency_ids:
        return 1.0
    repeated = any(
        normalize_text(candidate.value) == normalize_text(graph.node(value, ClaimNode).value)
        for value in dependency_ids
    )
    identity_cues = ("same", "also known", "which entity", "what is the name")
    explicit_alias = any(
        _explicit_parenthetical_alias(graph.node(ref, EvidenceNode).source_span, candidate.value)
        for ref in candidate.evidence_refs
    )
    if repeated and not explicit_alias and not any(cue in question.lower() for cue in identity_cues):
        return min(model_score, 0.1)
    return model_score


def _explicit_parenthetical_alias(text: str, alias: str) -> bool:
    """Recognize only a literal ``Long Form (Alias)`` evidence construction."""
    value = str(alias).strip()
    if not value or len(value.split()) > 8:
        return False
    escaped = r"\s+".join(re.escape(token) for token in value.split())
    long_form = r"[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*){1,7}"
    return re.search(
        rf"\b{long_form}\s*\(\s*{escaped}\s*\)", str(text), re.IGNORECASE,
    ) is not None
