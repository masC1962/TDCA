from __future__ import annotations

import re

from ..budget import Budget
from ..llm import BaseLLM
from ..models import Claim, ClaimStatus, ReasoningSlot, RetrievalHit
from ..utils import estimate_message_tokens, evidence_context, normalize_text


SYSTEM = """Verify a candidate claim against supplied evidence. Return JSON only with:
evidence_relevance, relation_entailment, answer_type_match, dependency_consistency,
contradiction_detected, confidence, reasons.
Each numeric score is in [0,1]. Reject claims unsupported by a source span even when plausible from prior knowledge.
Judge the current relation in the context of the root question, but never use outside knowledge.
The reasons field must contain at most 3 short machine-readable codes, never prose."""


def normalize_typed_answer(answer: str, answer_type: str, question: str = "") -> str:
    value = " ".join((answer or "").strip().split())
    kind = (answer_type or "").lower()
    if kind in {"boolean", "yes_no"}:
        normalized = normalize_text(value)
        if normalized.startswith("yes"):
            return "yes"
        if normalized.startswith("no"):
            return "no"
    if kind == "decade" or "decade" in question.lower():
        match = re.search(r"\b(1\d{3}|20\d{2})\b", value)
        if match:
            year = int(match.group(1))
            return f"{year // 10 * 10}s"
    if kind == "year":
        match = re.search(r"\b(1\d{3}|20\d{2})\b", value)
        if match:
            return match.group(1)
    return value.strip(" .")


class ClaimVerifier:
    def __init__(self, llm: BaseLLM, budget: Budget, max_tokens: int, context_chars: int, minimum: float = 0.55, temperature: float = 0.0, compaction: str = "none") -> None:
        self.llm = llm
        self.budget = budget
        self.max_tokens = max_tokens
        self.minimum = minimum
        self.temperature = temperature
        self.context_chars = context_chars
        self.compaction = compaction

    @staticmethod
    def _spans_are_grounded(claim: Claim, hits: list[RetrievalHit]) -> bool:
        documents = {
            hit.passage.passage_id: f"{hit.passage.title}\n{hit.passage.text}"
            for hit in hits
        }
        if not claim.source_document_ids or not claim.source_spans:
            return False
        return all(
            any(normalize_text(span) in normalize_text(documents.get(document_id, "")) for document_id in claim.source_document_ids)
            for span in claim.source_spans
        )

    def verify(
        self, claim: Claim, slot: ReasoningSlot, bound_question: str,
        hits: list[RetrievalHit], dependencies_complete: bool,
        root_question: str = "", dependency_claims: list[Claim] | None = None,
    ) -> tuple[Claim, list[str]]:
        reasons: list[str] = []
        if not self._spans_are_grounded(claim, hits):
            claim.status = ClaimStatus.REJECTED
            claim.entailment_score = 0.0
            claim.type_score = 0.0
            claim.calibrated_confidence = 0.0
            return claim, ["source_span_not_grounded"]
        context = evidence_context(hits, bound_question, self.context_chars, self.compaction)
        dependency_text = "\n".join(
            f"- {dependency.claim_id}: ({dependency.subject}, {dependency.relation}, {dependency.object})"
            for dependency in (dependency_claims or [])
        ) or "(none)"
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Root question: {root_question or bound_question}\nCurrent subquestion: {bound_question}\nExpected type: {slot.answer_type}\nCandidate: {claim.object}\nDependencies complete: {dependencies_complete}\nVerified dependency claims:\n{dependency_text}\nEvidence:\n{context}"},
        ]
        self.budget.require(self.max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages))
        data, generation = self.llm.generate_json(
            messages,
            schema_name="claim_verification_v1",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self.budget.record_generation(generation)
        relevance = _unit(data.get("evidence_relevance"))
        entailment = _unit(data.get("relation_entailment"))
        type_score = _unit(data.get("answer_type_match"))
        if slot.answer_type.lower() == "entity" and claim.object.strip():
            # Entity is the deliberately broad fallback type. A non-empty answer
            # cannot contradict it; finer types remain independently verified.
            type_score = 1.0
        dependency = _unit(data.get("dependency_consistency")) if dependencies_complete else 0.0
        contradiction = bool(data.get("contradiction_detected", False))
        model_confidence = _unit(data.get("confidence"))
        # Absolute confidence is a conservative product/geometric blend, not a query-relative rank.
        calibrated = (relevance * entailment * type_score * max(0.01, dependency)) ** 0.25
        calibrated = min(calibrated, model_confidence)
        claim.entailment_score = entailment
        claim.type_score = type_score
        claim.calibrated_confidence = calibrated
        claim.object = normalize_typed_answer(claim.object, slot.answer_type, bound_question)
        if contradiction:
            reasons.append("contradiction_detected")
        if relevance < self.minimum:
            reasons.append("low_evidence_relevance")
        if entailment < self.minimum:
            reasons.append("low_relation_entailment")
        if type_score < self.minimum:
            reasons.append("answer_type_mismatch")
        if dependency < self.minimum:
            reasons.append("dependency_inconsistent")
        if calibrated < self.minimum:
            reasons.append("low_calibrated_confidence")
        claim.status = ClaimStatus.REJECTED if reasons else ClaimStatus.VERIFIED
        return claim, reasons


def _unit(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
