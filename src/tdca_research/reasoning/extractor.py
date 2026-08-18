from __future__ import annotations

from ..budget import Budget
from ..llm import BaseLLM
from ..models import Claim, ClaimStatus, ReasoningSlot, RetrievalHit
from ..utils import estimate_message_tokens, evidence_context, normalize_text


SYSTEM = """Extract one evidence-grounded answer candidate for the subquestion. Return JSON only with:
answer, subject, relation, answer_type, source_document_ids, source_spans, confidence.
Use only the supplied passages. A source span must be a short verbatim substring of a supplied passage.
If evidence is insufficient, set answer to an empty string and confidence to 0. Do not solve a different question.
Interpret terse or relation-style subquestions in the context of the root question. Return the shortest answer
that completely satisfies the requested relation. Preserve complete dates, ranges, names and units when asked.
Keep source_spans short (at most 240 characters each)."""


class ClaimExtractor:
    def __init__(self, llm: BaseLLM, budget: Budget, max_tokens: int, context_chars: int, temperature: float = 0.0, compaction: str = "none") -> None:
        self.llm = llm
        self.budget = budget
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_chars = context_chars
        self.compaction = compaction

    def extract(self, slot: ReasoningSlot, bound_question: str, hits: list[RetrievalHit], dependency_claim_ids: list[str], step: int, root_question: str = "") -> Claim | None:
        context = evidence_context(hits, bound_question, self.context_chars, self.compaction)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Root question: {root_question or bound_question}\nCurrent subquestion: {bound_question}\nExpected type: {slot.answer_type}\n\nPassages:\n{context}"},
        ]
        self.budget.require(self.max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages))
        data, generation = self.llm.generate_json(
            messages,
            schema_name="claim_candidate_v1",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self.budget.record_generation(generation)
        answer = str(data.get("answer", "")).strip()
        if not answer:
            return None
        available_ids = {hit.passage.passage_id for hit in hits}
        source_ids = [str(value) for value in data.get("source_document_ids", []) if str(value) in available_ids]
        # Grounding must be against the exact bounded prompt view, not hidden
        # suffixes of retrieved passages that the model never received.
        visible_context = normalize_text(context)
        source_spans = [
            str(value).strip() for value in data.get("source_spans", [])
            if str(value).strip() and normalize_text(str(value)) in visible_context
        ]
        if not source_spans:
            source_spans = self._recover_answer_span(answer, source_ids, hits, visible_context)
        retrieval = max((hit.raw_score for hit in hits if hit.passage.passage_id in source_ids), default=0.0)
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        return Claim(
            claim_id=f"claim_{step}_{slot.slot_id}",
            subject=str(data.get("subject", "")).strip(),
            relation=str(data.get("relation", "")).strip(),
            object=answer,
            answer_type=str(data.get("answer_type", slot.answer_type)).strip() or slot.answer_type,
            target_slot=slot.slot_id,
            source_document_ids=source_ids,
            source_spans=source_spans,
            depends_on_claim_ids=dependency_claim_ids,
            retrieval_score=retrieval,
            calibrated_confidence=confidence,
            status=ClaimStatus.PROPOSED,
            created_step=step,
        )

    @staticmethod
    def _recover_answer_span(answer: str, source_ids: list[str], hits: list[RetrievalHit], visible_context: str) -> list[str]:
        """Recover an exact visible sentence when the model's span is malformed.

        This never invents evidence: the answer and returned span must both occur in
        the bounded prompt view and the declared source document.
        """
        import re

        normalized_answer = normalize_text(answer)
        if not normalized_answer or normalized_answer not in visible_context:
            return []
        wanted = set(source_ids)
        for hit in hits:
            if hit.passage.passage_id not in wanted:
                continue
            blocks = [hit.passage.title] + re.split(r"(?<=[.!?])\s+|\n+", hit.passage.text)
            for block in blocks:
                value = block.strip()
                if normalized_answer in normalize_text(value) and normalize_text(value) in visible_context:
                    return [value[:500]]
        return []
