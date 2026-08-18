from __future__ import annotations

from ..budget import Budget
from ..llm import BaseLLM
from ..models import Claim, ReasoningPlan, RetrievalHit
from ..verification import normalize_typed_answer
from ..utils import estimate_message_tokens, evidence_context


SYSTEM = """Select and verify the final short answer using only the supplied verified claim chain and evidence.
Return JSON only with answer, confidence, supported, reasons. Do not replace the candidate with prior knowledge.
If the chain does not answer the root question, set supported=false and answer to an empty string.
The reasons field must contain at most 3 short machine-readable codes, never prose."""


class Finalizer:
    def __init__(self, llm: BaseLLM, budget: Budget, max_tokens: int, context_chars: int, minimum: float, temperature: float = 0.0, compaction: str = "none") -> None:
        self.llm = llm
        self.budget = budget
        self.max_tokens = max_tokens
        self.minimum = minimum
        self.temperature = temperature
        self.context_chars = context_chars
        self.compaction = compaction

    def finalize(self, question: str, plan: ReasoningPlan, terminal: Claim, claims: list[Claim], hits: list[RetrievalHit]) -> tuple[str | None, float, list[str]]:
        claim_text = "\n".join(
            f"- {claim.claim_id}: ({claim.subject}, {claim.relation}, {claim.object}); sources={claim.source_document_ids}"
            for claim in claims
        )
        unique_hits = {hit.passage.passage_id: hit for hit in hits}
        evidence = evidence_context(list(unique_hits.values()), question, self.context_chars, self.compaction)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Root question: {question}\nTerminal candidate: {terminal.object}\n\nVerified claims:\n{claim_text}\n\nEvidence:\n{evidence}"},
        ]
        self.budget.require(self.max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages), final=True)
        data, generation = self.llm.generate_json(
            messages,
            schema_name="final_answer_v1",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self.budget.record_generation(generation)
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        supported = bool(data.get("supported", False))
        reasons = [str(value) for value in data.get("reasons", [])] if isinstance(data.get("reasons"), list) else []
        answer = str(data.get("answer", "")).strip()
        terminal_slot = next((slot for slot in plan.slots if slot.slot_id == terminal.target_slot), None)
        answer_type = terminal_slot.answer_type if terminal_slot else terminal.answer_type
        answer = normalize_typed_answer(answer, answer_type, question)
        if not supported or not answer or confidence < self.minimum:
            if not supported:
                reasons.append("final_not_supported")
            if confidence < self.minimum:
                reasons.append("final_confidence_below_threshold")
            return None, confidence, list(dict.fromkeys(reasons))
        return answer, min(confidence, terminal.calibrated_confidence), list(dict.fromkeys(reasons))
