from __future__ import annotations

import time

from ..budget import Budget, BudgetExceeded
from ..llm import BaseLLM, InfrastructureError, StructuredOutputError
from ..models import Prediction, QAExample, RunStatus, Usage
from ..retrieval import BaseRetriever
from ..utils import bounded_context, estimate_message_tokens


STEP_SYSTEM = """Interleave one concise reasoning step with retrieval. Return JSON only with:
reasoning, next_query, final_answer. Use an empty final_answer until the supplied evidence supports the answer."""

REPAIR_SYSTEM = """Return valid compact JSON only with exactly these string fields:
reasoning, next_query, final_answer. Keep reasoning under 25 words and next_query under 15 words.
Use an empty final_answer unless the evidence already supports a short answer."""


def _generate_step_with_one_repair(
    llm: BaseLLM,
    budget: Budget,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
):
    """Retry one malformed/truncated JSON response inside the declared budget."""
    requested = min(400, max_tokens)
    budget.require(requested, estimated_prompt_tokens=estimate_message_tokens(messages))
    try:
        data, generation = llm.generate_json(messages, "ircot_step_v1", requested, temperature)
    except StructuredOutputError as exc:
        budget.record_generation(exc.generation)
        repair_messages = [
            {"role": "system", "content": REPAIR_SYSTEM},
            messages[-1],
        ]
        repair_tokens = min(220, max_tokens)
        budget.require(repair_tokens, estimated_prompt_tokens=estimate_message_tokens(repair_messages))
        try:
            data, generation = llm.generate_json(repair_messages, "ircot_step_repair_v1", repair_tokens, temperature)
        except StructuredOutputError as retry_exc:
            budget.record_generation(retry_exc.generation)
            raise InfrastructureError(f"structured output repair failed: {retry_exc}") from retry_exc
    budget.record_generation(generation)
    return data


def run_ircot(
    example: QAExample,
    llm: BaseLLM,
    retriever: BaseRetriever,
    top_k: int,
    max_steps: int,
    max_calls: int,
    max_tokens: int,
    final_reserve: int,
    temperature: float = 0.0,
    evidence_chars: int = 6000,
) -> Prediction:
    started = time.perf_counter()
    usage = Usage()
    budget = Budget(max_calls, max_tokens, final_reserve, usage)
    query = example.question
    reasoning: list[str] = []
    hits_by_id = {}
    try:
        for _ in range(max_steps):
            budget.record_retrieval()
            hits = retriever.search(query, top_k)
            for hit in hits:
                hits_by_id[hit.passage.passage_id] = hit
            context = bounded_context([f"[{hit.passage.passage_id}] {hit.passage.title}\n{hit.passage.text}" for hit in hits], evidence_chars)
            messages = [
                {"role": "system", "content": STEP_SYSTEM},
                {"role": "user", "content": f"Question: {example.question}\nPrior reasoning: {reasoning}\nEvidence:\n{context}"},
            ]
            data = _generate_step_with_one_repair(llm, budget, messages, max_tokens, temperature)
            step = str(data.get("reasoning", "")).strip()
            if step:
                reasoning.append(step)
            answer = str(data.get("final_answer", "")).strip()
            if answer:
                usage.wall_seconds = time.perf_counter() - started
                return Prediction(example.qid, example.question, RunStatus.ANSWER, answer, 0.5, "ircot_final_answer", retrieved=list(hits_by_id.values()), usage=usage)
            query = str(data.get("next_query", "")).strip() or step or example.question
        context = bounded_context([f"[{hit.passage.passage_id}] {hit.passage.title}\n{hit.passage.text}" for hit in hits_by_id.values()], evidence_chars)
        messages = [
            {"role": "system", "content": "Return one short answer supported by evidence, or ABSTAIN."},
            {"role": "user", "content": f"Question: {example.question}\nReasoning: {reasoning}\nEvidence:\n{context}"},
        ]
        budget.require(min(300, max_tokens), estimated_prompt_tokens=estimate_message_tokens(messages), final=True)
        generation = llm.generate_text(messages, min(300, max_tokens), temperature)
        budget.record_generation(generation)
        answer = generation.text.strip()
        if not answer or answer.upper() == "ABSTAIN":
            usage.wall_seconds = time.perf_counter() - started
            return Prediction(example.qid, example.question, RunStatus.ABSTAIN, None, 0.0, "ircot_abstain", retrieved=list(hits_by_id.values()), usage=usage)
        usage.wall_seconds = time.perf_counter() - started
        return Prediction(example.qid, example.question, RunStatus.ANSWER, answer, 0.5, "ircot_final_synthesis", retrieved=list(hits_by_id.values()), usage=usage)
    except InfrastructureError as exc:
        budget.record_infrastructure_failure(exc)
        usage.wall_seconds = time.perf_counter() - started
        return Prediction(example.qid, example.question, RunStatus.INFRASTRUCTURE_FAILURE, None, 0.0, "infrastructure_failure", retrieved=list(hits_by_id.values()), usage=usage, error=str(exc))
    except BudgetExceeded:
        usage.wall_seconds = time.perf_counter() - started
        return Prediction(example.qid, example.question, RunStatus.ABSTAIN, None, 0.0, "budget_exhausted", retrieved=list(hits_by_id.values()), usage=usage)
