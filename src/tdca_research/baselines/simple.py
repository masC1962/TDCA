from __future__ import annotations

import time

from ..budget import Budget, BudgetExceeded
from ..llm import BaseLLM, InfrastructureError
from ..models import Prediction, QAExample, RunStatus, Usage
from ..retrieval import BaseRetriever
from ..utils import bounded_context, estimate_message_tokens


def _answer(llm: BaseLLM, budget: Budget, question: str, context: str, max_tokens: int, temperature: float) -> Prediction:
    messages = [
        {"role": "system", "content": "Answer the question with one short answer. Use only supplied evidence when present. If unsupported, reply ABSTAIN."},
        {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
    ]
    budget.require(max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages), final=True)
    generation = llm.generate_text(messages, max_tokens=max_tokens, temperature=temperature)
    budget.record_generation(generation)
    text = generation.text.strip()
    if text.upper() == "ABSTAIN" or not text:
        return Prediction("", question, RunStatus.ABSTAIN, None, 0.0, "baseline_abstain", usage=budget.usage)
    return Prediction("", question, RunStatus.ANSWER, text, 0.5, "baseline_answer", usage=budget.usage)


def run_closed_book(example: QAExample, llm: BaseLLM, max_calls: int, max_tokens: int, temperature: float = 0.0) -> Prediction:
    started = time.perf_counter()
    usage = Usage()
    budget = Budget(max_calls, max_tokens, 0, usage)
    try:
        prediction = _answer(llm, budget, example.question, "", min(512, max_tokens), temperature)
    except InfrastructureError as exc:
        budget.record_infrastructure_failure(exc)
        prediction = Prediction("", example.question, RunStatus.INFRASTRUCTURE_FAILURE, None, 0.0, "infrastructure_failure", usage=usage, error=str(exc))
    except BudgetExceeded:
        prediction = Prediction("", example.question, RunStatus.ABSTAIN, None, 0.0, "budget_exhausted", usage=usage)
    prediction.qid = example.qid
    usage.wall_seconds = time.perf_counter() - started
    return prediction


def run_rag(example: QAExample, llm: BaseLLM, retriever: BaseRetriever, top_k: int, max_calls: int, max_tokens: int, temperature: float = 0.0, evidence_chars: int = 6000) -> Prediction:
    started = time.perf_counter()
    usage = Usage(retrieval_calls=1)
    budget = Budget(max_calls, max_tokens, 0, usage)
    hits = retriever.search(example.question, top_k)
    context = bounded_context([f"[{hit.passage.passage_id}] {hit.passage.title}\n{hit.passage.text}" for hit in hits], evidence_chars)
    try:
        prediction = _answer(llm, budget, example.question, context, min(512, max_tokens), temperature)
    except InfrastructureError as exc:
        budget.record_infrastructure_failure(exc)
        prediction = Prediction("", example.question, RunStatus.INFRASTRUCTURE_FAILURE, None, 0.0, "infrastructure_failure", retrieved=hits, usage=usage, error=str(exc))
    except BudgetExceeded:
        prediction = Prediction("", example.question, RunStatus.ABSTAIN, None, 0.0, "budget_exhausted", retrieved=hits, usage=usage)
    prediction.qid = example.qid
    prediction.retrieved = hits
    usage.wall_seconds = time.perf_counter() - started
    return prediction
