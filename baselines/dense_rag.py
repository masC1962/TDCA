from __future__ import annotations

from typing import Dict

from dataset_adapters.base import QAExample
from retriever import DenseTextRetriever
from .common import build_prompt, build_records_from_example, generate_baseline_text


def run_dense_rag(example: QAExample, llm, top_k: int = 5, encoder_path: str = "", index_path: str = "", max_new_tokens: int = 1200) -> Dict:
    records = build_records_from_example(example)
    retriever = DenseTextRetriever(records=records, encoder_path=encoder_path, index_path="" if records else index_path)
    hits = retriever.search(example.question, top_k=top_k, source="dense")
    contexts = [{"title": h.metadata.get("title", h.item_id), "text": h.text, "score": h.score, "metadata": h.metadata} for h in hits]
    prompt = build_prompt(question=example.question, contexts=contexts)
    generation = generate_baseline_text(
        llm,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        question=example.question,
        contexts=contexts,
        cot_steps=[],
        enable_tdca_fallback=False,
    )
    return {
        "prediction": generation["prediction"],
        "retrieved": contexts,
        "reasoning_steps": [],
        "stop_reason": "single_step_rag",
        "raw_generation": generation["raw_generation"],
        "generation_empty": generation["generation_empty"],
        "llm_finish_reason": generation["llm_finish_reason"],
        "llm_usage": generation["llm_usage"],
        "llm_error": generation["llm_error"],
        "llm_debug": generation["llm_debug"],
        "llm_debug_summary": generation["llm_debug_summary"],
        "prediction_source": generation["prediction_source"],
        "raw_generation_primary": generation["raw_generation_primary"],
        "fallback_raw_generation": generation["fallback_raw_generation"],
        "fallback_generation_empty": generation["fallback_generation_empty"],
        "fallback_llm_finish_reason": generation["fallback_llm_finish_reason"],
        "fallback_llm_usage": generation["fallback_llm_usage"],
        "fallback_llm_error": generation["fallback_llm_error"],
    }
