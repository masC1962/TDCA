from __future__ import annotations

from typing import Dict

from dataset_adapters.base import QAExample
from .common import build_prompt, generate_baseline_text


def run_closed_book(example: QAExample, llm, max_new_tokens: int = 1200) -> Dict:
    prompt = build_prompt(question=example.question, contexts=None)
    generation = generate_baseline_text(
        llm,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        question=example.question,
        contexts=[],
        cot_steps=[],
        enable_tdca_fallback=False,
    )
    return {
        "prediction": generation["prediction"],
        "retrieved": [],
        "reasoning_steps": [],
        "stop_reason": "closed_book",
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
