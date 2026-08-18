from __future__ import annotations

import re
from typing import Dict, List

from dataset_adapters.base import QAExample
from retriever import DenseTextRetriever, HybridRetriever, SparseTextRetriever
from .common import build_prompt, build_records_from_example, cleanup_prediction, generate_baseline_text


def _parse_ircot_step(text: str) -> Dict[str, str]:
    cleaned = text.strip()
    # soft parser: support lines like
    # Reasoning: ...
    # Query: ...
    # Final Answer: ...
    out: Dict[str, str] = {}
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"Reasoning\s*:\s*(.+)$", line, flags=re.I)
        if m:
            out["reasoning"] = m.group(1).strip()
            continue
        m = re.match(r"Query\s*:\s*(.+)$", line, flags=re.I)
        if m:
            out["next_query"] = m.group(1).strip()
            continue
        m = re.match(r"Final Answer\s*:\s*(.+)$", line, flags=re.I)
        if m:
            out["final_answer"] = m.group(1).strip()
            continue
    if not out and cleaned:
        out["reasoning"] = cleaned.splitlines()[0].strip()
    return out


def _build_step_prompt(question: str, contexts: List[dict], cot_steps: List[str]) -> str:
    format_hint = (
        "Return plain text with up to three lines:\n"
        "Reasoning: <one concise reasoning step>\n"
        "Query: <next retrieval query if needed>\n"
        "Final Answer: <answer if you can already answer>\n"
        "If you cannot finish yet, omit Final Answer. If no new query is needed, omit Query."
    )
    return build_prompt(question=question, contexts=contexts, cot_steps=cot_steps, require_short_answer=False) + "\n\n" + format_hint


def run_ircot(
    example: QAExample,
    llm,
    retriever_type: str = "dense",
    top_k: int = 5,
    max_steps: int = 4,
    encoder_path: str = "",
    index_path: str = "",
    final_max_new_tokens: int = 1200,
    step_max_new_tokens: int = 512,
) -> Dict:
    records = build_records_from_example(example)
    retriever_type = (retriever_type or "dense").lower()
    if retriever_type == "sparse":
        retriever = SparseTextRetriever(records)
    elif retriever_type == "hybrid":
        retriever = HybridRetriever(SparseTextRetriever(records), DenseTextRetriever(records=records, encoder_path=encoder_path, index_path="" if records else index_path))
    else:
        retriever = DenseTextRetriever(records=records, encoder_path=encoder_path, index_path="" if records else index_path)

    cot_steps: List[str] = []
    retrieved_contexts: List[dict] = []
    query = example.question

    for _ in range(max_steps):
        hits = retriever.search(query, top_k=top_k, source=retriever_type)
        contexts = [{"title": h.metadata.get("title", h.item_id), "text": h.text, "score": h.score, "metadata": h.metadata} for h in hits]
        retrieved_contexts.extend(contexts)
        prompt = _build_step_prompt(example.question, contexts, cot_steps)
        generation = generate_baseline_text(
            llm,
            prompt,
            max_new_tokens=step_max_new_tokens,
            temperature=0.1,
            do_sample=False,
            question=example.question,
            contexts=contexts,
            cot_steps=cot_steps,
            enable_tdca_fallback=False,
        )
        parsed = _parse_ircot_step(generation["text"])

        if parsed.get("reasoning"):
            cot_steps.append(parsed["reasoning"])
        if parsed.get("final_answer"):
            return {
                "prediction": cleanup_prediction(parsed["final_answer"]),
                "retrieved": retrieved_contexts,
                "reasoning_steps": cot_steps,
                "stop_reason": "final_answer",
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
        query = parsed.get("next_query") or parsed.get("reasoning") or example.question

    final_prompt = build_prompt(question=example.question, contexts=retrieved_contexts[-top_k:], cot_steps=cot_steps)
    generation = generate_baseline_text(
        llm,
        final_prompt,
        max_new_tokens=final_max_new_tokens,
        temperature=0.0,
        do_sample=False,
        question=example.question,
        contexts=retrieved_contexts[-top_k:],
        cot_steps=cot_steps,
        enable_tdca_fallback=False,
    )
    return {
        "prediction": generation["prediction"],
        "retrieved": retrieved_contexts,
        "reasoning_steps": cot_steps,
        "stop_reason": "max_steps",
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
