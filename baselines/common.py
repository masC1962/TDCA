from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from dataset_adapters.base import QAExample
from llm_evaluator import LLMGeneration
from retriever import TextRecord
from utils import extract_final_answer_text


def build_prompt(question: str, contexts: List[dict] | None = None, cot_steps: List[str] | None = None, require_short_answer: bool = True) -> str:
    parts = []
    if contexts:
        parts.append("Context:")
        for i, ctx in enumerate(contexts, start=1):
            title = str(ctx.get("title") or ctx.get("metadata", {}).get("title") or f"Doc {i}")
            text = str(ctx.get("text") or "")
            parts.append(f"[{i}] {title}: {text}")
    if cot_steps:
        parts.append("Reasoning so far:")
        for i, step in enumerate(cot_steps, start=1):
            parts.append(f"Step {i}: {step}")
    parts.append(f"Question: {question}")
    if require_short_answer:
        parts.append("Return ONLY one line in exactly this format:")
        parts.append("Final Answer: <short answer>")
    return "\n".join(parts)


def build_records_from_example(example: QAExample) -> List[TextRecord]:
    return [
        TextRecord(item_id=doc.doc_id, text=doc.text, metadata={"title": doc.title, **doc.metadata})
        for doc in example.docs
    ]


def contexts_to_titles(contexts: Iterable[dict]) -> List[str]:
    titles = []
    for ctx in contexts:
        title = str(ctx.get("title") or ctx.get("metadata", {}).get("title") or "").strip()
        if title:
            titles.append(title)
    return titles


def save_predictions(output_dir: str | Path, rows: List[Dict]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.csv"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    import csv
    preferred = [
        "index",
        "sample_id",
        "question",
        "gold",
        "pred",
        "exact_match",
        "soft_em",
        "answer_f1",
        "rouge1_f",
        "rouge2_f",
        "rougeL_f",
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "meteor",
        "title_hit",
        "retrieved_titles",
        "gold_titles",
        "num_retrieved",
        "num_reasoning_steps",
        "stop_reason",
        "generation_empty",
        "llm_finish_reason",
        "llm_error",
        "prediction_source",
        "llm_debug_summary",
        "run_dir",
    ]
    fieldnames = [key for key in preferred if any(key in row for row in rows)]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(_csv_ready_rows(rows))


def cleanup_prediction(text: str) -> str:
    cleaned = extract_final_answer_text(text or "")
    return cleaned or (text or "").strip()


def generation_to_debug_dict(generation: LLMGeneration) -> Dict:
    debug = generation.debug or {}
    return {
        "raw_generation": generation.raw_text,
        "generation_empty": generation.generation_empty,
        "llm_finish_reason": generation.finish_reason,
        "llm_usage": generation.usage,
        "llm_error": generation.error,
        "llm_debug": debug,
        "llm_debug_summary": _debug_summary(debug),
    }


def _debug_summary(debug: Dict) -> str:
    if not debug:
        return ""
    parts = []
    api_path = debug.get("api_path")
    if api_path:
        parts.append(f"api={api_path}")
    if debug.get("chat_empty_recovery_attempted"):
        labels = []
        for attempt in debug.get("chat_empty_recovery_attempts") or []:
            label = attempt.get("label", "")
            empty = attempt.get("generation_empty")
            if label:
                labels.append(f"{label}:empty={empty}")
        if labels:
            parts.append("recovery=" + "|".join(labels))
    if debug.get("responses_http_error"):
        parts.append(f"responses_http_error={debug.get('responses_http_error')}")
    return "; ".join(parts)


def generate_baseline_text(
    llm,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    do_sample: bool = False,
    *,
    question: str = "",
    contexts: List[dict] | None = None,
    cot_steps: List[str] | None = None,
    enable_tdca_fallback: bool = False,
) -> Dict:
    generation = llm.generate_with_info(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    prediction = cleanup_prediction(generation.text)
    return {
        "text": generation.text,
        "prediction": prediction,
        "prediction_source": "primary" if prediction else "empty",
        **generation_to_debug_dict(generation),
        "raw_generation_primary": generation.raw_text,
        "fallback_raw_generation": "",
        "fallback_generation_empty": False,
        "fallback_llm_finish_reason": "",
        "fallback_llm_usage": {},
        "fallback_llm_error": None,
    }


def _csv_ready_rows(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for row in rows:
        normalized = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value, ensure_ascii=False)
            else:
                normalized[key] = value
        out.append(normalized)
    return out
