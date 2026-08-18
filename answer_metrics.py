from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List


def gold_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if str(v).strip()]
            except Exception:
                pass
        return [text] if text else []
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower().strip()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def answer_tokens(text: str) -> List[str]:
    norm = normalize_answer(text)
    if not norm:
        return []
    if " " not in norm and any("\u4e00" <= ch <= "\u9fff" for ch in norm):
        return list(norm)
    return norm.split()


def exact_match(pred: str, golds: Any) -> int:
    pred_norm = normalize_answer(pred)
    return int(any(pred_norm == normalize_answer(g) and normalize_answer(g) for g in gold_list(golds)))


def token_f1(pred: str, golds: Any) -> float:
    pred_tokens = answer_tokens(pred)
    best = 0.0
    for gold in gold_list(golds):
        gold_tokens = answer_tokens(gold)
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if overlap <= 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f(pred: str, golds: Any, n: int) -> float:
    pred_tokens = answer_tokens(pred)
    best = 0.0
    for gold in gold_list(golds):
        gold_tokens = answer_tokens(gold)
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if len(pred_tokens) < n or len(gold_tokens) < n:
            continue
        pred_counts = Counter(_ngrams(pred_tokens, n))
        gold_counts = Counter(_ngrams(gold_tokens, n))
        overlap = sum((pred_counts & gold_counts).values())
        if overlap <= 0:
            continue
        precision = overlap / max(sum(pred_counts.values()), 1)
        recall = overlap / max(sum(gold_counts.values()), 1)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _lcs_len(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for tok_a in a:
        prev = 0
        for j, tok_b in enumerate(b, start=1):
            cur = dp[j]
            if tok_a == tok_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l_f(pred: str, golds: Any) -> float:
    pred_tokens = answer_tokens(pred)
    best = 0.0
    for gold in gold_list(golds):
        gold_tokens = answer_tokens(gold)
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        lcs = _lcs_len(pred_tokens, gold_tokens)
        if lcs <= 0:
            continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def bleu_n(pred: str, golds: Any, n: int) -> float:
    pred_tokens = answer_tokens(pred)
    best = 0.0
    for gold in gold_list(golds):
        gold_tokens = answer_tokens(gold)
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens or len(pred_tokens) < n:
            continue
        pred_counts = Counter(_ngrams(pred_tokens, n))
        gold_counts = Counter(_ngrams(gold_tokens, n))
        clipped = sum((pred_counts & gold_counts).values())
        precision = clipped / max(sum(pred_counts.values()), 1)
        if precision <= 0:
            continue
        brevity = 1.0 if len(pred_tokens) >= len(gold_tokens) else math.exp(1 - len(gold_tokens) / max(len(pred_tokens), 1))
        best = max(best, brevity * precision)
    return best


def meteor_like(pred: str, golds: Any) -> float:
    pred_tokens = answer_tokens(pred)
    best = 0.0
    for gold in gold_list(golds):
        gold_tokens = answer_tokens(gold)
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if overlap <= 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        alpha = 0.9
        best = max(best, (precision * recall) / (alpha * precision + (1 - alpha) * recall))
    return best


def _numeric_equivalent(pred: str, gold: str) -> bool:
    pred_digits = re.sub(r"\D", "", pred or "")
    gold_digits = re.sub(r"\D", "", gold or "")
    return bool(pred_digits and pred_digits == gold_digits)


def soft_exact_match(pred: str, golds: Any, *, f1_threshold: float = 0.80) -> int:
    if exact_match(pred, golds):
        return 1
    pred_norm = normalize_answer(pred)
    if not pred_norm:
        return 0
    for gold in gold_list(golds):
        gold_norm = normalize_answer(gold)
        if not gold_norm:
            continue
        if pred_norm in gold_norm or gold_norm in pred_norm:
            return 1
        if _numeric_equivalent(pred, gold):
            return 1
    return int(token_f1(pred, golds) >= f1_threshold)


def title_hit(retrieved_titles: Any, gold_titles: Any) -> int:
    pred = {normalize_answer(t) for t in gold_list(retrieved_titles)}
    gold = {normalize_answer(t) for t in gold_list(gold_titles)}
    pred.discard("")
    gold.discard("")
    return int(bool(pred & gold))


def compute_answer_metrics(pred: str, golds: Any, row: Dict[str, Any] | None = None) -> Dict[str, Any]:
    row = row or {}
    return {
        "soft_em": soft_exact_match(pred, golds),
        "answer_f1": round(token_f1(pred, golds), 6),
        "rouge1_f": round(rouge_n_f(pred, golds, 1), 6),
        "rouge2_f": round(rouge_n_f(pred, golds, 2), 6),
        "rougeL_f": round(rouge_l_f(pred, golds), 6),
        "bleu1": round(bleu_n(pred, golds, 1), 6),
        "bleu2": round(bleu_n(pred, golds, 2), 6),
        "bleu3": round(bleu_n(pred, golds, 3), 6),
        "bleu4": round(bleu_n(pred, golds, 4), 6),
        "meteor": round(meteor_like(pred, golds), 6),
        "title_hit": title_hit(row.get("retrieved_titles", ""), row.get("gold_titles", "")),
    }


METRIC_KEYS = [
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
]


def aggregate_metric_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    materialized = list(rows)
    aggregate: Dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = []
        for row in materialized:
            try:
                vals.append(float(row.get(key, 0.0) or 0.0))
            except Exception:
                pass
        aggregate[key] = round(sum(vals) / len(vals), 6) if vals else 0.0
    return aggregate
