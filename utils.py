from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str | os.PathLike[str], rows: Iterable[Dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with open(path_obj, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | os.PathLike[str], row: Dict[str, Any]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with open(path_obj, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | os.PathLike[str], data: Dict[str, Any]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    if np.allclose(arr.max(), arr.min()):
        return [1.0 for _ in scores] if arr.max() > 0 else [0.0 for _ in scores]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr.tolist()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def extract_json_block(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = fenced if fenced else []
    brace_match = re.search(r"\{.*\}", text, flags=re.S)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def normalize_text(text: str) -> str:
    tokens = simple_tokenize(text)
    return " ".join(tokens)


def lexical_jaccard(a: str, b: str) -> float:
    ta = set(simple_tokenize(a))
    tb = set(simple_tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```(?:text)?\s*<think>.*?</think>\s*```", "", text, flags=re.S | re.I)
    return text.strip()


def canonicalize_state_text(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"\s+", " ", text.strip())
    out = re.sub(r"^Who is the answer to:\s*", "", out, flags=re.I)
    out = re.sub(r"^What is the answer to:\s*", "", out, flags=re.I)
    out = re.sub(r"^Where is the answer to:\s*", "", out, flags=re.I)
    out = re.sub(r"^Which evidence title .*?:\s*", "", out, flags=re.I)
    out = re.sub(r"^Verify the strongest evidence sentence before answering:\s*", "", out, flags=re.I)
    out = re.sub(r"^Verify the strongest evidence chain for:\s*", "", out, flags=re.I)
    out = re.sub(r"^What is the birth city of (.+)$", r"Where was \1 born", out, flags=re.I)
    out = re.sub(r"^Where was the director of (.+) born$", r"What is the birth city of the director of \1", out, flags=re.I)
    out = re.sub(r"^What is the birthplace of (.+)$", r"Where was \1 born", out, flags=re.I)
    out = re.sub(r"^What is the director of (.+)$", r"Who is the director of \1", out, flags=re.I)
    out = re.sub(r"^What is the author of (.+)$", r"Who is the author of \1", out, flags=re.I)
    out = re.sub(r"^Does the evidence support the main relation in:\s*", "Does the evidence support that ", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip()
    if out and not out.endswith("?") and not out.lower().startswith("conclusion:"):
        out += "?"
    return out


META_STATE_PATTERNS = [
    r"which evidence title",
    r"strongest evidence",
    r"missing intermediate entity",
    r"resolve the missing entity",
    r"consider \[",
    r"state to expand",
    r"under limited compute",
]


def is_meta_state_text(text: str) -> bool:
    lower = (text or "").lower()
    return any(re.search(pattern, lower) for pattern in META_STATE_PATTERNS)


ENTITY_STOPWORDS = {
    "what", "which", "who", "where", "when", "is", "the", "a", "an", "of", "to", "for", "in", "on", "movie", "film", "city",
    "birth", "born", "director", "author", "answer", "relation", "verify", "chain", "evidence", "support", "does"
}


def extract_capitalized_phrases(text: str) -> List[str]:
    if not text:
        return []
    phrases = re.findall(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b", text)
    out: List[str] = []
    for phrase in phrases:
        tokens = [t for t in phrase.split() if t.lower() not in ENTITY_STOPWORDS]
        if tokens:
            out.append(" ".join(tokens))
    return list(dict.fromkeys(out))


def relation_keywords(text: str) -> List[str]:
    lower = text.lower()
    keys: List[str] = []
    mapping = {
        "director": ["director", "directed"],
        "author": ["author", "wrote", "written"],
        "birth": ["birth", "born", "birthplace"],
        "spouse": ["spouse", "wife", "husband", "married"],
        "mother": ["mother"],
        "father": ["father"],
        "comparison": ["compare", "same", "different"],
    }
    for variants in mapping.values():
        if any(v in lower for v in variants):
            keys.extend(variants)
    return list(dict.fromkeys(keys))


def relation_signature(text: str) -> str:
    lower = canonicalize_state_text(text).lower()
    if "director" in lower:
        return "director"
    if "born" in lower or "birth city" in lower or "birthplace" in lower:
        return "birth"
    if "author" in lower or "wrote" in lower or "written" in lower:
        return "author"
    if "spouse" in lower or "wife" in lower or "husband" in lower or "married" in lower:
        return "spouse"
    if "mother" in lower:
        return "mother"
    if "father" in lower:
        return "father"
    if "same" in lower or "different" in lower or "compare" in lower:
        return "comparison"
    if lower.startswith("does the evidence support"):
        return "verification"
    return "generic"


def extract_final_answer_text(text: str) -> str:
    text = strip_think_blocks(text or "")
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        line = re.sub(r"^[-*]\s*", "", line)
        m = re.match(r"(?:\*\*\s*)?Final Answer:(?:\*\*\s*)?\s*(.+)$", line, flags=re.I)
        if m:
            answer = m.group(1).strip().rstrip(". ")
            return "" if "<short answer>" in answer.lower() else answer
        m = re.match(r"Answer:\s*(.+)$", line, flags=re.I)
        if m:
            answer = m.group(1).strip().rstrip(". ")
            return "" if "<short answer>" in answer.lower() else answer
    compact = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"(?:final\s+answer|answer)\s*:?", compact, flags=re.I):
        return ""
    if "<short answer>" in compact.lower():
        return ""
    m = re.search(r"The answer is\s+(.+?)(?:[.!?]|$)", compact, flags=re.I)
    if m:
        return m.group(1).strip().rstrip(". ")
    pieces = re.split(r"(?<=[.!?])\s+", compact)
    pieces = [p.strip() for p in pieces if p.strip()]
    if pieces:
        last = pieces[-1]
        if len(last) < 3 and len(pieces) >= 2:
            last = pieces[-2]
        return last.rstrip(". ")
    return compact.rstrip(". ")


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def best_match_by_jaccard(query: str, candidates: Sequence[str]) -> Tuple[str, float]:
    best_text = ""
    best_score = 0.0
    for candidate in candidates:
        score = lexical_jaccard(query, candidate)
        if score > best_score:
            best_text = candidate
            best_score = score
    return best_text, best_score
