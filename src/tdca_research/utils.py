from __future__ import annotations

import hashlib
import json
import os
import re
import string
import tempfile
from pathlib import Path
from typing import Any, Iterable


def normalize_text(text: str) -> str:
    # Matches the bundled official MuSiQue/SQuAD answer scorer: lowercase,
    # remove ASCII punctuation/articles, then normalize whitespace.
    text = (text or "").lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    # Same-directory replace is atomic on supported local filesystems and keeps
    # checkpoints from becoming truncated JSON if a process dies during a write.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_error(exc: BaseException) -> str:
    text = str(exc)
    text = re.sub(r"(?i)(api[_ -]?key|authorization|bearer)\s*[:=]?\s*\S+", r"\1=<redacted>", text)
    return text[:1000]


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative English prompt estimate used only for pre-call budget guards.

    Actual usage from the provider remains authoritative and is checked after calls.
    """
    characters = sum(len(message.get("content", "")) for message in messages)
    return max(1, (characters + 2) // 3 + 8 * len(messages))


def bounded_context(blocks: list[str], max_characters: int) -> str:
    selected: list[str] = []
    used = 0
    for block in blocks:
        remaining = max_characters - used
        if remaining <= 0:
            break
        value = block if len(block) <= remaining else block[:remaining]
        selected.append(value)
        used += len(value)
    return "\n\n".join(selected)


def query_compacted_passages(
    hits: list[Any], query: str, max_characters: int, sentences_per_passage: int = 3,
) -> str:
    """Select query-relevant verbatim sentences while preserving passage provenance.

    This is generic lexical compaction, not content filtering. It never rewrites a
    sentence and falls back to the leading sentences when lexical overlap is absent.
    """
    query_terms = set(tokenize(query))
    blocks = []
    for hit in hits:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", hit.passage.text)
            if sentence.strip()
        ]
        ranked = sorted(
            enumerate(sentences),
            key=lambda pair: (-len(query_terms & set(tokenize(pair[1]))), pair[0]),
        )
        selected_indices = sorted(index for index, _ in ranked[:sentences_per_passage])
        selected = [sentences[index] for index in selected_indices]
        blocks.append(f"[{hit.passage.passage_id}] {hit.passage.title}\n" + " ".join(selected))
    return bounded_context(blocks, max_characters)


def evidence_context(hits: list[Any], query: str, max_characters: int, compaction: str = "none") -> str:
    if compaction == "query_sentence":
        return query_compacted_passages(hits, query, max_characters)
    if compaction != "none":
        raise ValueError(f"unknown evidence compaction mode: {compaction}")
    return bounded_context(
        [f"[{hit.passage.passage_id}] {hit.passage.title}\n{hit.passage.text}" for hit in hits],
        max_characters,
    )
