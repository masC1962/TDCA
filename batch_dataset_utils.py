from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dataset_adapters.base import DocRecord, QAExample


def load_dataset_items(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        raise ValueError(f"Unsupported JSON dataset shape for {path}")

    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for row in load_dataset_items(path):
        yield row


def get_question(item: Dict[str, Any]) -> str:
    for key in ["question", "query", "input"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Cannot find question field in item keys={list(item.keys())}")


def get_gold(item: Dict[str, Any]) -> str:
    for key in ["answer", "gold", "target"]:
        value = item.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            return str(value[0])
    aliases = item.get("answer_aliases") or item.get("golden_answers") or []
    if isinstance(aliases, list) and aliases:
        return str(aliases[0])
    return ""


def _flatten_sentences(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            text = _flatten_sentences(item)
            if text:
                parts.append(text)
        return " ".join(parts)
    return ""


def extract_evidence_rows(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    context = item.get("context")
    if isinstance(context, list):
        for idx, entry in enumerate(context, start=1):
            title = ""
            sentences: Any = ""
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                title = str(entry[0] or f"doc_{idx}")
                sentences = entry[1]
            elif isinstance(entry, dict):
                title = str(entry.get("title") or entry.get("name") or f"doc_{idx}")
                sentences = entry.get("sentences") or entry.get("text") or entry.get("paragraph") or entry.get("paragraph_text") or ""
            else:
                title = f"doc_{idx}"
                sentences = entry
            text = _flatten_sentences(sentences)
            if text:
                rows.append({"id": f"doc_{idx}", "title": title, "text": text})
    if rows:
        return rows

    for field in ["paragraphs", "documents", "docs", "evidence"]:
        docs = item.get(field)
        if not isinstance(docs, list):
            continue
        for idx, doc in enumerate(docs, start=1):
            if isinstance(doc, dict):
                title = str(doc.get("title") or doc.get("name") or f"doc_{idx}")
                text = _flatten_sentences(doc.get("text") or doc.get("sentences") or doc.get("paragraph") or doc.get("paragraph_text") or "")
            else:
                title = f"doc_{idx}"
                text = _flatten_sentences(doc)
            if text:
                rows.append({"id": f"doc_{idx}", "title": title, "text": text})
        if rows:
            return rows
    return rows


def extract_hotpot_evidence_rows(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return extract_evidence_rows(item)


def extract_gold_titles(item: Dict[str, Any]) -> List[str]:
    titles: List[str] = []
    sf = item.get("supporting_facts")
    if isinstance(sf, list):
        for entry in sf:
            if isinstance(entry, (list, tuple)) and entry:
                titles.append(str(entry[0]))
            elif isinstance(entry, dict) and entry.get("title"):
                titles.append(str(entry.get("title")))
    paragraphs = item.get("paragraphs")
    if isinstance(paragraphs, list):
        for doc in paragraphs:
            if isinstance(doc, dict) and doc.get("is_supporting") and doc.get("title"):
                titles.append(str(doc.get("title")))
        qdecomp = item.get("question_decomposition")
        if isinstance(qdecomp, list):
            for step in qdecomp:
                if not isinstance(step, dict):
                    continue
                support_idx = step.get("paragraph_support_idx")
                if isinstance(support_idx, int) and 0 <= support_idx < len(paragraphs):
                    para = paragraphs[support_idx]
                    if isinstance(para, dict) and para.get("title"):
                        titles.append(str(para.get("title")))
    return list(dict.fromkeys(titles))


def get_gold_answers(item: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    primary = get_gold(item)
    if primary:
        values.append(primary)
    for key in ["answer_aliases", "golden_answers"]:
        aliases = item.get(key)
        if isinstance(aliases, list):
            values.extend(str(v).strip() for v in aliases if str(v).strip())
    dedup: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            dedup.append(value)
            seen.add(value)
    return dedup


def build_qa_example(item: Dict[str, Any], dataset_name: str, fallback_id: str) -> QAExample:
    rows = extract_evidence_rows(item)
    docs = [
        DocRecord(
            doc_id=str(row.get("id") or f"doc_{idx}"),
            title=str(row.get("title") or f"doc_{idx}"),
            text=str(row.get("text") or ""),
            metadata={"title": str(row.get("title") or f"doc_{idx}")},
        )
        for idx, row in enumerate(rows, start=1)
        if str(row.get("text") or "").strip()
    ]
    gold_answers = get_gold_answers(item)
    return QAExample(
        qid=str(item.get("id") or item.get("_id") or fallback_id),
        question=get_question(item),
        gold_answers=gold_answers,
        docs=docs,
        metadata={
            "dataset_name": dataset_name,
            "question_type": item.get("type") or item.get("question_type") or "",
            "gold_titles": extract_gold_titles(item),
            "raw": item,
        },
    )
