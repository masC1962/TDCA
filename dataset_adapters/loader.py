from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .base import DocRecord, QAExample


def _read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
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


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(_flatten_text(v) for v in value if _flatten_text(v))
    if isinstance(value, dict):
        for key in ["text", "sentences", "paragraph", "paragraph_text"]:
            if key in value:
                return _flatten_text(value[key])
    return ""


def _extract_docs(item: Dict[str, Any]) -> List[DocRecord]:
    docs: List[DocRecord] = []
    context = item.get("context")
    if isinstance(context, list):
        for idx, entry in enumerate(context, start=1):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                title = str(entry[0] or f"doc_{idx}")
                text = _flatten_text(entry[1])
            elif isinstance(entry, dict):
                title = str(entry.get("title") or entry.get("name") or f"doc_{idx}")
                text = _flatten_text(entry)
            else:
                title = f"doc_{idx}"
                text = _flatten_text(entry)
            if text:
                docs.append(DocRecord(doc_id=f"doc_{idx}", title=title, text=text, metadata={"title": title}))
        if docs:
            return docs

    for field in ["paragraphs", "documents", "docs", "evidence"]:
        value = item.get(field)
        if not isinstance(value, list):
            continue
        for idx, doc in enumerate(value, start=1):
            if isinstance(doc, dict):
                title = str(doc.get("title") or doc.get("name") or f"doc_{idx}")
                text = _flatten_text(doc)
            else:
                title = f"doc_{idx}"
                text = _flatten_text(doc)
            if text:
                docs.append(DocRecord(doc_id=f"doc_{idx}", title=title, text=text, metadata={"title": title}))
        if docs:
            return docs
    return docs


def _extract_gold_answers(item: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ["answer", "gold", "target"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, list):
            values.extend(str(v).strip() for v in value if str(v).strip())
    for key in ["answer_aliases", "golden_answers"]:
        value = item.get(key)
        if isinstance(value, list):
            values.extend(str(v).strip() for v in value if str(v).strip())
    dedup = []
    seen = set()
    for v in values:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


def _extract_gold_titles(item: Dict[str, Any]) -> List[str]:
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


def load_examples(path: str | Path, dataset_name: str = "generic", limit: int = -1) -> List[QAExample]:
    rows = _read_json_or_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    out: List[QAExample] = []
    for idx, item in enumerate(rows):
        qid = str(item.get("id") or item.get("_id") or idx)
        question = str(item.get("question") or item.get("query") or item.get("input") or "").strip()
        docs = _extract_docs(item)
        gold_answers = _extract_gold_answers(item)
        metadata = {
            "dataset_name": dataset_name,
            "question_type": item.get("type") or item.get("question_type") or "",
            "gold_titles": _extract_gold_titles(item),
            "raw": item,
        }
        out.append(QAExample(qid=qid, question=question, gold_answers=gold_answers, docs=docs, metadata=metadata))
    return out
