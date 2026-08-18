from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..models import Passage, QAExample


def _text(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(filter(None, (_text(item) for item in value)))
    return ""


def _answers(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("answer", "gold", "target"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
    for key in ("answer_aliases", "golden_answers"):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
    return list(dict.fromkeys(values))


def _top_level_passages(row: dict[str, Any]) -> list[tuple[str, str, bool, dict[str, Any]]]:
    out: list[tuple[str, str, bool, dict[str, Any]]] = []
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    context = row.get("context") or metadata.get("context")
    if isinstance(context, dict):
        # 2WikiMultihopQA uses parallel title/content arrays.
        titles = context.get("title", [])
        contents = context.get("content", context.get("sentences", []))
        if isinstance(titles, list) and isinstance(contents, list):
            for index, (title, body) in enumerate(zip(titles, contents)):
                out.append((str(title or f"doc_{index}"), _text(body), False, {"idx": index}))
            if out:
                return out
    if isinstance(context, list):
        for index, item in enumerate(context):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((str(item[0]), _text(item[1]), False, {}))
            elif isinstance(item, dict):
                title = str(item.get("title") or item.get("name") or f"doc_{index}")
                body = _text(item.get("sentences") or item.get("text") or item.get("paragraph") or item.get("paragraph_text"))
                out.append((title, body, bool(item.get("is_supporting", False)), dict(item)))
        if out:
            return out
    for key in ("paragraphs", "documents", "docs", "evidence"):
        values = row.get(key)
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("name") or f"doc_{index}")
                body = _text(item.get("text") or item.get("sentences") or item.get("paragraph") or item.get("paragraph_text"))
                out.append((title, body, bool(item.get("is_supporting", False)), dict(item)))
            else:
                out.append((f"doc_{index}", _text(item), False, {}))
        if out:
            return out
    return out


def _nested_support_passages(row: dict[str, Any]) -> list[tuple[str, str, bool, dict[str, Any]]]:
    """Read compact subsets whose evidence lives under decomposition metadata."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    decomposition = metadata.get("question_decomposition") or row.get("question_decomposition") or []
    out: list[tuple[str, str, bool, dict[str, Any]]] = []
    if not isinstance(decomposition, list):
        return out
    for step in decomposition:
        if not isinstance(step, dict):
            continue
        support = step.get("support_paragraph")
        if not isinstance(support, dict):
            continue
        title = str(support.get("title") or f"support_{len(out)}")
        body = _text(support.get("paragraph_text") or support.get("text") or support.get("sentences"))
        if body:
            out.append((title, body, True, dict(support)))
    return out


def _gold_titles_and_ids(row: dict[str, Any], passages: list[Passage], raw: list[tuple[str, str, bool, dict[str, Any]]]) -> tuple[list[str], list[str]]:
    # MuSiQue may contain many chunks with the same Wikipedia title. Preserve
    # paragraph-level labels when present instead of expanding a supporting
    # title to every same-title distractor chunk.
    titles = [title for title, _, supporting, _ in raw if supporting]
    explicit_ids = []
    for passage_index, (_, body, supporting, raw_meta) in enumerate(raw):
        if not supporting or not body:
            continue
        source_id = next(
            (raw_meta[key] for key in ("id", "idx", "doc_id") if key in raw_meta and raw_meta[key] is not None),
            None,
        )
        explicit_ids.append(str(source_id if source_id is not None else f"doc_{passage_index}"))
    supporting_facts = row.get("supporting_facts")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if supporting_facts is None:
        supporting_facts = metadata.get("supporting_facts")
    if isinstance(supporting_facts, list):
        for fact in supporting_facts:
            if isinstance(fact, (list, tuple)) and fact:
                titles.append(str(fact[0]))
            elif isinstance(fact, dict) and fact.get("title"):
                titles.append(str(fact["title"]))
    elif isinstance(supporting_facts, dict):
        raw_titles = supporting_facts.get("title", [])
        if isinstance(raw_titles, list):
            titles.extend(str(title) for title in raw_titles if str(title))
    titles = list(dict.fromkeys(filter(None, titles)))
    if explicit_ids:
        return titles, list(dict.fromkeys(explicit_ids))
    title_set = set(titles)
    ids = [p.passage_id for p in passages if p.title in title_set]
    return titles, ids


def _oracle_decomposition(row: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = row.get("question_decomposition") or metadata.get("question_decomposition") or []
    return [dict(step) for step in value if isinstance(step, dict)] if isinstance(value, list) else []


def parse_example(row: dict[str, Any], dataset: str, index: int = 0) -> QAExample:
    question = next((str(row.get(key)).strip() for key in ("question", "query", "input") if isinstance(row.get(key), str) and str(row.get(key)).strip()), "")
    if not question:
        raise ValueError(f"row {index} has no question")
    raw_passages = _top_level_passages(row)
    if not raw_passages:
        raw_passages = _nested_support_passages(row)
    passages: list[Passage] = []
    for passage_index, (title, body, _, raw_meta) in enumerate(raw_passages):
        if not body:
            continue
        source_id = next(
            (raw_meta[key] for key in ("id", "idx", "doc_id") if key in raw_meta and raw_meta[key] is not None),
            None,
        )
        passage_id = str(source_id if source_id is not None else f"doc_{passage_index}")
        passages.append(Passage(passage_id=passage_id, title=title, text=body))
    gold_titles, gold_ids = _gold_titles_and_ids(row, passages, raw_passages)
    oracle = _oracle_decomposition(row)
    hop_count = len(oracle) or len(gold_titles) or None
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return QAExample(
        qid=str(row.get("id") or row.get("_id") or index),
        question=question,
        passages=passages,
        answers=_answers(row),
        gold_document_ids=gold_ids,
        gold_titles=gold_titles,
        oracle_decomposition=oracle,
        answerable=bool(row.get("answerable", metadata.get("answerable", True))),
        hop_count=hop_count,
        metadata={
            "dataset": dataset,
            "source_index": row.get("_tdca_source_index", index),
            # Evaluation-only grouping label. QAExample.inference_view deliberately
            # excludes metadata, so normal prompts cannot observe this field.
            "question_type": row.get("type") or metadata.get("type"),
        },
    )


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("data", data.get("examples"))
        if not isinstance(data, list):
            raise ValueError(f"unsupported JSON shape in {path}")
        yield from (row for row in data if isinstance(row, dict))
        return
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def load_examples(path: str | Path, dataset: str) -> list[QAExample]:
    source = Path(path)
    return [parse_example(row, dataset, index) for index, row in enumerate(_rows(source))]
