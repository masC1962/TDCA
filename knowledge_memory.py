from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from core_models import RetrievedContext
from retriever import BaseRetriever, SparseTextRetriever, TextRecord, build_retriever
from utils import (
    append_jsonl,
    canonicalize_state_text,
    extract_capitalized_phrases,
    lexical_jaccard,
    load_jsonl,
    normalize_scores,
    normalize_text,
    relation_keywords,
    relation_signature,
    save_jsonl,
    simple_tokenize,
)


class EvidenceStore:
    def __init__(
        self,
        evidence_path: str | Path,
        retriever: Optional[BaseRetriever] = None,
        retriever_type: str = "sparse",
        encoder_path: str = "",
        index_path: str = "",
    ) -> None:
        self.evidence_path = Path(evidence_path)
        self.records = self._load_or_create_default()
        if retriever is not None:
            self.retriever = retriever
        else:
            try:
                self.retriever = build_retriever(
                    retriever_type=retriever_type,
                    records=self.records,
                    encoder_path=encoder_path,
                    index_path=index_path,
                )
                self.retriever_type = (retriever_type or "sparse").strip().lower()
            except Exception as exc:
                print(f"[EvidenceStore] WARNING: retriever_type={retriever_type!r} unavailable ({exc}); falling back to sparse.")
                self.retriever = SparseTextRetriever(self.records)
                self.retriever_type = "sparse"

    def _load_or_create_default(self) -> List[TextRecord]:
        rows = load_jsonl(self.evidence_path)
        if not rows:
            default_rows = [
                {"id": "doc_1", "title": "Inception Overview", "text": "Inception is a 2010 science fiction film directed by Christopher Nolan."},
                {"id": "doc_2", "title": "Christopher Nolan Biography", "text": "Christopher Nolan is a film director born in London."},
                {"id": "doc_3", "title": "Distractor Movie Fact", "text": "Leonardo DiCaprio starred in Inception and was born in Los Angeles."},
                {"id": "doc_4", "title": "Reasoning Hint", "text": "To answer a compositional question, first identify the intermediate entity, then retrieve the asked attribute."},
                {"id": "doc_5", "title": "Distractor Film", "text": "Interstellar is another Christopher Nolan film."},
            ]
            save_jsonl(self.evidence_path, default_rows)
            rows = default_rows

        records: List[TextRecord] = []
        for row in rows:
            item_id = str(row.get("id") or row.get("doc_id") or len(records))
            text = str(row.get("text") or "")
            metadata = {k: v for k, v in row.items() if k not in {"id", "doc_id", "text"}}
            records.append(TextRecord(item_id=item_id, text=text, metadata=metadata))
        return records

    def _rerank(self, query: str, items: List[RetrievedContext], top_k: int) -> List[RetrievedContext]:
        if not items:
            return []
        query = canonicalize_state_text(query)
        q_tokens = set(simple_tokenize(query))
        q_entities = extract_capitalized_phrases(query)
        q_rel = relation_keywords(query)
        scores: List[float] = []
        reranked: List[RetrievedContext] = []

        for item in items:
            text = item.text
            doc_tokens = set(simple_tokenize(text))
            doc_entities = extract_capitalized_phrases(text)
            base = float(item.metadata.get("raw_score", item.score))
            overlap = len(q_tokens & doc_tokens) / max(1, len(q_tokens))
            entity_overlap = 0.0
            if q_entities:
                entity_overlap = max((lexical_jaccard(qe, de) for qe in q_entities for de in doc_entities), default=0.0)
            rel_hit = 0.0
            if q_rel:
                rel_hit = 1.0 if any(k in text.lower() for k in q_rel) else 0.0

            lower_q = query.lower()
            lower_t = text.lower()
            distractor_penalty = 0.0
            if q_entities and doc_entities and entity_overlap < 0.45:
                distractor_penalty += 0.42
            if "born" in lower_q and "born" in lower_t and q_entities and entity_overlap < 0.55:
                distractor_penalty += 0.40
            if "director" in lower_q and not any(x in lower_t for x in ["director", "directed"]):
                distractor_penalty += 0.22
            if "who is the director" in lower_q and "born in" in lower_t and entity_overlap < 0.55:
                distractor_penalty += 0.25
            if "born" in lower_q and any(x in lower_t for x in ["film", "movie"]) and "born in" not in lower_t:
                distractor_penalty += 0.18
            if "director" in lower_q and any(x in lower_t for x in ["another", "also", "other"]) and "directed by" not in lower_t:
                distractor_penalty += 0.20
            if q_rel and rel_hit == 0.0 and entity_overlap >= 0.65:
                distractor_penalty += 0.22

            title = str(item.metadata.get("title", "")).lower()
            reasoning_bonus = 0.015 if ("composition" in lower_t or "reasoning" in title) else 0.0
            fact_bonus = 0.14 if entity_overlap >= 0.8 and rel_hit > 0 else 0.0
            score = 0.34 * base + 0.20 * overlap + 0.28 * entity_overlap + 0.24 * rel_hit + reasoning_bonus + fact_bonus - distractor_penalty
            score = max(0.0, score)
            reranked.append(RetrievedContext(item_id=item.item_id, text=item.text, score=score, source=item.source, metadata=item.metadata))
            scores.append(score)

        if not any(s > 0 for s in scores):
            return []
        order = sorted(range(len(reranked)), key=lambda i: scores[i], reverse=True)[:top_k]
        norm = normalize_scores([scores[i] for i in order])
        out: List[RetrievedContext] = []
        for rank, idx in enumerate(order):
            item = reranked[idx]
            item.score = norm[rank] if rank < len(norm) else item.score
            out.append(item)
        return out

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedContext]:
        candidates = self.retriever.search(query=query, top_k=max(top_k * 3, top_k), source="kg", return_raw=True)
        return self._rerank(query, candidates, top_k=top_k)


class MemoryBank:
    def __init__(self, memory_path: str | Path, retriever: Optional[BaseRetriever] = None) -> None:
        self.memory_path = Path(memory_path)
        self.records = self._load_or_create_default()
        self.retriever = retriever or SparseTextRetriever(self.records)

    def _load_or_create_default(self) -> List[TextRecord]:
        rows = load_jsonl(self.memory_path)
        if not rows:
            default_rows = [
                {"id": "mem_1", "text": "Template: for compositional multi-hop questions, first solve the intermediate entity and then ask for the final attribute.", "score": 0.96, "tag": "compositional_template", "memory_kind": "template"},
                {"id": "mem_2", "text": "Template: for comparison questions, split the problem into one sub-question per entity before comparing them.", "score": 0.90, "tag": "comparison_template", "memory_kind": "template"},
                {"id": "mem_3", "text": "Template: when evidence already grounds the answer, switch from branching to short verification and synthesis.", "score": 0.88, "tag": "verification_then_stop", "memory_kind": "template"},
            ]
            save_jsonl(self.memory_path, default_rows)
            rows = default_rows

        records: List[TextRecord] = []
        for row in rows:
            item_id = str(row.get("id") or len(records))
            text = str(row.get("text") or "")
            metadata = {k: v for k, v in row.items() if k not in {"id", "text"}}
            records.append(TextRecord(item_id=item_id, text=text, metadata=metadata))
        return records

    def retrieve(self, query: str, top_k: int = 2) -> List[RetrievedContext]:
        results = self.retriever.search(query=query, top_k=top_k, source="memory")
        for item in results:
            item.score *= float(item.metadata.get("score", 1.0))
        return results

    def add_memory(self, text: str, score: float, metadata: Optional[Dict[str, Any]] = None) -> str:
        metadata = metadata or {}
        norm_new = normalize_text(text)
        target_norm = normalize_text(str(metadata.get("target_question_norm") or metadata.get("target_question") or ""))
        answer_norm = normalize_text(str(metadata.get("answer_text") or ""))
        sig = relation_signature(str(metadata.get("target_question") or text))
        for record in self.records:
            rec_meta = record.metadata or {}
            rec_target = normalize_text(str(rec_meta.get("target_question_norm") or rec_meta.get("target_question") or ""))
            rec_answer = normalize_text(str(rec_meta.get("answer_text") or ""))
            rec_sig = relation_signature(str(rec_meta.get("target_question") or record.text))
            same_target_answer = target_norm and answer_norm and rec_target == target_norm and rec_answer == answer_norm and rec_sig == sig
            same_surface = normalize_text(record.text) == norm_new
            if same_target_answer or same_surface:
                rec_meta["score"] = max(float(rec_meta.get("score", 0.0)), float(score))
                record.metadata = rec_meta
                return record.item_id
        row = {"id": f"mem_{len(self.records) + 1}", "text": text, "score": score, **metadata}
        append_jsonl(self.memory_path, row)
        self.records.append(TextRecord(item_id=row["id"], text=text, metadata={k: v for k, v in row.items() if k not in {"id", "text"}}))
        self.retriever = SparseTextRetriever(self.records)
        return row["id"]
