from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import json
import numpy as np

from core_models import RetrievedContext
from utils import normalize_scores, simple_tokenize


@dataclass
class TextRecord:
    item_id: str
    text: str
    metadata: Dict[str, Any]


class BaseRetriever(Protocol):
    def search(self, query: str, top_k: int, source: str = "retrieval", return_raw: bool = False) -> List[RetrievedContext]:
        ...


class SparseTextRetriever:
    """
    A lightweight BM25-style sparse retriever implemented with only stdlib + numpy.
    """

    def __init__(self, records: List[TextRecord]) -> None:
        self.records = records
        self.doc_tokens: List[List[str]] = []
        self.doc_term_freqs: List[Counter[str]] = []
        self.doc_lens: List[int] = []
        self.df: Counter[str] = Counter()
        self.avgdl: float = 0.0
        self.num_docs: int = len(records)
        self.k1: float = 1.5
        self.b: float = 0.75

        if not records:
            return

        for record in records:
            tokens = simple_tokenize(record.text)
            tf = Counter(tokens)
            self.doc_tokens.append(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            self.df.update(tf.keys())

        self.avgdl = float(np.mean(self.doc_lens)) if self.doc_lens else 0.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df <= 0 or self.num_docs <= 0:
            return 0.0
        return log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def _score_doc(self, query_terms: List[str], idx: int) -> float:
        if idx >= len(self.doc_term_freqs):
            return 0.0
        tf = self.doc_term_freqs[idx]
        dl = max(1, self.doc_lens[idx])
        avgdl = max(1e-8, self.avgdl)
        score = 0.0
        qtf = Counter(query_terms)
        for term, qf in qtf.items():
            f = tf.get(term, 0)
            if f <= 0:
                continue
            idf = self._idf(term)
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            score += qf * idf * (f * (self.k1 + 1.0)) / max(1e-8, denom)
        return float(score)

    def search(self, query: str, top_k: int, source: str, return_raw: bool = False) -> List[RetrievedContext]:
        if not query or not self.records:
            return []

        query_terms = simple_tokenize(query)
        if not query_terms:
            return []

        raw_scores = np.array([self._score_doc(query_terms, i) for i in range(len(self.records))], dtype=float)
        if raw_scores.size == 0 or np.allclose(raw_scores, 0.0):
            return []

        result_k = max(0, min(len(self.records), top_k))
        top_idx = np.argsort(raw_scores)[::-1][:result_k]
        top_raw = [float(raw_scores[i]) for i in top_idx]
        norm_scores = normalize_scores(top_raw)

        results: List[RetrievedContext] = []
        for rank_idx, idx in enumerate(top_idx):
            score = float(raw_scores[idx])
            if score <= 0:
                continue
            record = self.records[idx]
            metadata = dict(record.metadata)
            metadata["raw_score"] = score
            results.append(
                RetrievedContext(
                    item_id=record.item_id,
                    text=record.text,
                    score=(score if return_raw else (norm_scores[rank_idx] if rank_idx < len(norm_scores) else score)),
                    source=source,
                    metadata=metadata,
                )
            )
        return results


class DenseTextRetriever:
    """
    Dense retriever with two backends:
    1) sentence-transformers if installed and encoder_path is provided.
    2) sklearn TF-IDF fallback for environments without dense deps.

    It can either build embeddings from in-memory records or load a prebuilt .npz index
    produced by scripts/build_dense_index.py.
    """

    def __init__(
        self,
        records: Optional[List[TextRecord]] = None,
        encoder_path: str = "",
        index_path: str = "",
    ) -> None:
        self.records: List[TextRecord] = records or []
        self.encoder_path = encoder_path
        self.index_path = index_path
        self.backend = "tfidf"
        self._st_model = None
        self._vectorizer = None
        self._doc_matrix = None
        self._doc_embeddings = None
        if index_path:
            self._load_index(index_path)
        elif self.records:
            self._fit_from_records(self.records)

    def _try_load_sentence_transformer(self) -> bool:
        if not self.encoder_path:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.encoder_path)
            self.backend = "sentence_transformers"
            return True
        except Exception:
            return False

    def _fit_from_records(self, records: List[TextRecord]) -> None:
        self.records = records
        texts = [r.text for r in records]
        if self._try_load_sentence_transformer():
            emb = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self._doc_embeddings = np.asarray(emb, dtype=np.float32)
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.backend = "tfidf"
        self._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=50000)
        self._doc_matrix = self._vectorizer.fit_transform(texts)

    def _load_index(self, index_path: str) -> None:
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(f"Dense index not found: {index_path}")
        data = np.load(path, allow_pickle=True)
        payload = data["payload"].tolist()
        self.backend = payload.get("backend", "tfidf")
        self.encoder_path = payload.get("encoder_path", self.encoder_path)
        self.records = [TextRecord(**row) for row in payload.get("records", [])]
        if self.backend == "sentence_transformers":
            self._doc_embeddings = data["doc_embeddings"].astype(np.float32)
            self._try_load_sentence_transformer()
        else:
            # Rebuild TF-IDF from records to avoid scipy sparse serialization complexity.
            self._fit_from_records(self.records)

    def save_index(self, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "encoder_path": self.encoder_path,
            "records": [
                {"item_id": r.item_id, "text": r.text, "metadata": r.metadata}
                for r in self.records
            ],
        }
        if self.backend == "sentence_transformers" and self._doc_embeddings is not None:
            np.savez_compressed(out, payload=payload, doc_embeddings=self._doc_embeddings)
        else:
            np.savez_compressed(out, payload=payload)

    def _score_st(self, query: str) -> np.ndarray:
        q = self._st_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q, dtype=np.float32)
        return np.matmul(self._doc_embeddings, q[0])

    def _score_tfidf(self, query: str) -> np.ndarray:
        qv = self._vectorizer.transform([query])
        sims = (self._doc_matrix @ qv.T).toarray().reshape(-1)
        return sims.astype(float)

    def search(self, query: str, top_k: int, source: str = "dense", return_raw: bool = False) -> List[RetrievedContext]:
        if not query or not self.records:
            return []
        if self.backend == "sentence_transformers":
            if self._st_model is None:
                if not self._try_load_sentence_transformer():
                    raise RuntimeError(
                        "sentence-transformers backend requested but package/model unavailable. "
                        "Install sentence-transformers or rebuild using TF-IDF fallback."
                    )
            raw_scores = self._score_st(query)
        else:
            raw_scores = self._score_tfidf(query)
        if raw_scores.size == 0 or np.allclose(raw_scores, 0.0):
            return []
        result_k = max(0, min(len(self.records), top_k))
        top_idx = np.argsort(raw_scores)[::-1][:result_k]
        top_raw = [float(raw_scores[i]) for i in top_idx]
        norm_scores = normalize_scores(top_raw)
        results: List[RetrievedContext] = []
        for rank_idx, idx in enumerate(top_idx):
            score = float(raw_scores[idx])
            if score <= 0:
                continue
            record = self.records[idx]
            metadata = dict(record.metadata)
            metadata["raw_score"] = score
            metadata["dense_backend"] = self.backend
            results.append(
                RetrievedContext(
                    item_id=record.item_id,
                    text=record.text,
                    score=(score if return_raw else (norm_scores[rank_idx] if rank_idx < len(norm_scores) else score)),
                    source=source,
                    metadata=metadata,
                )
            )
        return results


class HybridRetriever:
    def __init__(self, sparse: BaseRetriever, dense: BaseRetriever, alpha: float = 0.5) -> None:
        self.sparse = sparse
        self.dense = dense
        self.alpha = alpha

    def search(self, query: str, top_k: int, source: str = "hybrid", return_raw: bool = False) -> List[RetrievedContext]:
        sparse_hits = self.sparse.search(query, top_k=top_k * 2, source="sparse", return_raw=True)
        dense_hits = self.dense.search(query, top_k=top_k * 2, source="dense", return_raw=True)
        merged: Dict[str, RetrievedContext] = {}
        sparse_scores = {h.item_id: float(h.metadata.get("raw_score", h.score)) for h in sparse_hits}
        dense_scores = {h.item_id: float(h.metadata.get("raw_score", h.score)) for h in dense_hits}
        sparse_norm = normalize_scores(list(sparse_scores.values())) if sparse_scores else []
        dense_norm = normalize_scores(list(dense_scores.values())) if dense_scores else []
        sparse_norm_map = {k: sparse_norm[i] for i, k in enumerate(sparse_scores.keys())}
        dense_norm_map = {k: dense_norm[i] for i, k in enumerate(dense_scores.keys())}

        for hit in sparse_hits + dense_hits:
            if hit.item_id not in merged:
                merged[hit.item_id] = RetrievedContext(
                    item_id=hit.item_id,
                    text=hit.text,
                    score=0.0,
                    source=source,
                    metadata=dict(hit.metadata),
                )
            merged[hit.item_id].score = self.alpha * sparse_norm_map.get(hit.item_id, 0.0) + (1 - self.alpha) * dense_norm_map.get(hit.item_id, 0.0)
        out = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:top_k]
        if return_raw:
            return out
        norm = normalize_scores([x.score for x in out])
        for i, item in enumerate(out):
            item.score = norm[i] if i < len(norm) else item.score
        return out


def records_from_jsonl(path: str | Path) -> List[TextRecord]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out: List[TextRecord] = []
    for row in rows:
        item_id = str(row.get("id") or row.get("doc_id") or len(out))
        text = str(row.get("text") or row.get("paragraph_text") or "")
        metadata = {k: v for k, v in row.items() if k not in {"id", "doc_id", "text", "paragraph_text"}}
        out.append(TextRecord(item_id=item_id, text=text, metadata=metadata))
    return out


def build_retriever(
    retriever_type: str,
    records: Optional[Sequence[TextRecord]] = None,
    index_path: str = "",
    encoder_path: str = "",
    hybrid_alpha: float = 0.5,
) -> BaseRetriever:
    records_list = list(records) if records is not None else []
    retriever_type = (retriever_type or "sparse").strip().lower()
    if retriever_type == "sparse":
        return SparseTextRetriever(records_list)
    if retriever_type == "dense":
        return DenseTextRetriever(records=records_list or None, encoder_path=encoder_path, index_path=index_path)
    if retriever_type == "hybrid":
        sparse = SparseTextRetriever(records_list)
        dense = DenseTextRetriever(records=records_list or None, encoder_path=encoder_path, index_path=index_path)
        return HybridRetriever(sparse=sparse, dense=dense, alpha=hybrid_alpha)
    raise ValueError(f"Unsupported retriever_type: {retriever_type}")
