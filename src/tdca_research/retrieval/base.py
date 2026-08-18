from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Passage, RetrievalHit


class DenseDependencyError(RuntimeError):
    pass


class BaseRetriever(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        raise NotImplementedError


def build_retriever(kind: str, passages: list[Passage], dense_model: str = "sentence-transformers/all-MiniLM-L6-v2", dense_fallback: str = "error", dense_index_path: str = "") -> BaseRetriever:
    from .bm25 import BM25Retriever
    from .dense import DenseRetriever
    from .hybrid import HybridRetriever
    from .entity import EntityAwareRetriever

    normalized = kind.strip().lower()
    if normalized in {"bm25", "sparse"}:
        return BM25Retriever(passages)
    if normalized == "dense":
        return DenseRetriever(passages, model_name=dense_model, fallback=dense_fallback, index_path=dense_index_path)
    if normalized == "hybrid":
        return HybridRetriever(
            BM25Retriever(passages),
            DenseRetriever(passages, model_name=dense_model, fallback=dense_fallback, index_path=dense_index_path),
        )
    if normalized in {"entity", "entity_aware"}:
        return EntityAwareRetriever(BM25Retriever(passages))
    raise ValueError(f"unknown retriever {kind}")
