from __future__ import annotations

from ..models import RetrievalHit
from .base import BaseRetriever


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, sparse: BaseRetriever, dense: BaseRetriever, sparse_weight: float = 0.5) -> None:
        self.sparse = sparse
        self.dense = dense
        self.sparse_weight = sparse_weight

    @staticmethod
    def _rrf(rank: int, constant: int = 60) -> float:
        return 1.0 / (constant + rank)

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        fetch = max(top_k * 3, top_k)
        sparse_hits = self.sparse.search(query, fetch)
        dense_hits = self.dense.search(query, fetch)
        passages = {hit.passage.passage_id: hit.passage for hit in sparse_hits + dense_hits}
        scores: dict[str, float] = {key: 0.0 for key in passages}
        for hit in sparse_hits:
            scores[hit.passage.passage_id] += self.sparse_weight * self._rrf(hit.rank)
        for hit in dense_hits:
            scores[hit.passage.passage_id] += (1 - self.sparse_weight) * self._rrf(hit.rank)
        order = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
        return [
            RetrievalHit(passages[key], scores[key], rank, f"hybrid({self.sparse.name},{self.dense.name})", query)
            for rank, key in enumerate(order, start=1)
        ]

