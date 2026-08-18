from __future__ import annotations

import math
from collections import Counter

from ..models import Passage, RetrievalHit
from ..utils import tokenize
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, passages: list[Passage], k1: float = 1.2, b: float = 0.75) -> None:
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(f"{passage.title} {passage.text}") for passage in passages]
        self.counts = [Counter(tokens) for tokens in self.tokens]
        self.average_length = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    def _idf(self, token: str) -> float:
        n = len(self.passages)
        df = self.document_frequency[token]
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_tokens = tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self.counts):
            length = len(self.tokens[index])
            score = 0.0
            for token in query_tokens:
                frequency = counts[token]
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / max(1.0, self.average_length))
                score += self._idf(token) * frequency * (self.k1 + 1) / denominator
            scored.append((score, index))
        ranked = sorted(scored, key=lambda pair: (-pair[0], self.passages[pair[1]].passage_id))[:top_k]
        return [
            RetrievalHit(passage=self.passages[index], raw_score=float(score), rank=rank, retriever=self.name, query=query)
            for rank, (score, index) in enumerate(ranked, start=1)
        ]

