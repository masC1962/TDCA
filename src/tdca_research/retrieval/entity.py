from __future__ import annotations

import re

from ..models import RetrievalHit
from ..utils import normalize_text
from .base import BaseRetriever


class EntityAwareRetriever(BaseRetriever):
    """Generic entity-preserving reranker over any first-stage retriever."""

    name = "entity_aware"

    def __init__(self, base: BaseRetriever, fetch_multiplier: int = 4, entity_weight: float = 0.25) -> None:
        self.base = base
        self.fetch_multiplier = fetch_multiplier
        self.entity_weight = entity_weight

    @staticmethod
    def _entities(query: str) -> list[str]:
        # Generic surface-form candidates: consecutive capitalized tokens with
        # common ASCII/Unicode apostrophes and hyphens. No dataset lexicon or
        # entity-specific exception is used.
        token = r"[A-Z][\w'’\-]*"
        candidates = re.findall(rf"\b{token}(?:\s+{token}){{0,5}}\b", query)
        normalized = [normalize_text(value) for value in candidates]
        return [value for value in dict.fromkeys(normalized) if len(value) >= 3]

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        candidates = self.base.search(query, max(top_k, top_k * self.fetch_multiplier))
        entities = self._entities(query)
        rescored = []
        for hit in candidates:
            document = normalize_text(f"{hit.passage.title} {hit.passage.text}")
            matches = sum(entity in document for entity in entities)
            rescored.append((hit.raw_score + self.entity_weight * matches, hit))
        rescored.sort(key=lambda item: (-item[0], item[1].passage.passage_id))
        return [
            RetrievalHit(hit.passage, float(score), rank, f"entity_aware({self.base.name})", query)
            for rank, (score, hit) in enumerate(rescored[:top_k], start=1)
        ]
