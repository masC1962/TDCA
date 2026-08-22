from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterable

from ..models import Passage, RetrievalHit
from ..retrieval import BaseRetriever
from ..utils import normalize_text, tokenize


@dataclass(frozen=True)
class CorpusPassageRecord:
    passage_id: str
    title: str
    text_digest: str
    entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class CorpusEntityRecord:
    entity_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    passage_ids: tuple[str, ...]


@dataclass(frozen=True)
class ActivatedPassage:
    passage_id: str
    evidence_node_id: str
    subgoal_id: str
    branch_id: str
    query: str
    rank: int
    score: float
    entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class ActivatedEntity:
    entity_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    passage_ids: tuple[str, ...]
    query_overlap: float


@dataclass(frozen=True)
class MemoryActivation:
    passages: tuple[ActivatedPassage, ...]
    entities: tuple[ActivatedEntity, ...]
    edges: tuple[dict[str, str], ...]

    def to_payload(self) -> dict:
        return {
            "passages": [row.__dict__ | {"entity_ids": list(row.entity_ids)} for row in self.passages],
            "entities": [row.__dict__ | {
                "aliases": list(row.aliases), "passage_ids": list(row.passage_ids),
            } for row in self.entities],
            "edges": [dict(row) for row in self.edges],
        }


@dataclass
class RelationLightCorpusMemory:
    """Immutable passage/entity index with no generated relations or labels.

    The index deliberately retains original passages and only adds surface-form
    association edges.  Missing or noisy OpenIE relations therefore cannot erase
    evidence before the question-local proof graph is constructed.
    """

    passages: dict[str, CorpusPassageRecord]
    entities: dict[str, CorpusEntityRecord]
    fingerprint: str
    _passage_text: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, passages: Iterable[Passage]) -> "RelationLightCorpusMemory":
        passage_rows = list(passages)
        aliases_by_entity: dict[str, set[str]] = defaultdict(set)
        passages_by_entity: dict[str, set[str]] = defaultdict(set)
        entity_ids_by_passage: dict[str, list[str]] = {}
        text_by_passage: dict[str, str] = {}
        digest = sha256()
        for passage in sorted(passage_rows, key=lambda value: value.passage_id):
            digest.update(
                f"{passage.passage_id}\0{passage.title}\0{passage.text}\0".encode("utf-8")
            )
            text_by_passage[passage.passage_id] = passage.text
            mentions = _surface_entities(passage.title, passage.text)
            entity_ids = []
            for mention in mentions:
                canonical = normalize_text(mention)
                if not canonical:
                    continue
                entity_id = f"entity_{sha256(canonical.encode('utf-8')).hexdigest()[:16]}"
                aliases_by_entity[entity_id].add(mention.strip())
                passages_by_entity[entity_id].add(passage.passage_id)
                entity_ids.append(entity_id)
            entity_ids_by_passage[passage.passage_id] = list(dict.fromkeys(entity_ids))
        entities = {
            entity_id: CorpusEntityRecord(
                entity_id=entity_id,
                canonical_name=min(
                    aliases, key=lambda value: (len(value), normalize_text(value), value)
                ),
                aliases=tuple(sorted(aliases, key=lambda value: (normalize_text(value), value))),
                passage_ids=tuple(sorted(passages_by_entity[entity_id])),
            )
            for entity_id, aliases in aliases_by_entity.items()
        }
        records = {
            passage.passage_id: CorpusPassageRecord(
                passage_id=passage.passage_id,
                title=passage.title,
                text_digest=sha256(passage.text.encode("utf-8")).hexdigest(),
                entity_ids=tuple(entity_ids_by_passage[passage.passage_id]),
            )
            for passage in passage_rows
        }
        return cls(records, entities, digest.hexdigest(), text_by_passage)

    @classmethod
    def from_retriever(cls, retriever: BaseRetriever) -> "RelationLightCorpusMemory":
        passages = _retriever_passages(retriever)
        if passages is None:
            raise TypeError("retriever does not expose an immutable passage collection")
        return cls.build(passages)

    def activate(
        self,
        hits: list[RetrievalHit],
        evidence_node_ids: list[str],
        question: str,
        subgoal_id: str,
        branch_id: str,
        query: str,
    ) -> MemoryActivation:
        if len(hits) != len(evidence_node_ids):
            raise ValueError("activation hits and evidence node IDs must align")
        query_tokens = set(tokenize(f"{question} {query}"))
        activated_passages: list[ActivatedPassage] = []
        entity_passages: dict[str, set[str]] = defaultdict(set)
        for hit, evidence_node_id in zip(hits, evidence_node_ids):
            record = self.passages.get(hit.passage.passage_id)
            entity_ids = record.entity_ids if record is not None else ()
            activated_passages.append(ActivatedPassage(
                passage_id=hit.passage.passage_id,
                evidence_node_id=evidence_node_id,
                subgoal_id=subgoal_id,
                branch_id=branch_id,
                query=query,
                rank=hit.rank,
                score=float(hit.raw_score),
                entity_ids=entity_ids,
            ))
            for entity_id in entity_ids:
                entity_passages[entity_id].add(hit.passage.passage_id)
        activated_entities = []
        for entity_id in sorted(entity_passages):
            entity = self.entities[entity_id]
            alias_tokens = set(tokenize(" ".join(entity.aliases)))
            overlap = len(query_tokens & alias_tokens) / max(1, len(alias_tokens))
            activated_entities.append(ActivatedEntity(
                entity_id=entity_id,
                canonical_name=entity.canonical_name,
                aliases=entity.aliases,
                passage_ids=tuple(sorted(entity_passages[entity_id])),
                query_overlap=min(1.0, float(overlap)),
            ))
        edges = []
        for passage in activated_passages:
            for entity_id in passage.entity_ids:
                edges.append({
                    "source": entity_id,
                    "target": passage.evidence_node_id,
                    "edge_type": "entity_mentioned_in_evidence",
                })
        return MemoryActivation(
            tuple(activated_passages), tuple(activated_entities),
            tuple(sorted(edges, key=lambda row: (row["source"], row["target"]))),
        )

    def entity_id(self, surface: str) -> str:
        normalized = normalize_text(surface)
        for entity_id, row in self.entities.items():
            if any(normalize_text(alias) == normalized for alias in row.aliases):
                return entity_id
        return f"entity_{sha256(normalized.encode('utf-8')).hexdigest()[:16]}" if normalized else ""


def _retriever_passages(retriever: BaseRetriever) -> list[Passage] | None:
    direct = getattr(retriever, "passages", None)
    if isinstance(direct, list):
        return direct
    for attribute in ("sparse", "dense", "base"):
        child = getattr(retriever, attribute, None)
        if child is None:
            continue
        rows = _retriever_passages(child)
        if rows is not None:
            return rows
    return None


def _surface_entities(title: str, text: str) -> list[str]:
    # Titles are the most reliable relation-free entity anchors.  Proper-noun,
    # acronym, year and numeric mentions add associations without claiming a
    # semantic relation.  No dataset IDs, answers or labels enter this function.
    rows = [title.strip()] if title.strip() else []
    token = r"(?:[A-Z][\w'’\-]*|[A-Z]{2,}|\d{3,4})"
    rows.extend(re.findall(rf"(?<!\w){token}(?:\s+{token}){{0,5}}", text))
    return [value for value in dict.fromkeys(rows) if normalize_text(value)]
