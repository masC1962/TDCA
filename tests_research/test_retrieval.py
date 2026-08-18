import importlib.util

import pytest

from tdca_research.models import Passage
from tdca_research.retrieval import BM25Retriever, DenseDependencyError, DenseRetriever, EntityAwareRetriever


PASSAGES = [
    Passage("a", "City", "The inventor was born in River City."),
    Passage("b", "Other", "A mountain contains snow."),
]


def test_bm25_returns_raw_non_normalized_scores():
    hits = BM25Retriever(PASSAGES).search("inventor born", 2)
    assert hits[0].passage.passage_id == "a"
    assert hits[0].raw_score > hits[1].raw_score
    assert hits[0].raw_score != 1.0


def test_dense_does_not_silently_fallback():
    if importlib.util.find_spec("sentence_transformers") is None:
        with pytest.raises(DenseDependencyError):
            DenseRetriever(PASSAGES, "missing", fallback="error")


def test_dense_explicit_fallback_is_labeled(monkeypatch):
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        "tdca_research.retrieval.dense.importlib.util.find_spec",
        lambda name: None if name == "sentence_transformers" else original_find_spec(name),
    )
    retriever = DenseRetriever(PASSAGES, "missing", fallback="explicit_tfidf")
    assert "explicit_tfidf_fallback" in retriever.search("inventor", 1)[0].retriever


def test_entity_aware_retrieval_uses_query_surface_forms_without_gold_labels():
    passages = [Passage("a", "Unrelated", "Ada is mentioned once."), Passage("b", "Ada Lovelace", "biography")]
    hits = EntityAwareRetriever(BM25Retriever(passages), entity_weight=10.0).search("Where was Ada Lovelace born?", 1)
    assert hits[0].passage.passage_id == "b"
    assert hits[0].retriever.startswith("entity_aware")


def test_entity_surface_forms_support_apostrophes_and_hyphens_without_lowercase_noise():
    entities = EntityAwareRetriever._entities("Where did D’Arcy Smith meet Anne-Marie in the city?")
    assert "d’arcy smith" in entities
    assert "annemarie" in entities
    assert all("city" not in entity for entity in entities)
