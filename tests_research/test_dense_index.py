from pathlib import Path

import pytest

from tdca_research.models import Passage
from tdca_research.retrieval import DenseDependencyError, DenseRetriever


def test_tfidf_dense_fallback_index_roundtrip_and_fingerprint(tmp_path: Path, monkeypatch):
    original_find_spec = __import__("importlib").util.find_spec
    monkeypatch.setattr(
        "tdca_research.retrieval.dense.importlib.util.find_spec",
        lambda name: None if name == "sentence_transformers" else original_find_spec(name),
    )
    passages = [
        Passage("a", "Alpha", "Alpha contains apples."),
        Passage("b", "Beta", "Beta contains bananas."),
    ]
    original = DenseRetriever(passages, "test-encoder", fallback="explicit_tfidf")
    original.save(tmp_path)
    loaded = DenseRetriever(passages, "test-encoder", fallback="error", index_path=str(tmp_path))
    assert loaded.search("bananas", 1)[0].passage.passage_id == "b"
    with pytest.raises(DenseDependencyError, match="fingerprint"):
        DenseRetriever([Passage("a", "Alpha", "changed")], "test-encoder", index_path=str(tmp_path))


def test_dense_model_cache_is_process_scoped_by_model_name():
    assert isinstance(DenseRetriever._model_cache, dict)
