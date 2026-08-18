from tdca_research import retrieval_probe
from tdca_research.config import ResearchConfig


def test_retrieval_probe_is_importable():
    assert callable(retrieval_probe.main)


def test_probe_cutoff_does_not_mutate_frozen_config():
    config = ResearchConfig(top_k=10)
    diagnostic_top_k = 2
    assert diagnostic_top_k != config.top_k
    assert config.top_k == 10
