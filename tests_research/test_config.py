import pytest

from tdca_research.config import ResearchConfig


def test_cli_top_k_has_one_effective_field():
    config = ResearchConfig(top_k=9)
    assert config.top_k == 9
    assert "retrieve_top_k_evidence" not in config.to_dict()


def test_persistent_memory_is_disabled_to_prevent_cross_test_leakage():
    with pytest.raises(ValueError):
        ResearchConfig(persistent_episodic_memory=True)


def test_global_setting_requires_a_corpus_path():
    with pytest.raises(ValueError, match="global_corpus_path"):
        ResearchConfig(setting="global")
