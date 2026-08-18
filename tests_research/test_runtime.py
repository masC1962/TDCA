import json

import pytest

from tdca_research.config import ResearchConfig
from tdca_research.llm import DeterministicMockLLM
from tdca_research.experiments import ArtifactWriter
from tdca_research.models import Prediction, RunStatus
from tdca_research.runtime import _code_version, _load_resume_state, _resolved_retriever_kind, run


def test_baseline_method_names_force_the_claimed_retriever():
    assert _resolved_retriever_kind(ResearchConfig(method="bm25_rag", retriever="hybrid")) == "bm25"
    assert _resolved_retriever_kind(ResearchConfig(method="dense_rag", retriever="bm25")) == "dense"
    assert _resolved_retriever_kind(ResearchConfig(method="hybrid_rag", retriever="bm25")) == "hybrid"
    assert _resolved_retriever_kind(ResearchConfig(method="structured_tdca", retriever="entity")) == "entity"


def test_runtime_rejects_manifest_for_a_different_dataset(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({
            "id": "q1", "question": "Who?", "answer": "A",
            "paragraphs": [
                {"idx": 0, "title": "Gold", "paragraph_text": "A", "is_supporting": True},
                {"idx": 1, "title": "Noise", "paragraph_text": "B", "is_supporting": False},
            ],
        }) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "seed": 520,
        "dataset_sha256": "0" * 64,
        "splits": {"smoke": ["q1"], "tuning": [], "validation": [], "final": []},
    }), encoding="utf-8")
    config = ResearchConfig(
        dataset_path=str(dataset), split="smoke", split_manifest_path=str(manifest),
        output_root=str(tmp_path / "outputs"), llm_backend="mock",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        run(config, mock=DeterministicMockLLM())


def test_resume_uses_only_durable_ordered_prefix_and_discards_torn_append(tmp_path):
    config = ResearchConfig(output_root=str(tmp_path))
    writer = ArtifactWriter(tmp_path)
    writer.initialize(config, {
        "experiment_id": "resume-test",
        "dataset_sha256": "abc",
        "sample_ids": ["q1", "q2"],
    }, {"api_key_present": False})
    first = Prediction("q1", "First?", RunStatus.ANSWER, "a", 0.8, "done")
    writer.checkpoint_prediction(first, [{"qid": "q1", "rank": 1}], [], 1, 2)
    # Simulate a process dying after the prediction append but before the
    # atomic progress checkpoint advanced.
    writer.append_rows("predictions.jsonl", [
        Prediction("q2", "Second?", RunStatus.ANSWER, "b", 0.8, "torn").to_dict()
    ])

    predictions, retrieval, reasoning = _load_resume_state(
        tmp_path, config, "abc", ["q1", "q2"], writer,
    )

    assert [prediction.qid for prediction in predictions] == ["q1"]
    assert [row["qid"] for row in retrieval] == ["q1"]
    assert reasoning == []
    assert len((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_unversioned_source_tree_gets_stable_content_hash(tmp_path):
    source = tmp_path / "src" / "tdca_research"
    source.mkdir(parents=True)
    (source / "module.py").write_text("value = 1\n", encoding="utf-8")
    first = _code_version(tmp_path)
    assert first.startswith("source-tree-sha256:")
    assert first == _code_version(tmp_path)
    (source / "module.py").write_text("value = 2\n", encoding="utf-8")
    assert first != _code_version(tmp_path)


def test_distractor_runtime_rejects_shared_dense_index_path_before_execution(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({
        "id": "q1", "question": "Who?", "answer": "A",
        "paragraphs": [
            {"idx": 0, "title": "Gold", "paragraph_text": "A", "is_supporting": True},
            {"idx": 1, "title": "Noise", "paragraph_text": "B", "is_supporting": False},
        ],
    }) + "\n", encoding="utf-8")
    config = ResearchConfig(
        dataset_path=str(dataset), split="smoke", output_root=str(tmp_path / "out"),
        retriever="dense", dense_index_path=str(tmp_path / "index"), llm_backend="mock",
    )
    with pytest.raises(ValueError, match="global corpus"):
        run(config, mock=DeterministicMockLLM())
