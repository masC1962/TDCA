import json

import pytest

from scripts.verify_artifact import verify
from tdca_research.config import ResearchConfig
from tdca_research.experiments import ArtifactWriter
from tdca_research.models import Prediction, RunStatus


def _complete_artifact(path):
    writer = ArtifactWriter(path)
    writer.initialize(ResearchConfig(), {
        "experiment_id": "test", "sample_ids": ["q1"],
    }, {"api_key_present": False})
    prediction = Prediction("q1", "Question?", RunStatus.ANSWER, "answer", 0.8, "done")
    writer.write_predictions([prediction])
    writer.write_rows("retrieval_traces.jsonl", [])
    writer.write_rows("reasoning_traces.jsonl", [])
    writer.write_rows("per_example_metrics.jsonl", [{"qid": "q1"}])
    writer.write_metrics({"count": 1}, {}, [prediction])
    writer.finalize_manifest()
    writer.checksums()


def test_verify_artifact_accepts_complete_consistent_run(tmp_path):
    _complete_artifact(tmp_path)
    assert verify(tmp_path, 1)["verified"] is True


def test_verify_artifact_rejects_post_checksum_mutation(tmp_path):
    _complete_artifact(tmp_path)
    (tmp_path / "metrics.json").write_text(json.dumps({"count": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path, 1)


def test_legacy_schema_requires_explicit_opt_in(tmp_path):
    _complete_artifact(tmp_path)
    (tmp_path / "metrics_by_type.json").unlink()
    (tmp_path / "partial_progress.json").unlink()
    # Rebuild the historical checksum list after simulating an older writer.
    ArtifactWriter(tmp_path).checksums()
    with pytest.raises(ValueError, match="required artifacts"):
        verify(tmp_path, 1)
    report = verify(tmp_path, 1, allow_legacy_schema=True)
    assert report["schema"] == "legacy_pre_checkpoint"
