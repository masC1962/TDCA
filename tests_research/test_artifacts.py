import json

from tdca_research.experiments import ArtifactWriter
from tdca_research.models import Prediction, RunStatus, Usage
from tdca_research.utils import write_json


def test_cost_artifact_refuses_to_invent_unversioned_price(tmp_path):
    writer = ArtifactWriter(tmp_path)
    prediction = Prediction(
        "q",
        "Question?",
        RunStatus.ANSWER,
        "answer",
        0.5,
        "test",
        usage=Usage(llm_calls=1, prompt_tokens=10, completion_tokens=2),
    )
    writer.write_metrics({}, {}, [prediction])
    estimate = json.loads((tmp_path / "estimated_cost.json").read_text(encoding="utf-8"))
    assert estimate["status"] == "not_computed"
    assert estimate["amount"] is None
    assert estimate["prompt_tokens"] == 10
    assert estimate["provider_prompt_tokens"] == 0
    assert json.loads((tmp_path / "metrics_by_type.json").read_text(encoding="utf-8")) == {}


def test_prediction_checkpoint_is_append_only_and_tracks_progress(tmp_path):
    writer = ArtifactWriter(tmp_path)
    prediction = Prediction("q1", "Question?", RunStatus.ABSTAIN, None, 0.2, "insufficient_evidence")
    writer.checkpoint_prediction(prediction, [{"qid": "q1", "rank": 1}], [], 1, 3)
    writer.checkpoint_prediction(
        Prediction("q2", "Second?", RunStatus.INFRASTRUCTURE_FAILURE, None, 0.0, "api_failure"),
        [], [], 2, 3,
    )
    rows = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    progress = json.loads((tmp_path / "partial_progress.json").read_text(encoding="utf-8"))
    assert progress == {**progress, "status": "running", "completed": 2, "total": 3, "last_qid": "q2"}
    assert len((tmp_path / "failures.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_initialized_artifact_contains_every_required_file(tmp_path):
    from tdca_research.config import ResearchConfig

    writer = ArtifactWriter(tmp_path)
    writer.initialize(ResearchConfig(), {"experiment_id": "test"}, {"api_key_present": False})
    missing = [name for name in writer.FILES if not (tmp_path / name).exists()]
    assert missing == []


def test_json_checkpoint_replace_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "progress.json"
    write_json(target, {"completed": 1})
    write_json(target, {"completed": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 2}
    assert list(tmp_path.glob(".progress.json.*.tmp")) == []
