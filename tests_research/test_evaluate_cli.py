import json
import subprocess
import sys


def test_independent_evaluator_rejects_incomplete_expected_split(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("".join([
        json.dumps({"id": "q1", "question": "Q1?", "answer": "a"}) + "\n",
        json.dumps({"id": "q2", "question": "Q2?", "answer": "b"}) + "\n",
    ]), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "qid": "q1", "question": "Q1?", "status": "answer", "answer": "a",
        "confidence": 1, "stop_reason": "done",
    }) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"validation": ["q1", "q2"]}}), encoding="utf-8")
    completed = subprocess.run([
        sys.executable, "-m", "tdca_research.evaluate", "--dataset_path", str(dataset),
        "--dataset", "musique", "--predictions", str(predictions),
        "--output", str(tmp_path / "eval.json"), "--expected-qids", str(manifest),
        "--split", "validation",
    ], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "expected split" in completed.stderr
