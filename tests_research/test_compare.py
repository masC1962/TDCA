import json

import pytest

from tdca_research.compare import compare


def test_self_comparison_has_exact_zero_interval(tmp_path):
    artifact = tmp_path / "rows.jsonl"
    rows = [
        {"qid": "a", "status": "answer", "exact_match": 1, "f1": 0.75, "total_tokens": 10},
        {"qid": "b", "status": "abstain", "exact_match": 0, "f1": 0.25, "total_tokens": 20},
    ]
    artifact.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = compare(artifact, artifact, samples=100)
    interval = result["paired_bootstrap_exact_match"]
    assert result["aligned_count"] == 2
    assert interval["mean_difference"] == 0
    assert interval["lower_95"] == 0
    assert interval["upper_95"] == 0
    assert result["paired_bootstrap_f1"]["mean_difference"] == 0
    assert result["paired_bootstrap_f1"]["lower_95"] == 0
    assert result["paired_bootstrap_f1"]["upper_95"] == 0
    assert result["quality_cost_summary"]["base"] == result["quality_cost_summary"]["new"]


def test_paired_comparison_rejects_duplicates_and_partial_intersection(tmp_path):
    base = tmp_path / "base.jsonl"
    new = tmp_path / "new.jsonl"
    base.write_text('\n'.join([
        json.dumps({"qid": "a", "exact_match": 1}),
        json.dumps({"qid": "a", "exact_match": 0}),
    ]) + '\n', encoding="utf-8")
    new.write_text(json.dumps({"qid": "a", "exact_match": 1}) + '\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        compare(base, new)
    base.write_text(json.dumps({"qid": "b", "exact_match": 1}) + '\n', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the same"):
        compare(base, new)
