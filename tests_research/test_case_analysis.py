from tdca_research.case_analysis import _jsonl, categorize


def test_case_analysis_retrieval_miss_precedes_reasoning_categories():
    assert categorize({"exact_match": 0, "all_gold_recalled": 0}, {"status": "abstain"}) == "retrieval_miss"


def test_case_analysis_detects_answer_present_verifier_rejection():
    row = {"exact_match": 0, "all_gold_recalled": 1, "answer_in_context": 1}
    assert categorize(row, {"status": "abstain"}) == "verification_false_reject_or_missing_terminal"


def test_case_analysis_jsonl_rejects_duplicate_qids(tmp_path):
    import json
    import pytest

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"qid": "q"}) + "\n" for _ in range(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _jsonl(path)
