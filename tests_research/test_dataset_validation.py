import pytest

from tdca_research.data import DatasetIntegrityError, validate_dataset_integrity
from tdca_research.models import Passage, QAExample


def test_support_only_subset_cannot_be_reported_as_distractor():
    examples = [QAExample("q", "?", [Passage("g", "Gold", "text")], answers=["a"], gold_document_ids=["g"])]
    with pytest.raises(DatasetIntegrityError, match="support-only"):
        validate_dataset_integrity(examples, "distractor")


def test_real_distractor_shape_passes_integrity_gate():
    examples = [QAExample(
        "q", "?", [Passage("g", "Gold", "text"), Passage("n", "Noise", "noise")],
        answers=["a"], gold_document_ids=["g"],
    )]
    report = validate_dataset_integrity(examples, "distractor")
    assert report["no_distractor_ids"] == []


def test_global_question_file_may_omit_per_question_passages():
    report = validate_dataset_integrity([QAExample("q", "?", [], answers=["a"])], "global")
    assert report["missing_passage_ids"] == ["q"]


def test_integrity_rejects_duplicate_question_ids():
    examples = [QAExample("q", "?", [], answers=["a"]), QAExample("q", "?", [], answers=["b"])]
    with pytest.raises(DatasetIntegrityError, match="duplicate question"):
        validate_dataset_integrity(examples, "global")


def test_integrity_rejects_missing_answers_before_evaluation():
    with pytest.raises(DatasetIntegrityError, match="no gold answer"):
        validate_dataset_integrity([QAExample("q", "?", [])], "global")


def test_integrity_rejects_gold_id_absent_from_distractor_context():
    example = QAExample(
        "q", "?", [Passage("n", "Noise", "noise")], answers=["a"], gold_document_ids=["g"],
    )
    with pytest.raises(DatasetIntegrityError, match="absent"):
        validate_dataset_integrity([example], "distractor")


def test_integrity_rejects_missing_gold_evidence_for_distractor_evaluation():
    example = QAExample("q", "?", [Passage("n", "Noise", "noise")], answers=["a"])
    with pytest.raises(DatasetIntegrityError, match="no gold evidence"):
        validate_dataset_integrity([example], "distractor")
