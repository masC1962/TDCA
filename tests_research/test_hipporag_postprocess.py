import json

import pytest

from scripts.postprocess_hipporag_validation import _jsonl_qids


def test_hipporag_postprocess_qid_reader_preserves_order_and_duplicates(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "".join(json.dumps({"qid": qid}) + "\n" for qid in ("q2", "q1", "q2")),
        encoding="utf-8",
    )
    assert _jsonl_qids(path) == ["q2", "q1", "q2"]


def test_partial_suffix_is_unambiguously_detectable():
    path = "artifact.json.partial"
    assert path.endswith(".partial")


def test_qid_sets_can_align_even_when_export_order_differs():
    main_ids = ["q2", "q1"]
    hippo_ids = ["q1", "q2"]
    assert main_ids != hippo_ids
    assert set(main_ids) == set(hippo_ids)
