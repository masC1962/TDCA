from tdca_research.evaluation import exact_match, grouped_metrics, retrieval_scores, token_f1


def test_exact_match_preserves_official_empty_string_semantics():
    assert exact_match("", [""]) == 1.0
    assert exact_match("", ["answer"]) == 0.0


def test_official_style_answer_metrics_do_not_use_substring_soft_em():
    assert exact_match("Paris is the answer", ["Paris"]) == 0
    assert exact_match("The Paris", ["Paris"]) == 1
    assert 0 < token_f1("Paris France", ["Paris"]) < 1


def test_complete_support_recall_is_distinct_from_any_hit():
    scores = retrieval_scores(["a"], ["a", "b"])
    assert scores["support_recall"] == 0.5
    assert scores["all_gold_recalled"] == 0
    assert scores["recall_at_1"] == 0.5


def test_ordered_evidence_path_recall_penalizes_reversed_chain():
    forward = retrieval_scores(["a", "x", "b"], ["a", "b"])
    reverse = retrieval_scores(["b", "a"], ["a", "b"])
    assert forward["ordered_evidence_path_recall"] == 1.0
    assert reverse["ordered_evidence_path_recall"] == 0.5


def test_official_musique_unicode_and_punctuation_normalization_parity():
    assert exact_match("Małgorzata Braunek!", ["Małgorzata Braunek"]) == 1
    assert token_f1("The café, Paris", ["café Paris"]) == 1


def test_hotpot_and_2wiki_categorical_answers_get_no_partial_credit():
    assert exact_match("yes indeed", ["yes"]) == 0
    assert token_f1("yes indeed", ["yes"]) == 0
    assert token_f1("no", ["yes"]) == 0
    assert token_f1("no answer", ["noanswer"]) == 0
    assert token_f1("Paris France", ["Paris"]) > 0


def test_grouped_metrics_uses_one_definition_for_runtime_and_rescoring():
    rows = [
        {"hop_count": 2, "question_type": "bridge", "status": "answer", "exact_match": 1, "f1": 1,
         "support_recall": 0.5, "all_gold_recalled": 0, "total_tokens": 100},
        {"hop_count": 2, "question_type": "bridge", "status": "abstain", "exact_match": 0, "f1": 0,
         "support_recall": 1, "all_gold_recalled": 1, "total_tokens": 200},
    ]
    group = grouped_metrics(rows, "question_type")["bridge"]
    assert group["count"] == 2
    assert group["exact_match"] == 0.5
    assert group["answered_rate"] == 0.5
    assert group["support_recall"] == 0.75
    assert group["total_tokens"] == 150
