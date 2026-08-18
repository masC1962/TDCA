import json

from tdca_research.data.loader import parse_example
from tdca_research.data.splits import build_split_manifest
from tdca_research.evaluation.metrics import evaluate_predictions, exact_match, token_f1
from tdca_research.models import Prediction, QAExample, RunStatus


def test_disjoint_manifest_has_no_overlap_and_is_reproducible():
    examples = [
        parse_example(
            {
                "id": f"q{index}",
                "question": f"Question {index}?",
                "answer": "answer",
                "paragraphs": [
                    {"idx": 0, "title": "Gold", "paragraph_text": "answer", "is_supporting": True},
                    {"idx": 1, "title": "Distractor", "paragraph_text": "noise", "is_supporting": False},
                ],
            },
            "musique",
        )
        for index in range(300)
    ]
    first = build_split_manifest(examples, seed=520)
    second = build_split_manifest(examples, seed=520)
    assert first == second
    split_sets = [set(ids) for ids in first["splits"].values()]
    for index, left in enumerate(split_sets):
        for right in split_sets[index + 1 :]:
            assert left.isdisjoint(right)


def test_inference_view_excludes_all_evaluation_and_oracle_labels():
    example = parse_example(
        {
            "id": "held-out",
            "type": "comparison",
            "question": "Which one?",
            "answer": "Secret answer",
            "answer_aliases": ["Secret alias"],
            "supporting_facts": [["Gold", 0]],
            "context": [["Gold", ["Secret answer appears here."]], ["Noise", ["noise"]]],
            "question_decomposition": [{"question": "Gold plan", "answer": "Secret bridge"}],
        },
        "2wikimultihopqa",
    )
    serialized = json.dumps(example.inference_view())
    assert "comparison" not in serialized
    assert "Secret alias" not in serialized
    assert "Gold plan" not in serialized
    assert "Secret bridge" not in serialized
    assert "hop_count" not in serialized


def test_nested_metadata_support_is_loaded_without_exposing_gold():
    row = {
        "id": "q1",
        "question": "Who is related to the performer?",
        "golden_answers": ["Hidden Gold"],
        "metadata": {"question_decomposition": [{
            "id": 1,
            "question": "first hop",
            "answer": "Hidden Bridge",
            "support_paragraph": {"title": "Evidence", "paragraph_text": "A performer is related to a person."},
        }]},
    }
    example = parse_example(row, "musique")
    assert len(example.passages) == 1
    assert example.gold_titles == ["Evidence"]
    serialized = json.dumps(example.inference_view())
    assert "Hidden Gold" not in serialized
    assert "Hidden Bridge" not in serialized
    assert "question_decomposition" not in serialized


def test_top_level_paragraphs_and_aliases():
    row = {
        "id": "x",
        "question": "Where?",
        "answer": "Paris",
        "answer_aliases": ["Paris, France"],
        "paragraphs": [{"idx": 7, "title": "Paris", "paragraph_text": "Paris is in France.", "is_supporting": True}],
    }
    example = parse_example(row, "musique")
    assert example.answers == ["Paris", "Paris, France"]
    assert example.gold_document_ids == ["7"]


def test_duplicate_titles_do_not_expand_musique_gold_ids():
    row = {
        "id": "q-duplicate-title",
        "question": "When did the founder die?",
        "answer": "1572",
        "paragraphs": [
            {"idx": 7, "title": "Presbyterianism", "paragraph_text": "John Knox founded it.", "is_supporting": True},
            {"idx": 8, "title": "Presbyterianism", "paragraph_text": "Unrelated same-title chunk.", "is_supporting": False},
            {"idx": 15, "title": "Presbyterianism", "paragraph_text": "John Knox died in 1572.", "is_supporting": True},
        ],
        "question_decomposition": [{"question": "Who founded it?"}, {"question": "When did #1 die?"}],
    }
    example = parse_example(row, "musique")
    assert example.gold_titles == ["Presbyterianism"]
    assert example.gold_document_ids == ["7", "15"]
    assert example.hop_count == 2


def test_2wiki_parallel_context_and_nested_supporting_facts():
    row = {
        "id": "2w",
        "type": "bridge_comparison",
        "question": "Who?",
        "answer": "A",
        "metadata": {
            "supporting_facts": {"title": ["Gold One", "Gold Two"], "sent_id": [0, 1]},
            "context": {
                "title": ["Gold One", "Distractor", "Gold Two"],
                "content": [["first evidence"], ["noise"], ["second evidence"]],
            },
        },
    }
    example = parse_example(row, "2wikimultihopqa")
    assert [passage.title for passage in example.passages] == ["Gold One", "Distractor", "Gold Two"]
    assert example.gold_document_ids == ["0", "2"]
    assert example.hop_count == 2
    assert example.metadata["question_type"] == "bridge_comparison"
    assert "bridge_comparison" not in json.dumps(example.inference_view())


def test_hotpot_parallel_titles_and_sentences_are_passages():
    row = {
        "id": "hp", "question": "Where?", "answer": "Here",
        "metadata": {
            "supporting_facts": {"title": ["Gold"], "sent_id": [0]},
            "context": {"title": ["Gold", "Noise"], "sentences": [["Here is evidence."], ["noise"]]},
        },
    }
    example = parse_example(row, "hotpotqa")
    assert len(example.passages) == 2
    assert example.gold_document_ids == ["0"]


def test_english_official_answer_normalization_matches_hotpot_style():
    assert exact_match("The Eiffel Tower", ["Eiffel Tower"]) == 1.0
    assert token_f1("Paris, France", ["Paris France"]) == 1.0


def test_official_answer_normalization_does_not_use_substring_soft_em():
    assert exact_match("The correct answer is Paris", ["Paris"]) == 0.0
    assert token_f1("The correct answer is Paris", ["Paris"]) < 1.0


def test_missing_gold_answer_never_counts_as_answer_in_context():
    example = QAExample("missing", "Question?", [], answers=[])
    prediction = Prediction("missing", "Question?", RunStatus.ABSTAIN, None, 0.0, "no_answer")
    metrics, rows = evaluate_predictions([example], [prediction])
    assert rows[0]["answer_in_context"] == 0.0
    assert metrics["answer_in_context_rate"] == 0.0
