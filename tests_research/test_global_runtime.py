import json

from tdca_research.config import ResearchConfig
from tdca_research.llm import DeterministicMockLLM
from tdca_research.runtime import run


def test_global_runtime_uses_shared_corpus_without_per_question_passages(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text("\n".join([
        json.dumps({"id": "q1", "question": "Where?", "answer": "Paris", "supporting_facts": [["France", 0]]}),
        json.dumps({"id": "q2", "question": "Who?", "answer": "Ada", "supporting_facts": [["Biography", 0]]}),
    ]) + "\n", encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join([
        json.dumps({"id": "p1", "title": "France", "text": "Paris is in France."}),
        json.dumps({"id": "p2", "title": "Biography", "text": "Ada wrote notes."}),
        json.dumps({"id": "p3", "title": "Noise", "text": "unrelated"}),
    ]) + "\n", encoding="utf-8")
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "seed": 520,
        "splits": {"smoke": ["q1", "q2"], "tuning": [], "validation": [], "final": []},
    }), encoding="utf-8")
    config = ResearchConfig(
        method="bm25_rag", dataset="hotpotqa", dataset_path=str(questions),
        global_corpus_path=str(corpus), setting="global", split="smoke",
        split_manifest_path=str(manifest), output_root=str(tmp_path / "outputs"),
        retriever="bm25", top_k=2, max_llm_calls=2, max_total_tokens=3000,
        final_reserve_tokens=100,
    )
    run_dir = run(config, mock=DeterministicMockLLM())
    predictions = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(predictions) == 2
    assert all(prediction["retrieved"] for prediction in predictions)
    retrieved_ids = {hit["passage_id"] for prediction in predictions for hit in prediction["retrieved"]}
    assert retrieved_ids <= {"p1", "p2", "p3"}
    metrics = [json.loads(line) for line in (run_dir / "per_example_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["support_recall"] >= 0 for row in metrics)
