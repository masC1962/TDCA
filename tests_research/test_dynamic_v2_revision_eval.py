import hashlib
import json
from pathlib import Path

from tdca_research.dynamic.graph import CandidateStatus
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.revision_eval import (
    IndependentRevisionScorer,
    RevisionInput,
    apply_natural_revision_episode,
    load_unlabeled_inputs,
    predict_items,
    score_prediction_rows,
)
from tdca_research.llm import DeterministicMockLLM


def revision_config():
    return DynamicV2ResearchConfig(
        llm_backend="mock",
        max_total_tokens=4000,
        final_reserve_tokens=200,
        max_candidates_per_subgoal=4,
        max_graph_nodes=32,
        max_graph_operations=16,
    )


def item():
    return RevisionInput(
        item_id="public_1",
        case_id="case_1",
        page="Alpha",
        claim="Alpha was founded in 1990.",
        evidence="The company records state that Alpha was founded in 2001.",
        content_sha256="fixture",
    )


def assessment(relation, contradiction, support=0.05, insufficiency=0.05):
    model = DeterministicMockLLM(json_responses=[{
        "relation": relation,
        "grounding": 0.95,
        "support_entailment": support,
        "contradiction_entailment": contradiction,
        "insufficiency": insufficiency,
        "confidence": 0.92,
        "evidence_span": "Alpha was founded in 2001",
        "reason_codes": ["changed_date"],
    }])
    return IndependentRevisionScorer(model).assess(item())[0]


def test_public_natural_contradiction_triggers_controller_owned_revision():
    graph, decision = apply_natural_revision_episode(
        item(), assessment("refutes", 0.94), revision_config(),
    )
    assert decision["predicted_action"] == "should_revise"
    assert decision["invariant_valid"]
    assert len(graph.supersession_history) == 1
    assert graph.supersession_history[0].natural
    assert graph.nodes["prior_claim"].status == CandidateStatus.INVALID
    assert graph.controller_state_hash == graph.state_hash()


def test_supporting_evidence_does_not_trigger_destructive_revision():
    graph, decision = apply_natural_revision_episode(
        item(), assessment("supports", 0.02, support=0.96), revision_config(),
    )
    assert decision["predicted_action"] == "should_not_revise"
    assert not graph.supersession_history
    assert graph.nodes["prior_claim"].status == CandidateStatus.COMMITTED
    graph.validate()


def test_high_coverage_support_overrides_spurious_raw_refutation():
    support_item = RevisionInput(
        item_id="support_1", case_id="case_support", page="Las Vegas Motor Speedway",
        claim="The Las Vegas Motor Speedway holds the playoff race for the truck series.",
        evidence="The venue hosts the second playoff race for the truck series.",
        content_sha256="fixture",
    )
    graph, decision = apply_natural_revision_episode(
        support_item, assessment("refutes", 0.88, support=0.15), revision_config(),
    )
    assert decision["direct_support_override"]
    assert decision["direct_support_audit"]["coverage"] >= 0.80
    assert decision["predicted_action"] == "should_not_revise"
    assert not graph.supersession_history


def test_changed_numeric_value_is_not_hidden_by_lexical_support_guard():
    graph, decision = apply_natural_revision_episode(
        item(), assessment("refutes", 0.94), revision_config(),
    )
    assert not decision["direct_support_override"]
    assert decision["direct_support_audit"]["missing_claim_numbers"] == ["1990"]
    assert decision["predicted_action"] == "should_revise"


def test_changed_directional_relation_is_not_hidden_by_lexical_support_guard():
    directional_item = RevisionInput(
        item_id="direction_1", case_id="case_direction", page="Alpha",
        claim="Alpha closed before launch.",
        evidence="Alpha closed at launch.", content_sha256="fixture",
    )
    graph, decision = apply_natural_revision_episode(
        directional_item, assessment("refutes", 0.94), revision_config(),
    )
    assert not decision["direct_support_override"]
    assert decision["direct_support_audit"]["missing_directional_terms"] == ["before"]
    assert decision["predicted_action"] == "should_revise"
    assert graph.supersession_history


def test_prediction_loader_has_no_gold_label_channel(tmp_path: Path):
    row = {
        "item_id": "x", "case_id": "c", "page": "P",
        "claim": "A is B.", "evidence": "A is C.",
    }
    row["content_sha256"] = hashlib.sha256(json.dumps(
        {"claim": row["claim"], "evidence": row["evidence"]},
        ensure_ascii=False, sort_keys=True,
    ).encode()).hexdigest()
    cache = tmp_path / "unlabeled.jsonl"
    cache.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "provenance": {
            "unlabeled_cache": str(cache),
            "unlabeled_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        },
        "splits": {"development": [{
            "item_id": "x", "case_id": "c", "page": "P",
            "content_sha256": row["content_sha256"],
        }]},
    }
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, _ = load_unlabeled_inputs(inputs, "development")
    assert loaded[0].claim == "A is B."
    assert "label" not in loaded[0].__dict__


def test_revision_metrics_keep_precision_recall_and_fpr_separate():
    predictions = [
        {"item_id": "p", "decision": {"predicted_action": "should_revise", "invariant_valid": True},
         "usage": {"provider_calls": 1, "provider_reported_tokens": 10}},
        {"item_id": "n", "decision": {"predicted_action": "should_not_revise", "invariant_valid": True},
         "usage": {"provider_calls": 1, "provider_reported_tokens": 10}},
    ]
    labels = {"splits": {"evaluation": [
        {"item_id": "p", "expected_action": "should_revise", "gold_relation": "REFUTES", "category": "numeric"},
        {"item_id": "n", "expected_action": "should_not_revise", "gold_relation": "SUPPORTS", "category": "support"},
    ]}}
    metrics, rows = score_prediction_rows(predictions, labels, "evaluation")
    assert metrics["precision"] == metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["passes_natural_revision_gate"]
    assert len(rows) == 2


def test_cached_revision_prediction_does_not_count_provider_usage():
    class CachedMock(DeterministicMockLLM):
        def _usage(self, messages, text):
            generation = super()._usage(messages, text)
            generation.cached = True
            generation.metadata["provider_attempts"] = 1
            return generation

    model = CachedMock(json_responses=[{
        "relation": "supports", "grounding": 0.95,
        "support_entailment": 0.96, "contradiction_entailment": 0.01,
        "insufficiency": 0.03, "confidence": 0.92,
        "evidence_span": "Alpha was founded in 1990",
        "reason_codes": ["same_fact"],
    }])
    row = predict_items([item()], model, revision_config())[0]
    assert row["usage"]["provider_calls"] == 0
    assert row["usage"]["provider_reported_tokens"] == 0
    assert row["usage"]["prompt_tokens"] > 0
