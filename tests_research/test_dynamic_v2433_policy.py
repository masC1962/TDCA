from __future__ import annotations

import pytest
import json
from pathlib import Path

from tdca_research.dynamic.graph import OperationType
from tdca_research.dynamic_v2.allocator import AdaptiveComputationAllocator, EVCSignals

from tests_research.test_dynamic_v2431_policy import _graph, _operation
from tests_research.test_dynamic_v2432_policy import _transition_config


def _config(**overrides):
    values = {
        "preserve_subgoal_question_on_retrieval": True,
        "one_step_progress_immediate_value": True,
    }
    values.update(overrides)
    return _transition_config(**values)


def _retrieval(query: str):
    return _operation(40, OperationType.RETRIEVE, payload={
        "query": query,
        "allocated_top_k": 1,
        "hit_count": 1,
        "evidence": [{
            "node_id": "e40", "document_id": "p40", "passage_id": "p40",
            "title": "Alpha", "source_span": "Alpha City is in Beta Country.",
            "retrieval_rank": 1, "retrieval_score": 1.0,
            "retrieval_query": query, "retriever_identity": "fixture",
        }],
    })


def test_v2433_retrieval_query_does_not_overwrite_reasoning_objective():
    cfg = _config()
    controller, graph = _graph(cfg)
    original = graph.nodes["s_root"].instantiated_question
    recovery_query = "Independent source for a missing country relation."
    graph = controller.apply(graph, _retrieval(recovery_query))
    assert graph.nodes["s_root"].instantiated_question == original
    assert graph.retrieval_attempt_history[-1].query == recovery_query
    assert graph.evidence("s_root", "branch_root")[-1].retrieval_query == recovery_query


def test_v2433_frozen_legacy_retrieval_semantics_remain_available():
    cfg = _config(preserve_subgoal_question_on_retrieval=False)
    controller, graph = _graph(cfg)
    recovery_query = "Independent source for a missing country relation."
    graph = controller.apply(graph, _retrieval(recovery_query))
    assert graph.nodes["s_root"].instantiated_question == recovery_query


def test_v2433_immediate_value_is_additive_normalized_one_step_progress():
    allocator = AdaptiveComputationAllocator(_config())
    signals = EVCSignals(
        evidence_novelty=0.2,
        obligation_closure=0.4,
        operation_closure_probability=0.5,
        expected_obligation_delta=0.6,
        terminal_gap=0.8,
        answer_impact=0.7,
    )
    immediate, _, _ = allocator._horizon_scores(signals, "retrieve:default")
    assert immediate == pytest.approx((0.2 + 0.4 + 0.3 + 0.8 + 0.7) / 5.0)


def test_v2433_frozen_configs_share_only_the_remaining_campaign_budget():
    root = Path(__file__).resolve().parents[1]
    smoke = type(_config()).from_yaml(
        root / "configs/dynamic_hypergraph_v2433_qwen_smoke20.yaml"
    )
    shadow = type(smoke).from_yaml(
        root / "configs/dynamic_hypergraph_v2433_qwen_shadow20.yaml"
    )
    assert smoke.preserve_subgoal_question_on_retrieval
    assert smoke.one_step_progress_immediate_value
    assert smoke.meta_stop_evc_threshold == 0.08
    assert smoke.campaign_id == shadow.campaign_id
    assert smoke.campaign_ledger_path == shadow.campaign_ledger_path
    assert smoke.campaign_provider_call_cap == 1483
    assert smoke.campaign_provider_token_cap == 1_432_292
    replay = json.loads((
        root / "configs/dynamic_v2433_offline_replay.json"
    ).read_text(encoding="utf-8"))
    assert replay["gold_used_for_policy_change"] is False
    assert replay["normalized_one_step_progress_pearson"] > 0.1
    assert replay["decision"] == "GO_SOURCE_FREEZE_AND_DIAGNOSTIC_SMOKE"
