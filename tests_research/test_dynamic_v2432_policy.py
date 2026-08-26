from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from tdca_research.budget import Budget
from tdca_research.dynamic.graph import GraphOperation, OperationType
from tdca_research.dynamic_v2.allocator import AdaptiveComputationAllocator
from tdca_research.dynamic_v2.termination import MetaStopPolicy
from tdca_research.dynamic_v2.transitions import certified_transition_value
from tdca_research.models import Usage

from tests_research.test_dynamic_v2431_policy import _config, _graph, _operation


def _transition_config(**overrides):
    return _config(certified_transition_option_value=True, **overrides)


def _scored_claim_graph(cfg, *, two: bool = False):
    controller, graph = _graph(cfg)
    graph = controller.apply(graph, _operation(
        19, OperationType.RETRIEVE,
        payload={
            "query": "Alpha City country", "allocated_top_k": 2,
            "hit_count": 1,
            "evidence": [{
                "node_id": "e1", "document_id": "p1", "passage_id": "p1",
                "title": "Alpha", "source_span": "Alpha City is in Beta Country.",
                "retrieval_rank": 1, "retrieval_score": 1.0,
                "retrieval_query": "Alpha City country",
                "retriever_identity": "fixture",
            }],
        },
    ))
    candidates = [{
        "node_id": "c1", "subject": "Alpha City", "relation": "country",
        "value": "Beta Country", "answer_type": "country",
        "evidence_refs": ["e1"], "dependency_claim_ids": [],
        "extraction_confidence": 0.9,
    }]
    if two:
        candidates.append({
            "node_id": "c2", "subject": "Alpha City", "relation": "country",
            "value": "Gamma Country", "answer_type": "country",
            "evidence_refs": ["e1"], "dependency_claim_ids": [],
            "extraction_confidence": 0.85,
        })
    graph = controller.apply(graph, _operation(
        20, OperationType.BRANCH,
        payload={"mode": "candidates", "candidates": candidates},
    ))
    scores = {
        row["node_id"]: {
            "grounding": 0.9, "entailment": 0.9, "type_match": 1.0,
            "dependency_consistency": 1.0, "retrieval_support": 0.9,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
            "absolute_support": 0.9,
            "relative_weight": 0.8 if row["node_id"] == "c1" else 0.2,
            "set_entropy": 0.2, "evidence_gap": 0.1, "status": "scored",
        }
        for row in candidates
    }
    graph = controller.apply(graph, _operation(
        21, OperationType.VERIFY,
        payload={"scores": scores}, sources=list(scores),
    ))
    return controller, graph


def test_v2432_certified_commit_bypasses_only_the_generic_net_threshold():
    cfg = _transition_config()
    _, graph = _scored_claim_graph(cfg)
    commit = _operation(
        22, OperationType.COMMIT,
        payload={"candidate_id": "c1"}, sources=["c1"],
    )
    transition = certified_transition_value(graph, commit)
    assert transition["mandatory"] is True
    assert transition["deterministic"] is True
    assert transition["provider_calls"] == 0
    assert transition["predicted_transition_value"] > 0.0
    packets = AdaptiveComputationAllocator(cfg).allocate(
        graph, [commit], Budget(16, 16000, 0, Usage()),
    )
    packets = [replace(packets[0], predicted_evc=0.0)]
    decision = MetaStopPolicy(cfg).decide(
        graph, packets, Budget(16, 16000, 0, Usage()),
    )
    assert decision.outcome.value == "CONTINUE"
    assert decision.reason == "certified_state_transition"
    assert decision.selected_allocation_id == packets[0].allocation_id


def test_v2432_invalid_commit_cannot_forge_a_transition_certificate():
    cfg = _transition_config()
    _, graph = _scored_claim_graph(cfg)
    invalid = _operation(
        23, OperationType.COMMIT,
        payload={
            "candidate_id": "missing",
            "transition_certificate": {"mandatory": True},
        },
        sources=["missing"],
    )
    assert certified_transition_value(graph, invalid)["mandatory"] is False
    packets = AdaptiveComputationAllocator(cfg).allocate(
        graph, [invalid], Budget(16, 16000, 0, Usage()),
    )
    packets = [replace(
        packets[0], predicted_evc=0.0, predicted_gross_opportunity=0.0,
    )]
    decision = MetaStopPolicy(cfg).decide(
        graph, packets, Budget(16, 16000, 0, Usage()),
    )
    assert decision.outcome.value == "ABSTAIN"
    assert decision.reason == "affordable_proof_opportunity_below_net_value_threshold"


def test_v2432_commit_transition_is_realized_and_roundtrips():
    cfg = _transition_config()
    controller, graph = _scored_claim_graph(cfg)
    commit = _operation(
        24, OperationType.COMMIT,
        payload={"candidate_id": "c1"}, sources=["c1"],
    )
    budget = Budget(16, 16000, 0, Usage())
    packet = AdaptiveComputationAllocator(cfg).allocate(graph, [commit], budget)[0]
    graph = controller.apply(
        graph, AdaptiveComputationAllocator.attach(commit, packet),
    )
    graph = controller.reconcile_allocation(
        graph, packet, {"graph_operations": 1.0}, True,
    )
    allocation = graph.allocation_history[-1]
    assert graph.proof_obligation_version == "proof-obligation-state-v2.4.3.2"
    assert allocation.transition_realized is True
    assert allocation.actual_transition_value == allocation.predicted_transition_value
    assert allocation.transition_observations["assignment_materialized"] is True
    restored = type(graph).from_dict(graph.to_dict())
    assert restored.controller_state_hash == graph.controller_state_hash
    assert restored.allocation_history[-1].transition_realized is True


def test_v2432_assignment_branch_is_certified_but_stale_branch_is_not():
    cfg = _transition_config()
    controller, graph = _scored_claim_graph(cfg, two=True)
    branch = _operation(
        25, OperationType.BRANCH,
        payload={"mode": "assignments", "candidate_ids": ["c1", "c2"]},
        sources=["c1", "c2"],
    )
    assert certified_transition_value(graph, branch)["mandatory"] is True
    packet = AdaptiveComputationAllocator(cfg).allocate(
        graph, [branch], Budget(16, 16000, 0, Usage()),
    )[0]
    graph = controller.apply(
        graph, AdaptiveComputationAllocator.attach(branch, packet),
    )
    graph = controller.reconcile_allocation(
        graph, packet, {"graph_operations": 1.0}, True,
    )
    assert graph.allocation_history[-1].transition_realized is True
    assert certified_transition_value(graph, branch)["mandatory"] is False


def test_v2432_frozen_configs_and_counterfactual_replay():
    root = Path(__file__).resolve().parents[1]
    smoke = type(_transition_config()).from_yaml(
        root / "configs/dynamic_hypergraph_v2432_qwen_smoke20.yaml"
    )
    shadow = type(smoke).from_yaml(
        root / "configs/dynamic_hypergraph_v2432_qwen_shadow20.yaml"
    )
    assert smoke.certified_transition_option_value
    assert smoke.meta_stop_evc_threshold == 0.08
    assert smoke.campaign_ledger_path == shadow.campaign_ledger_path
    assert smoke.campaign_provider_call_cap == 1608
    assert smoke.campaign_provider_token_cap == 1_583_349
    replay = json.loads((
        root / "analysis_outputs/dynamic_v2432_campaign/offline_counterfactual.json"
    ).read_text(encoding="utf-8"))
    assert replay["gold_used"] is False
    assert replay["certified_transition_unlocked_count"] == 15
    assert replay["all_unlocked_are_zero_provider"] is True
