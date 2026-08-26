from __future__ import annotations

import json
from pathlib import Path

from tdca_research.budget import Budget
from tdca_research.dynamic.graph import OperationType
from tdca_research.dynamic_v2.allocator import AdaptiveComputationAllocator
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.obligations import (
    operation_conditioned_closure_value,
    operation_obligation_targets,
)
from tdca_research.dynamic_v2.recovery import (
    claim_projects_target,
    diagnose_proof_gap,
    proof_gap_recovery_query,
    proof_usable_target_claim,
)
from tdca_research.models import Usage

from tests_research.test_dynamic_v2431_policy import _operation
from tests_research.test_dynamic_v2432_policy import (
    _scored_claim_graph,
    _transition_config,
)


def _config(**overrides):
    values = {
        "proof_usable_target_gate": True,
        "proof_quality_obligation_alignment": True,
        "anchored_proof_recovery_query": True,
        "terminal_min_absolute_support": 0.95,
    }
    values.update(overrides)
    return _transition_config(**values)


def _target_graph(cfg):
    controller, graph = _scored_claim_graph(cfg)
    graph = controller.apply(graph, _operation(
        89, OperationType.VERIFY, sources=["c1"], payload={"scores": {
            "c1": {
                "grounding": 0.9, "entailment": 0.9, "type_match": 1.0,
                "dependency_consistency": 1.0, "retrieval_support": 0.9,
                "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
                "absolute_support": 0.9, "relative_weight": 1.0,
                "set_entropy": 0.0, "evidence_gap": 0.1,
                "status": "scored", "answer_position": "value",
            },
        }},
    ))
    return controller, graph


def test_v2436_projects_unusable_target_claim_into_controller_obligation():
    cfg = _config()
    _, graph = _target_graph(cfg)
    claim = graph.node("c1")
    subgoal = graph.node("s_root")
    assert claim_projects_target(graph, claim)
    verdict = proof_usable_target_claim(
        graph, claim, subgoal, cfg, projects_target=True,
    )
    assert verdict.usable is False
    obligations = [
        row for row in graph.proof_obligations.values()
        if row.obligation_type == "insufficient_target_proof"
    ]
    assert len(obligations) == 1
    assert graph.proof_obligation_version == "proof-obligation-state-v2.4.3.6"
    assert obligations[0].status == "OPEN"
    assert obligations[0].required_node_ids == ["c1"]
    assert "absolute_support_below_proof_floor" in obligations[0].reason_codes
    restored = type(graph).from_dict(graph.to_dict())
    restored.validate()
    assert restored.proof_obligation_version == graph.proof_obligation_version
    assert restored.proof_obligations == graph.proof_obligations


def test_v2436_recovery_retrieval_targets_the_projected_proof_gap():
    cfg = _config()
    controller, graph = _target_graph(cfg)
    claim = graph.node("c1")
    subgoal = graph.node("s_root")
    verdict = proof_usable_target_claim(
        graph, claim, subgoal, cfg, projects_target=True,
    )
    diagnosis = diagnose_proof_gap(
        graph, subgoal, [claim], [claim], {claim.node_id: verdict}, [],
    )
    query = proof_gap_recovery_query(
        graph, subgoal, subgoal.instantiated_question, [], [claim], diagnosis,
        anchored_objective=True,
    )
    recovery = _operation(90, OperationType.RETRIEVE, payload={
        "query": query,
        **diagnosis.to_payload(),
    })
    targets = operation_obligation_targets(graph, recovery, strict=True)
    assert targets
    assert all(
        graph.proof_obligations[value].obligation_type
        == "insufficient_target_proof"
        for value in targets
    )
    estimate = operation_conditioned_closure_value(graph, recovery)
    assert estimate["target_obligation_ids"] == targets
    assert estimate["delayed_value"] > 0.0
    packet = AdaptiveComputationAllocator(cfg).allocate(
        graph, [recovery], Budget(16, 16000, 0, Usage()),
    )[0]
    graph = controller.apply(
        graph, AdaptiveComputationAllocator.attach(recovery, packet),
    )
    restored = type(graph).from_dict(graph.to_dict())
    restored.validate()
    assert restored.allocation_history[-1].target_obligation_ids == targets
    assert restored.allocation_history[-1].obligation_estimate
    assert restored.allocation_history[-1].transition_certificate == (
        graph.allocation_history[-1].transition_certificate
    )


def test_v2436_anchored_query_is_relevant_and_has_no_meta_boilerplate():
    cfg = _config()
    _, graph = _target_graph(cfg)
    claim = graph.node("c1")
    subgoal = graph.node("s_root")
    verdict = proof_usable_target_claim(
        graph, claim, subgoal, cfg, projects_target=True,
    )
    diagnosis = diagnose_proof_gap(
        graph, subgoal, [claim], [claim], {claim.node_id: verdict}, [],
    )
    query = proof_gap_recovery_query(
        graph, subgoal, subgoal.instantiated_question, [], [claim], diagnosis,
        anchored_objective=True,
    )
    assert query
    assert "Alpha City" in query or "Beta Country" in query
    assert "Independent source" not in query
    assert "Gap:" not in query


def test_v2436_alignment_is_opt_in_for_frozen_versions():
    cfg = _config(proof_quality_obligation_alignment=False)
    _, graph = _scored_claim_graph(cfg)
    assert not any(
        row.obligation_type == "insufficient_target_proof"
        for row in graph.proof_obligations.values()
    )


def test_v2436_frozen_config_reverts_failed_v2435_trials_and_keeps_gates():
    root = Path(__file__).resolve().parents[1]
    config = DynamicV2ResearchConfig.from_yaml(
        root / "configs/dynamic_hypergraph_v2436_qwen_smoke20.yaml"
    )
    config.validate()
    assert config.bind_transition_certificate_to_execution
    assert config.terminal_certificate_accepts_ancestor_claims
    assert not config.feedback_conditioned_delayed_value
    assert not config.compact_objective_recovery_query
    assert config.proof_quality_obligation_alignment
    assert config.anchored_proof_recovery_query
    assert config.campaign_provider_call_cap == 1066
    assert config.campaign_provider_token_cap == 934_432
    prereg = json.loads((
        root / "configs/dynamic_v2436_preregistration.json"
    ).read_text(encoding="utf-8"))
    gates = prereg["adaptive_smoke_a20_hard_gates"]
    assert gates["candidate_presence_rate_min"] == 0.75
    assert gates["legacy_execution_plan_completion_rate_min"] == 0.75
    assert gates["f1_min"] == 0.58
    assert prereg["training"] is False
