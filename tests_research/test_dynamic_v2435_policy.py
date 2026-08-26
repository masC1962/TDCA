from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from tdca_research.budget import Budget
from tdca_research.dynamic.graph import GraphOperation, OperationType
from tdca_research.dynamic_v2.allocator import AdaptiveComputationAllocator, EVCSignals
from tdca_research.dynamic_v2.engine import _bind_transition_to_execution
from tdca_research.dynamic_v2.termination import TerminalBeliefReadout
from tdca_research.dynamic_v2.transitions import certified_transition_value
from tdca_research.dynamic_v2.recovery import (
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
        "certified_terminal_materialization": True,
        "bind_transition_certificate_to_execution": True,
        "terminal_certificate_accepts_ancestor_claims": True,
        "feedback_conditioned_delayed_value": True,
        "compact_objective_recovery_query": True,
    }
    values.update(overrides)
    return _transition_config(**values)


def test_v2435_transition_certificate_is_rebound_after_fidelity_truncation():
    cfg = _config()
    _, graph = _scored_claim_graph(cfg, two=True)
    proposed = _operation(
        80, OperationType.BRANCH,
        payload={"mode": "assignments", "candidate_ids": ["c1", "c2"]},
        sources=["c1", "c2"],
    )
    packet = AdaptiveComputationAllocator(cfg).allocate(
        graph, [proposed], Budget(16, 16000, 0, Usage()),
    )[0]
    invalid_concrete = GraphOperation(
        proposed.operation_id, proposed.operation_type, proposed.target_id,
        ["c1"], proposed.branch_id,
        {"mode": "assignments", "candidate_ids": ["c1"]},
        proposed.reason, proposed.proposed_by, proposed.estimated_cost,
    )
    rebound = _bind_transition_to_execution(
        graph, packet, invalid_concrete, cfg,
    )
    assert packet.transition_certificate["mandatory"] is True
    assert rebound.operation.source_ids == ["c1"]
    assert rebound.transition_certificate["mandatory"] is False
    assert rebound.predicted_transition_value == 0.0


def test_v2435_terminal_certificate_accepts_only_sealed_branch_ancestors():
    cfg = _config()
    controller, graph = _scored_claim_graph(cfg, two=True)
    split = _operation(
        81, OperationType.BRANCH,
        payload={"mode": "assignments", "candidate_ids": ["c1", "c2"]},
        sources=["c1", "c2"],
    )
    graph = controller.apply(graph, split)
    child = next(
        row for row in graph.active_branches()
        if row.assignments.get("s_root") == "c1"
    )
    terminal = GraphOperation(
        "op_82", OperationType.COMMIT, "s_root", ["c1"], child.branch_id,
        {"mode": "answer", "answer": {
            "node_id": "answer_82",
            "candidate_answer": "Beta Country",
            "answer_type": "country",
            "supporting_claims": ["c1"],
            "supporting_evidence": ["e1"],
            "derivation_edge": "hyperedge_answer_82",
            "confidence": 0.9,
            "answer_type_consistency": 1.0,
            "contradiction_risk": 0.0,
            "inference_type": "test_terminal",
            "status": "accepted",
        }},
        "test", "offline_test", {"llm_calls": 0.0, "tokens": 0.0},
    )
    accepted, _ = TerminalBeliefReadout(cfg).evaluate(graph, [terminal])
    assert len(accepted) == 1
    assert certified_transition_value(graph, accepted[0], cfg)["mandatory"] is True

    frozen = replace(cfg, terminal_certificate_accepts_ancestor_claims=False)
    legacy_operation, _ = TerminalBeliefReadout(frozen).evaluate(graph, [terminal])
    assert len(legacy_operation) == 1
    assert certified_transition_value(
        graph, legacy_operation[0], frozen,
    )["mandatory"] is False


def test_v2435_delayed_value_is_multiplicative_closure_and_local_feedback():
    allocator = AdaptiveComputationAllocator(_config())
    signals = EVCSignals(
        operation_closure_probability=0.8,
        expected_obligation_delta=0.5,
        observed_value=0.75,
        failure_cooldown=0.2,
        operation_redundancy=0.1,
        dead_end_risk=0.25,
    )
    _, delayed, _ = allocator._horizon_scores(signals, "retrieve:default")
    assert delayed == 0.8 * 0.5 * 0.75 * 0.8 * 0.9 * 0.75


def test_v2435_compact_recovery_query_removes_meta_instruction_boilerplate():
    cfg = _config()
    _, graph = _scored_claim_graph(cfg)
    subgoal = graph.node("s_root")
    claim = graph.node("c1")
    verdict = proof_usable_target_claim(
        graph, claim, subgoal, cfg, projects_target=False,
    )
    diagnosis = diagnose_proof_gap(
        graph, subgoal, [claim], [], {claim.node_id: verdict}, [],
    )
    query = proof_gap_recovery_query(
        graph, subgoal, subgoal.instantiated_question, [], [claim], diagnosis,
        compact_objective=True,
    )
    assert query
    assert "Independent source" not in query
    assert "Gap:" not in query
    assert "Alpha City" in query or "Beta Country" in query


def test_v2435_frozen_configs_and_gold_free_delayed_audit():
    root = Path(__file__).resolve().parents[1]
    smoke = type(_config()).from_yaml(
        root / "configs/dynamic_hypergraph_v2435_qwen_smoke20.yaml"
    )
    shadow = type(smoke).from_yaml(
        root / "configs/dynamic_hypergraph_v2435_qwen_shadow20.yaml"
    )
    assert smoke.bind_transition_certificate_to_execution
    assert smoke.terminal_certificate_accepts_ancestor_claims
    assert smoke.feedback_conditioned_delayed_value
    assert smoke.compact_objective_recovery_query
    assert smoke.meta_stop_evc_threshold == 0.08
    assert smoke.campaign_id == shadow.campaign_id
    assert smoke.campaign_ledger_path == shadow.campaign_ledger_path
    assert smoke.campaign_provider_call_cap == 1189
    assert smoke.campaign_provider_token_cap == 1_085_849
    audit = json.loads((
        root / "configs/dynamic_v2435_offline_replay.json"
    ).read_text(encoding="utf-8"))
    assert audit["gold_used_for_policy_change"] is False
    candidate = audit["delayed_candidates"]["closure_mass_x_feedback"]
    assert candidate["overall_spearman"] >= 0.15
    assert candidate["choice_conditioned_spearman"] > 0.0
