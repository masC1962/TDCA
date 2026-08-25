from __future__ import annotations

from tdca_research.budget import Budget
from tdca_research.dynamic.graph import (
    BranchState,
    BranchStatus,
    GraphLimits,
    GraphOperation,
    OperationType,
)
from tdca_research.dynamic_v2.allocator import (
    AdaptiveComputationAllocator,
    feedback_key,
    operation_coarse_region_key,
    operation_family,
)
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.controller import V2GraphController
from tdca_research.dynamic_v2.graph import (
    DynamicReasoningHypergraphV2,
    OperationFeedbackStats,
    TerminationKind,
)
from tdca_research.dynamic_v2.termination import MetaStopPolicy
from tdca_research.models import Usage
from pathlib import Path
import json

from scripts.analyze_dynamic_v243_offline import absolute_cost, resource_fraction


def _config(**overrides):
    return DynamicV2ResearchConfig(
        llm_backend="mock",
        max_llm_calls=16,
        max_total_tokens=16000,
        final_reserve_tokens=0,
        max_retrieval_calls=8,
        max_graph_operations=64,
        max_graph_nodes=128,
        horizon_aware_evc=True,
        delayed_credit_assignment=True,
        multi_resource_evc=True,
        choice_conditioned_evc=True,
        absolute_resource_cost=True,
        proof_obligation_tracking=True,
        graph_local_delayed_value=True,
        certified_meta_stop=True,
        **overrides,
    )


def _graph(cfg):
    graph = DynamicReasoningHypergraphV2(
        "Which country contains Alpha City?",
        GraphLimits(
            cfg.max_candidates_per_subgoal,
            cfg.max_active_branches,
            cfg.max_graph_nodes,
            cfg.max_hyperedges,
            cfg.max_graph_revisions,
            cfg.max_revision_per_candidate,
            cfg.max_graph_depth,
            cfg.max_graph_operations,
            cfg.max_retrieval_calls,
        ),
    )
    graph.branches["branch_root"] = BranchState(
        "branch_root", None, {}, [], 1.0, BranchStatus.ACTIVE, 0,
    )
    graph.seal_controller_state()
    controller = V2GraphController(cfg)
    plan = GraphOperation(
        "op_plan", OperationType.EXPAND, "subgoal_root", [], "branch_root",
        {"subgoals": [{
            "node_id": "s_root",
            "question_template": "Which country contains Alpha City?",
            "instantiated_question": "Which country contains Alpha City?",
            "dependencies": [],
            "variable_bindings": {},
            "answer_type": "country",
            "terminal": True,
        }]},
        "test", "offline_test", {"llm_calls": 0.0, "tokens": 0.0},
    )
    return controller, controller.apply(graph, plan)


def _operation(number, kind, *, target="s_root", payload=None):
    return GraphOperation(
        f"op_{number}", kind, target, [], "branch_root", payload or {},
        "test", "offline_test", {"llm_calls": 0.0, "tokens": 0.0},
    )


def _packet_by_operation(packets, operation_id):
    return max(
        (row for row in packets if row.operation.operation_id == operation_id),
        key=lambda row: row.fidelity_fraction,
    )


def test_v243_absolute_cost_is_ready_set_invariant_and_bounded():
    cfg = _config()
    _, graph = _graph(cfg)
    allocator = AdaptiveComputationAllocator(cfg)
    budget = Budget(16, 16000, 0, Usage())
    target = _operation(1, OperationType.BRANCH, payload={"mode": "extract_typed"})
    first = _packet_by_operation(allocator.allocate(graph, [target], budget), target.operation_id)
    dominated = _operation(2, OperationType.PRUNE)
    second = _packet_by_operation(
        allocator.allocate(graph, [target, dominated], budget), target.operation_id,
    )
    assert first.predicted_normalized_cost == second.predicted_normalized_cost
    assert 0.0 < first.predicted_normalized_cost < 0.25
    assert first.normalized.expected_call_cost == second.normalized.expected_call_cost
    assert first.normalized.expected_token_cost == second.normalized.expected_token_cost


def test_v243_absolute_cost_increases_monotonically_with_scarcity():
    cfg = _config()
    _, graph = _graph(cfg)
    operation = _operation(3, OperationType.VERIFY)
    fresh = Budget(16, 16000, 0, Usage())
    scarce_usage = Usage(llm_calls=12, prompt_tokens=12000)
    scarce = Budget(16, 16000, 0, scarce_usage)
    first = _packet_by_operation(
        AdaptiveComputationAllocator(cfg).allocate(graph, [operation], fresh),
        operation.operation_id,
    )
    second = _packet_by_operation(
        AdaptiveComputationAllocator(cfg).allocate(graph, [operation], scarce),
        operation.operation_id,
    )
    assert second.predicted_normalized_cost > first.predicted_normalized_cost


def test_v243_controller_projects_obligations_and_operation_declares_closure():
    cfg = _config()
    controller, graph = _graph(cfg)
    open_rows = [row for row in graph.proof_obligations.values() if row.status == "OPEN"]
    assert any(row.obligation_type == "missing_evidence" for row in open_rows)
    retrieve = _operation(4, OperationType.RETRIEVE, payload={"query": "Alpha City country"})
    packet = AdaptiveComputationAllocator(cfg).allocate(
        graph, [retrieve], Budget(16, 16000, 0, Usage()),
    )[0]
    assert packet.target_obligation_ids
    assert all(value in graph.proof_obligations for value in packet.target_obligation_ids)
    attached = AdaptiveComputationAllocator.attach(retrieve, packet)
    graph = controller.apply(graph, attached)
    assert graph.proof_obligation_history
    assert graph.controller_state_hash
    graph.validate()
    restored = DynamicReasoningHypergraphV2.from_dict(graph.to_dict())
    assert restored.proof_obligations.keys() == graph.proof_obligations.keys()


def test_v243_graph_local_delayed_value_has_no_family_capacity_fallback():
    cfg = _config()
    _, graph = _graph(cfg)
    retrieve = _operation(5, OperationType.RETRIEVE, payload={"query": "Alpha City country"})
    prune = _operation(6, OperationType.PRUNE)
    packets = AdaptiveComputationAllocator(cfg).allocate(
        graph, [retrieve, prune], Budget(16, 16000, 0, Usage()),
    )
    retrieve_packet = _packet_by_operation(packets, retrieve.operation_id)
    prune_packet = _packet_by_operation(packets, prune.operation_id)
    assert retrieve_packet.predicted_delayed_proof_return > 0.0
    assert retrieve_packet.predicted_delayed_proof_return > prune_packet.predicted_delayed_proof_return
    assert retrieve_packet.raw.obligation_closure > 0.0


def test_v243_coarse_family_feedback_is_diagnostic_only():
    cfg = _config(hierarchical_within_question_feedback=True)
    _, graph = _graph(cfg)
    operation = _operation(51, OperationType.RETRIEVE, payload={"query": "Alpha City"})
    budget = Budget(16, 16000, 0, Usage())
    baseline = _packet_by_operation(
        AdaptiveComputationAllocator(cfg).allocate(graph, [operation], budget),
        operation.operation_id,
    )
    coarse_key = feedback_key(
        operation_family(operation), operation_coarse_region_key(operation),
    )
    graph.operation_feedback[coarse_key] = OperationFeedbackStats(
        observations=20, successes=0, no_ops=20, cumulative_utility=0.0,
        cumulative_cost=20.0, posterior_value=0.01, posterior_success=0.01,
        consecutive_failures=20, cooldown_until_step=100,
    )
    graph.seal_controller_state()
    changed = _packet_by_operation(
        AdaptiveComputationAllocator(cfg).allocate(graph, [operation], budget),
        operation.operation_id,
    )
    assert changed.feedback_prior["coarse_observations"] > 0.0
    assert changed.predicted_evc == baseline.predicted_evc
    assert changed.requested_budget == baseline.requested_budget


def test_v243_no_executable_abstain_has_complete_dead_end_certificate():
    cfg = _config()
    _, graph = _graph(cfg)
    budget = Budget(16, 16000, 0, Usage())
    decision = MetaStopPolicy(cfg).decide(graph, [], budget)
    assert decision.outcome == TerminationKind.ABSTAIN
    assert decision.reason == "no_executable_computation_with_certificate"
    certificate = decision.dead_end_certificate
    assert certificate["certificate_version"] == "proof-obligation-dead-end-v2.4.3"
    assert certificate["open_obligations"]
    assert certificate["remaining_budget"]["llm_calls"] == 16
    assert certificate["candidate_operations"] == []


def test_v243_config_rejects_non_normalized_absolute_cost_weights():
    try:
        _config(absolute_cost_weight_call=0.40)
    except ValueError as exc:
        assert "absolute resource cost weights" in str(exc)
    else:
        raise AssertionError("invalid absolute cost weights were accepted")


def test_v243_frozen_configs_validate_and_share_only_the_frozen_policy():
    root = Path(__file__).resolve().parents[1]
    smoke = DynamicV2ResearchConfig.from_yaml(
        root / "configs/dynamic_hypergraph_v243_qwen_smoke20.yaml"
    )
    shadow = DynamicV2ResearchConfig.from_yaml(
        root / "configs/dynamic_hypergraph_v243_qwen_shadow20.yaml"
    )
    smoke.validate()
    shadow.validate()
    assert smoke.absolute_resource_cost
    assert smoke.proof_obligation_tracking
    assert smoke.graph_local_delayed_value
    assert smoke.certified_meta_stop
    assert smoke.campaign_provider_call_cap == 1871
    assert smoke.campaign_provider_token_cap == 1858349
    assert smoke.split_manifest_path != shadow.split_manifest_path
    assert smoke.merged(split_manifest_path=shadow.split_manifest_path) == shadow
    prereg = json.loads(
        (root / "configs/dynamic_v243_preregistration.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (root / "configs/dynamic_v243_offline_replay.json").read_text(encoding="utf-8")
    )
    assert prereg["campaign"]["provider_attempt_cap"] == smoke.campaign_provider_call_cap
    assert prereg["campaign"]["provider_reported_token_cap"] == smoke.campaign_provider_token_cap
    assert replay["gold_used"] is False
    assert replay["known_cases_have_positive_cost_only_counterfactual"] is True


def test_v243_offline_cost_equation_matches_frozen_weights():
    packet = {
        "operation_family": "branch:extract_typed",
        "requested_budget": {"max_tokens": 900},
        "remaining_global_budget": {
            "llm_calls": 16, "tokens": 16000, "retrieval_calls": 8,
        },
        "evc_components_raw": {"graph_growth_risk": 0.8},
    }
    value, components = absolute_cost(packet)
    expected = (
        0.35 * resource_fraction(1, 16, 16)
        + 0.35 * resource_fraction(900, 16000, 16000)
        + 0.10 * 0.8
    )
    assert value == expected
    assert components["retrieval"] == 0.0
    assert value < 0.25
