from __future__ import annotations

from tdca_research.budget import Budget
from tdca_research.dynamic.graph import (
    BranchState,
    BranchStatus,
    GraphLimits,
    GraphOperation,
    OperationType,
)
from tdca_research.dynamic_v2.allocator import AdaptiveComputationAllocator
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.controller import V2GraphController
from tdca_research.dynamic_v2.graph import DynamicReasoningHypergraphV2
from tdca_research.dynamic_v2.obligations import (
    operation_conditioned_closure_value,
    operation_obligation_targets,
)
from tdca_research.models import Usage
from pathlib import Path
import json


def _config(**overrides):
    values = dict(
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
        operation_conditioned_obligation_closure=True,
        exact_fidelity_resource_accounting=True,
        marginal_fidelity_evc_gate=True,
        critical_obligation_budget_reserve=True,
    )
    values.update(overrides)
    return DynamicV2ResearchConfig(**values)


def _operation(number, kind, *, payload=None, sources=None):
    return GraphOperation(
        f"op_{number}", kind, "s_root", sources or [], "branch_root", payload or {},
        "test", "offline_test", {"llm_calls": 0.0, "tokens": 0.0},
    )


def _graph(cfg):
    graph = DynamicReasoningHypergraphV2(
        "Which country contains Alpha City?",
        GraphLimits(
            cfg.max_candidates_per_subgoal, cfg.max_active_branches,
            cfg.max_graph_nodes, cfg.max_hyperedges, cfg.max_graph_revisions,
            cfg.max_revision_per_candidate, cfg.max_graph_depth,
            cfg.max_graph_operations, cfg.max_retrieval_calls,
        ),
    )
    graph.branches["branch_root"] = BranchState(
        "branch_root", None, {}, [], 1.0, BranchStatus.ACTIVE, 0,
    )
    graph.seal_controller_state()
    controller = V2GraphController(cfg)
    graph = controller.apply(graph, GraphOperation(
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
    ))
    return controller, graph


def test_v2431_importance_is_separate_from_operation_conditioned_closure():
    cfg = _config()
    _, graph = _graph(cfg)
    retrieve = _operation(1, OperationType.RETRIEVE, payload={
        "query": "Alpha City country",
    })
    estimate = operation_conditioned_closure_value(graph, retrieve)
    assert estimate["obligation_importance"] == 1.0
    assert 0.0 < estimate["operation_closure_probability"] <= 1.0
    assert estimate["target_obligation_ids"]
    commit = _operation(2, OperationType.COMMIT, payload={"candidate_id": "missing"})
    assert operation_obligation_targets(graph, commit, strict=True) == []
    assert operation_conditioned_closure_value(graph, commit)["delayed_value"] == 0.0


def test_v2431_extraction_requires_the_obligation_evidence_sources():
    cfg = _config()
    controller, graph = _graph(cfg)
    graph = controller.apply(graph, _operation(3, OperationType.RETRIEVE, payload={
        "query": "Alpha City country",
        "evidence": [{
            "node_id": "e1", "document_id": "p1", "passage_id": "p1",
            "title": "Alpha", "source_span": "Alpha City is in Beta Country.",
            "retrieval_rank": 1, "retrieval_score": 1.0,
            "retrieval_query": "Alpha City country", "retriever_identity": "fixture",
        }],
    }))
    missing_source = _operation(4, OperationType.BRANCH, payload={
        "mode": "extract_typed", "question": graph.question,
        "dependency_claim_ids": [],
    })
    assert operation_obligation_targets(graph, missing_source, strict=True) == []
    grounded = _operation(
        5, OperationType.BRANCH, sources=["e1"], payload={
            "mode": "extract_typed", "question": graph.question,
            "dependency_claim_ids": [],
        },
    )
    targets = operation_obligation_targets(graph, grounded, strict=True)
    assert targets
    assert all(
        graph.proof_obligations[value].obligation_type == "missing_claim"
        for value in targets
    )


def test_v2431_verification_cost_counts_requested_samples_exactly():
    cfg = _config(marginal_fidelity_evc_gate=False)
    _, graph = _graph(cfg)
    verify = _operation(6, OperationType.VERIFY, payload={"question": graph.question})
    packets = AdaptiveComputationAllocator(cfg).allocate(
        graph, [verify], Budget(16, 16000, 0, Usage()),
    )
    by_level = {row.fidelity_level: row for row in packets}
    assert by_level["low"].requested_budget["llm_calls"] == 1
    assert by_level["medium"].requested_budget["llm_calls"] == 1
    assert by_level["high"].requested_budget["llm_calls"] == 2
    assert by_level["high"].predicted_provider_calls == 2
    assert by_level["high"].normalized.expected_call_cost == 2 / 16
    assert by_level["medium"].normalized.expected_call_cost == 1 / 16
    assert by_level["high"].predicted_normalized_cost > by_level["medium"].predicted_normalized_cost


def test_v2431_high_fidelity_is_never_emitted_with_nonpositive_marginal_evc():
    cfg = _config()
    _, graph = _graph(cfg)
    verify = _operation(7, OperationType.VERIFY, payload={"question": graph.question})
    packets = AdaptiveComputationAllocator(cfg).allocate(
        graph, [verify], Budget(16, 16000, 0, Usage()),
    )
    high = [row for row in packets if row.fidelity_level == "high"]
    assert all(row.predicted_marginal_evc > 0.0 for row in high)
    assert all(row.reserve_feasible for row in high)


def test_v2431_controller_records_actual_target_closure_and_roundtrips():
    cfg = _config()
    controller, graph = _graph(cfg)
    placeholder = _operation(8, OperationType.RETRIEVE, payload={
        "query": "Alpha City country",
    })
    packet = AdaptiveComputationAllocator(cfg).allocate(
        graph, [placeholder], Budget(16, 16000, 0, Usage()),
    )[0]
    actual = _operation(8, OperationType.RETRIEVE, payload={
        "query": "Alpha City country",
        "allocated_top_k": 5,
        "hit_count": 1,
        "evidence": [{
            "node_id": "e8", "document_id": "p8", "passage_id": "p8",
            "title": "Alpha", "source_span": "Alpha City is in Beta Country.",
            "retrieval_rank": 1, "retrieval_score": 1.0,
            "retrieval_query": "Alpha City country", "retriever_identity": "fixture",
        }],
    })
    graph = controller.apply(graph, AdaptiveComputationAllocator.attach(actual, packet))
    graph = controller.reconcile_allocation(
        graph, packet, {"retrieval_calls": 1.0}, True,
    )
    allocation = graph.allocation_history[-1]
    assert allocation.predicted_provider_calls == 0
    assert allocation.actual_closed_target_ids
    assert allocation.actual_target_closure_rate == 1.0
    assert graph.proof_obligation_version == "proof-obligation-state-v2.4.3.1"
    restored = DynamicReasoningHypergraphV2.from_dict(graph.to_dict())
    assert restored.controller_state_hash == graph.controller_state_hash
    assert restored.allocation_history[-1].actual_closed_target_ids == (
        allocation.actual_closed_target_ids
    )


def test_v2431_frozen_configs_enable_only_the_preregistered_policy_delta():
    root = Path(__file__).resolve().parents[1]
    smoke = DynamicV2ResearchConfig.from_yaml(
        root / "configs/dynamic_hypergraph_v2431_qwen_smoke20.yaml"
    )
    shadow = DynamicV2ResearchConfig.from_yaml(
        root / "configs/dynamic_hypergraph_v2431_qwen_shadow20.yaml"
    )
    smoke.validate()
    shadow.validate()
    assert smoke.operation_conditioned_obligation_closure
    assert smoke.exact_fidelity_resource_accounting
    assert smoke.marginal_fidelity_evc_gate
    assert smoke.critical_obligation_budget_reserve
    assert smoke.split_manifest_path != shadow.split_manifest_path
    assert smoke.merged(
        split_manifest_path=shadow.split_manifest_path,
        campaign_id=shadow.campaign_id,
        campaign_ledger_path=shadow.campaign_ledger_path,
    ) == shadow
    prereg = json.loads(
        (root / "configs/dynamic_v2431_preregistration.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (root / "configs/dynamic_v2431_offline_replay.json").read_text(encoding="utf-8")
    )
    assert prereg["stage_caps"]["smoke_a20"]["provider_attempts"] == 250
    assert prereg["stage_caps"]["smoke_a20"]["provider_reported_tokens"] == 250_000
    assert replay["gold_used"] is False
    assert replay["decision"] == "GO_SOURCE_FREEZE_AND_SMOKE_A"
