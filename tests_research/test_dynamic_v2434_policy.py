from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from tdca_research.budget import Budget
from tdca_research.dynamic_v2.allocator import AdaptiveComputationAllocator
from tdca_research.dynamic_v2.termination import MetaStopPolicy, TerminalBeliefReadout
from tdca_research.dynamic_v2.transitions import certified_transition_value
from tdca_research.models import Usage

from tests_research.test_dynamic_v2_core import chain_graph, terminal_operation


def _terminal_fixture():
    base, controller, graph = chain_graph()
    cfg = replace(
        base,
        certified_transition_option_value=True,
        certified_terminal_materialization=True,
        certified_meta_stop=True,
    )
    accepted, diagnostics = TerminalBeliefReadout(cfg).evaluate(
        graph, [terminal_operation(70)],
    )
    assert len(accepted) == 1
    assert diagnostics[0]["accepted"]
    return cfg, controller, graph, accepted[0]


def test_v2434_accepted_terminal_materialization_is_certified_and_realized():
    cfg, controller, graph, operation = _terminal_fixture()
    certificate = certified_transition_value(graph, operation, cfg)
    assert certificate["kind"] == "accepted_terminal_materialization"
    assert certificate["mandatory"] is True
    assert certificate["deterministic"] is True
    assert certificate["provider_calls"] == 0
    assert certificate["predicted_transition_value"] == 1.0

    allocator = AdaptiveComputationAllocator(cfg)
    budget = Budget(16, 6000, 200, Usage())
    packet = allocator.allocate(graph, [operation], budget)[0]
    decision = MetaStopPolicy(cfg).decide(graph, [packet], budget)
    assert decision.outcome.value == "CONTINUE"
    assert decision.reason == "certified_state_transition"

    graph = controller.apply(graph, allocator.attach(operation, packet))
    graph = controller.reconcile_allocation(
        graph, packet, {"graph_operations": 1.0}, True,
    )
    allocation = graph.allocation_history[-1]
    assert allocation.transition_realized is True
    assert allocation.actual_transition_value == 1.0
    assert allocation.transition_observations["answer_materialized"] is True
    assert allocation.transition_observations["terminal_belief_accepted"] is True


def test_v2434_terminal_family_or_accepted_flag_cannot_forge_certificate():
    cfg, _, graph, operation = _terminal_fixture()
    del operation.payload["answer"]["terminal_competition"]
    assert certified_transition_value(graph, operation, cfg)["mandatory"] is False


def test_v2434_tampered_relative_competition_fails_closed():
    cfg, _, graph, operation = _terminal_fixture()
    rows = operation.payload["answer"]["terminal_competition"]["candidate_values"]
    rows[0]["absolute_support"] = max(0.0, rows[0]["absolute_support"] - 0.2)
    assert certified_transition_value(graph, operation, cfg)["mandatory"] is False


def test_v2434_feature_is_opt_in_for_frozen_versions():
    cfg, _, graph, operation = _terminal_fixture()
    frozen = replace(cfg, certified_terminal_materialization=False)
    assert certified_transition_value(graph, operation, frozen)["mandatory"] is False


def test_v2434_frozen_configs_and_gold_free_replay():
    root = Path(__file__).resolve().parents[1]
    smoke = type(_terminal_fixture()[0]).from_yaml(
        root / "configs/dynamic_hypergraph_v2434_qwen_smoke20.yaml"
    )
    shadow = type(smoke).from_yaml(
        root / "configs/dynamic_hypergraph_v2434_qwen_shadow20.yaml"
    )
    assert smoke.certified_terminal_materialization
    assert smoke.meta_stop_evc_threshold == 0.08
    assert smoke.campaign_id == shadow.campaign_id
    assert smoke.campaign_ledger_path == shadow.campaign_ledger_path
    assert smoke.campaign_provider_call_cap == 1339
    assert smoke.campaign_provider_token_cap == 1_263_513
    replay = json.loads((
        root / "configs/dynamic_v2434_offline_replay.json"
    ).read_text(encoding="utf-8"))
    assert replay["gold_used"] is False
    assert replay["accepted_terminal_readout_count"] == 13
    assert replay["blocked_terminal_materialization_count"] == 13
    assert replay["all_blocked_are_zero_provider"] is True
