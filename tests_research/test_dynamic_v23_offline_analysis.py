from scripts.analyze_dynamic_v23_offline import (
    allocation_calibration,
    join_frontier_audit,
    ready_set_audit,
    spearman,
    terminal_bottlenecks,
)


def test_spearman_handles_order_and_ties():
    assert spearman([1.0, 2.0, 3.0], [2.0, 4.0, 8.0]) == 1.0
    assert spearman([1.0, 2.0, 3.0], [8.0, 4.0, 2.0]) == -1.0
    assert spearman([1.0, 1.0], [0.0, 1.0]) is None


def _packet(operation, family, region, fidelity, evc):
    return {
        "operation_id": operation,
        "operation_family": family,
        "region_key": region,
        "fidelity_level": fidelity,
        "predicted_evc": evc,
    }


def test_ready_set_separates_fidelity_only_from_real_region_choice():
    rows = [
        {
            "qid": "q1", "event": "meta_decision", "step": 1,
            "outcome": "CONTINUE",
            "allocation_candidates": [
                _packet("retrieve-a", "retrieve:default", "a", "low", 2.0),
                _packet("retrieve-a", "retrieve:default", "a", "high", 1.0),
            ],
        },
        {
            "qid": "q1", "event": "meta_decision", "step": 2,
            "outcome": "CONTINUE",
            "allocation_candidates": [
                _packet("verify-a", "verify:default", "a", "high", 2.0),
                _packet("retrieve-b", "retrieve:default", "b", "low", 1.0),
            ],
        },
    ]
    report = ready_set_audit(rows)
    assert report["decision_count_with_candidates"] == 2
    assert report["fidelity_only_choice_rate"] == 0.5
    assert report["real_operation_choice_rate"] == 0.5
    assert report["cross_family_choice_rate"] == 0.5
    assert report["cross_region_choice_rate"] == 0.5


def test_allocation_calibration_reads_selected_outcomes():
    rows = []
    for index, (evc, utility) in enumerate(((1.0, -0.2), (2.0, 0.1), (3.0, 0.8))):
        rows.append({
            "qid": f"q{index}", "event": "allocation_reconciled",
            "allocation": {
                "operation_id": f"op{index}", "operation_family": "verify:default",
                "region_key": "r", "fidelity_level": "high", "predicted_evc": evc,
            },
            "actual_utility": utility,
            "progressed": utility > 0,
            "actual_utility_components_raw": {
                "terminal_gap_reduction": max(0.0, utility),
                "answer_chain_progress": max(0.0, utility),
            },
        })
    report = allocation_calibration(rows)
    assert report["overall"]["count"] == 3
    assert report["overall"]["spearman_predicted_evc_actual_utility"] == 1.0
    assert report["by_operation_family"]["verify:default"]["progress_rate"] == 2 / 3


def test_allocation_calibration_reports_real_choice_conditioned_correlation():
    rows = []
    for index, (evc, utility) in enumerate(((1.0, -0.2), (2.0, 0.1), (3.0, 0.8))):
        base_id = f"op{index}"
        rows.extend([
            {
                "qid": f"q{index}", "event": "meta_decision", "outcome": "CONTINUE",
                "allocation_candidates": [
                    _packet(base_id, "retrieve:default", f"r{index}", "high", evc),
                    _packet(f"alternative{index}", "branch:extract_typed", f"x{index}", "high", evc - 0.1),
                ],
            },
            {
                "qid": f"q{index}", "event": "allocation_reconciled",
                "allocation": _packet(
                    f"{base_id}_allocation_000001", "retrieve:default",
                    f"r{index}", "high", evc,
                ),
                "actual_utility": utility,
                "progressed": utility > 0,
                "actual_utility_components_raw": {},
            },
        ])
    report = allocation_calibration(rows)
    conditioned = report["choice_conditioned"]
    assert conditioned["count"] == 3
    assert conditioned["trace_match_rate"] == 1.0
    assert conditioned["minimum_choice_size"] == 2
    assert conditioned["spearman_predicted_evc_actual_utility"] == 1.0


def test_terminal_bottlenecks_are_posthoc_and_mutually_prioritized():
    predictions = [
        {"qid": "q1", "status": "abstain", "stop_reason": "none"},
        {"qid": "q2", "status": "abstain", "stop_reason": "none"},
        {"qid": "q3", "status": "budget_exhausted", "stop_reason": "cap"},
    ]
    metrics = [
        {"qid": "q1", "answer_in_context": True, "all_gold_recalled": True, "full_chain_complete": False},
        {"qid": "q2", "answer_in_context": False, "all_gold_recalled": False, "full_chain_complete": False},
        {"qid": "q3", "answer_in_context": True, "all_gold_recalled": True, "full_chain_complete": False},
    ]
    dynamic = [
        {"qid": "q1", "candidate_presence": False, "candidate_survival": False},
        {"qid": "q2", "candidate_presence": False, "candidate_survival": False},
        {"qid": "q3", "candidate_presence": True, "candidate_survival": True},
    ]
    report = terminal_bottlenecks(predictions, metrics, dynamic, [])
    assert report["bottleneck_counts"] == {
        "context_to_candidate_extraction": 1,
        "retrieval_access": 1,
        "budget_exhaustion": 1,
    }


def test_join_frontier_audit_separates_charged_rejection_and_answer_use():
    graphs = [{
        "qid": "q1",
        "graph": {
            "nodes": {
                "answer": {
                    "kind": "answer", "status": "accepted",
                    "supporting_claims": ["joined"],
                },
            },
            "join_attempt_history": [
                {
                    "operation_id": "j1", "join_kind": "relational_path",
                    "premise_ids": ["c1", "c2"], "accepted": True,
                    "conclusion_node_id": "joined", "creation_cost": {"llm_calls": 0},
                    "deterministic_validation": {"goal_alignment": 0.8},
                    "downstream_unlock": 1.0,
                },
                {
                    "operation_id": "j2", "join_kind": "shared_role",
                    "premise_ids": ["c2", "c3"], "accepted": False,
                    "rejection_reason": "join_rejected",
                    "creation_cost": {"llm_calls": 1, "tokens": 120},
                    "deterministic_validation": {"goal_alignment": 0.1},
                },
            ],
        },
    }]
    report = join_frontier_audit(graphs)
    assert report["attempt_count"] == 2
    assert report["accepted_count"] == 1
    assert report["charged_count"] == 2
    assert report["answer_used_count"] == 1
    assert report["llm_calls"] == 1
    assert report["rejection_reason_counts"] == {"join_rejected": 1}
