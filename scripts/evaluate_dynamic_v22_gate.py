#!/usr/bin/env python3
"""Fail-closed evaluator for the preregistered Dynamic Hypergraph TDCA v2.2 gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(map(str, value)) | set().union(*(
            _nested_keys(child) for child in value.values()
        ), set())
    if isinstance(value, list):
        return set().union(*(_nested_keys(child) for child in value), set())
    return set()


def _test_gate() -> dict[str, Any]:
    command = ["python", "-m", "pytest", "-q"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "passed": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-6:],
        "stderr_tail": completed.stderr.splitlines()[-6:],
    }


def _identity(run: Path) -> dict[str, Any]:
    config = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    manifest = _json(run / "run_manifest.json")
    predictions = _jsonl(run / "predictions.jsonl")
    progress = _json(run / "partial_progress.json")
    return {
        "allocator_mode": str(config.get("allocator_mode", "")),
        "sample_ids": [str(row.get("qid")) for row in predictions],
        "dataset_sha256": str(manifest.get("dataset_sha256", "")),
        "model": str(manifest.get("model", "")),
        "prompt_version": str(manifest.get("prompt_version", "")),
        "split_seed": int(manifest.get("split_seed", -1)),
        "budget_caps": {
            key: int(config.get(key, 0)) for key in (
                "max_llm_calls", "max_total_tokens", "max_retrieval_calls",
            )
        },
        "config_except_allocator": {
            key: value for key, value in config.items() if key != "allocator_mode"
        },
        "complete": progress.get("status") == "complete" and bool(predictions),
    }


def _cost(run: Path) -> dict[str, int]:
    row = _json(run / "cost_summary.json")
    return {
        "provider_calls": int(row.get("provider_calls", 0)),
        "provider_reported_tokens": (
            int(row.get("provider_prompt_tokens", 0))
            + int(row.get("provider_completion_tokens", 0))
        ),
        "logical_llm_calls": int(row.get("llm_calls", 0)),
        "retrieval_calls": int(row.get("retrieval_calls", 0)),
    }


def _strict_pareto(adaptive: Path, controls: list[Path]) -> dict[str, Any]:
    adaptive_identity = _identity(adaptive)
    adaptive_metrics = _json(adaptive / "metrics.json")
    adaptive_cost = _cost(adaptive)
    comparisons = []
    modes = set()
    for control in controls:
        identity = _identity(control)
        metrics = _json(control / "metrics.json")
        cost = _cost(control)
        mode = identity["allocator_mode"]
        modes.add(mode)
        matched = (
            adaptive_identity["sample_ids"] == identity["sample_ids"]
            and bool(adaptive_identity["sample_ids"])
            and adaptive_identity["dataset_sha256"] == identity["dataset_sha256"]
            and adaptive_identity["model"] == identity["model"]
            and adaptive_identity["prompt_version"] == identity["prompt_version"]
            and adaptive_identity["split_seed"] == identity["split_seed"]
            and adaptive_identity["budget_caps"] == identity["budget_caps"]
            and adaptive_identity["config_except_allocator"] == identity["config_except_allocator"]
            and adaptive_identity["complete"] and identity["complete"]
        )
        adaptive_f1 = float(adaptive_metrics.get("f1", 0.0))
        control_f1 = float(metrics.get("f1", 0.0))
        quality_noninferior = adaptive_f1 >= control_f1
        cost_non_higher = all(
            adaptive_cost[key] <= cost[key]
            for key in ("provider_reported_tokens", "logical_llm_calls", "retrieval_calls")
        )
        strict = (
            adaptive_f1 > control_f1
            or any(
                adaptive_cost[key] < cost[key]
                for key in ("provider_reported_tokens", "logical_llm_calls", "retrieval_calls")
            )
        )
        passed = matched and mode in {"uniform", "fixed_order"} and quality_noninferior and cost_non_higher and strict
        comparisons.append({
            "control_run": str(control), "control_allocator_mode": mode,
            "matched_compute_identity": matched,
            "adaptive_f1": adaptive_f1, "control_f1": control_f1,
            "adaptive_cost": adaptive_cost, "control_cost": cost,
            "quality_noninferior": quality_noninferior,
            "cost_non_higher_on_every_primary_axis": cost_non_higher,
            "strict_improvement": strict, "passed": passed,
        })
    return {
        "passed": (
            adaptive_identity["allocator_mode"] == "adaptive_evc"
            and modes == {"uniform", "fixed_order"}
            and len(comparisons) == 2
            and all(row["passed"] for row in comparisons)
        ),
        "comparisons": comparisons,
    }


def _campaign_audit(ledger: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    usage = ledger.get("usage", {})
    limits = ledger.get("limits", {})
    events = ledger.get("events", [])
    reservations = [row for row in events if row.get("event") == "request_reserved"]
    settlements = [
        row for row in events
        if row.get("event") in {"request_settled", "request_reconciled_from_cache"}
    ]
    request_ids = [str(row.get("request_id", "")) for row in reservations]
    settlement_ids = [str(row.get("request_id", "")) for row in settlements]
    recorded_tokens = sum(int(row.get("provider_reported_tokens", 0)) for row in settlements)
    call_cap = int(gate["computation_allocation"]["provider_call_campaign_cap"])
    token_cap = int(gate["computation_allocation"]["provider_reported_token_campaign_cap"])
    checks = {
        "schema": ledger.get("schema_version") == "tdca-campaign-budget-v1",
        "configured_caps": (
            int(limits.get("provider_calls", -1)) == call_cap
            and int(limits.get("provider_reported_tokens", -1)) == token_cap
        ),
        "within_caps": (
            int(usage.get("provider_calls", call_cap + 1)) <= call_cap
            and int(usage.get("provider_reported_tokens", token_cap + 1)) <= token_cap
        ),
        "no_pending_requests": not ledger.get("pending"),
        "active_not_denied": (
            ledger.get("status") == "active"
            and not any(row.get("event") == "request_denied" for row in events)
        ),
        "every_call_reserved_once": (
            len(reservations) == int(usage.get("provider_calls", -1))
            and len(request_ids) == len(set(request_ids))
            and all(request_ids)
        ),
        "every_reservation_settled_once": (
            len(settlements) == len(reservations)
            and sorted(settlement_ids) == sorted(request_ids)
        ),
        "provider_tokens_reconcile": (
            recorded_tokens == int(usage.get("provider_reported_tokens", -1))
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "snapshot": ledger}


def _terminal_audit(graph_rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = 0
    failures = []
    for row in graph_rows:
        graph = row.get("graph", {})
        nodes = graph.get("nodes", {})
        terminal = graph.get("terminal_beliefs", {})
        for answer_id, answer in nodes.items():
            if answer.get("kind") != "answer" or answer.get("status") != "accepted":
                continue
            answered += 1
            belief = terminal.get(answer_id)
            if not belief or not belief.get("accepted") or not belief.get("sufficient_chain"):
                failures.append({"qid": row.get("qid"), "answer_id": answer_id, "reason": "missing_or_failed_readout"})
                continue
            if belief.get("rejection_reasons"):
                failures.append({"qid": row.get("qid"), "answer_id": answer_id, "reason": "readout_has_rejections"})
            if set(belief.get("supporting_claims", [])) != set(answer.get("supporting_claims", [])):
                failures.append({"qid": row.get("qid"), "answer_id": answer_id, "reason": "claim_mismatch"})
            if set(belief.get("supporting_evidence", [])) != set(answer.get("supporting_evidence", [])):
                failures.append({"qid": row.get("qid"), "answer_id": answer_id, "reason": "evidence_mismatch"})
    return {"passed": not failures, "answered": answered, "failures": failures}


def evaluate(
    v1_run: Path,
    adaptive_run: Path,
    controls: list[Path],
    revision_eval: Path,
    campaign_ledger: Path,
    budget_curve: Path,
) -> dict[str, Any]:
    gate = _json(Path("configs/dynamic_v22_hard_gate.json"))
    tests = _test_gate()
    metrics = _json(adaptive_run / "metrics.json")
    dynamic = _json(adaptive_run / "dynamic_v2_metrics.json")
    per_example = _jsonl(adaptive_run / "dynamic_v2_per_example_metrics.jsonl")
    graphs = _jsonl(adaptive_run / "dynamic_v2_graphs.jsonl")
    reasoning = _jsonl(adaptive_run / "reasoning_traces.jsonl")
    predictions = _jsonl(adaptive_run / "predictions.jsonl")
    config = yaml.safe_load((adaptive_run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    revision = _json(revision_eval).get("metrics", {})
    curve = _json(budget_curve)
    campaign = _campaign_audit(_json(campaign_ledger), gate)
    terminal = _terminal_audit(graphs)
    pareto = _strict_pareto(adaptive_run, controls)
    forbidden = {
        "answers", "gold_document_ids", "gold_titles", "oracle_decomposition",
        "supporting_paragraphs", "hop_count",
    }
    leaked_keys = sorted(forbidden & _nested_keys(reasoning))
    allocations = [
        candidate
        for row in reasoning if row.get("event") == "meta_decision"
        for candidate in row.get("allocation_candidates", [])
    ]
    outcomes = [row for row in reasoning if row.get("event") == "allocation_reconciled"]
    expected_allocations = sum(int(row.get("allocation_count", 0)) for row in per_example)
    statuses = {"answer": "ANSWER", "abstain": "ABSTAIN", "budget_exhausted": "BUDGET_EXHAUSTED"}
    terminal_by_qid = {str(row.get("qid")): row.get("termination_outcome") for row in per_example}
    adaptive_manifest = _json(adaptive_run / "run_manifest.json")
    code_version = str(adaptive_manifest.get("code_version", ""))
    source_tree_sha256 = (
        code_version.split("source-tree-sha256:", 1)[1]
        if "source-tree-sha256:" in code_version else ""
    )
    termination_consistent = (
        len(predictions) == len(per_example) == int(metrics.get("count", -1))
        and all(
            statuses.get(str(row.get("status"))) == terminal_by_qid.get(str(row.get("qid")))
            and ((row.get("status") == "answer") == bool(row.get("answer")))
            for row in predictions
        )
    )
    join_cases = sum(bool(row.get("auditable_three_or_four_hop_join_case")) for row in per_example)
    nary_cases = sum(int(row.get("nary_join_downstream_used_count", 0)) > 0 for row in per_example)
    checks = {
        "infrastructure.zero_leakage": not leaked_keys and not config.get("oracle_evidence") and not config.get("oracle_decomposition"),
        "infrastructure.zero_invariant_violation": tests["passed"] and float(metrics.get("infrastructure_failure_rate", 1.0)) == 0.0,
        "infrastructure.controller_only_mutation": float(dynamic.get("controller_state_hash_present_rate", 0.0)) == 1.0,
        "infrastructure.zero_unaccounted_provider_calls": campaign["passed"],
        "infrastructure.complete_serialization": tests["passed"] and len(graphs) == int(metrics.get("count", -1)),
        "reasoning.minimum_f1": float(metrics.get("f1", 0.0)) >= float(gate["reasoning_capability"]["minimum_official_f1"]),
        "reasoning.minimum_candidate_presence": float(dynamic.get("candidate_presence_rate", 0.0)) >= float(gate["reasoning_capability"]["minimum_candidate_presence"]),
        "reasoning.minimum_full_chain": float(metrics.get("full_chain_completion_rate", 0.0)) >= float(gate["reasoning_capability"]["minimum_full_chain_completion"]),
        "reasoning.auditable_join_cases": join_cases >= int(gate["reasoning_capability"]["minimum_auditable_three_or_four_hop_join_cases"]),
        "reasoning.downstream_nary_cases": nary_cases >= int(gate["reasoning_capability"]["minimum_downstream_used_nary_join_cases"]),
        "dynamic.adversarial_revision": tests["passed"],
        "dynamic.frozen_revision_metrics": (
            int(revision.get("tp", 0)) + int(revision.get("fn", 0)) >= 30
            and int(revision.get("tn", 0)) + int(revision.get("fp", 0)) >= 30
            and float(revision.get("precision", 0.0)) >= 0.80
            and float(revision.get("recall", 0.0)) >= 0.60
            and float(revision.get("false_positive_rate", 1.0)) <= 0.10
            and bool(revision.get("complete_predictions"))
            and bool(revision.get("zero_invariant_violations"))
        ),
        "dynamic.non_uniform_allocation": any(bool(row.get("non_uniform_allocation")) for row in per_example),
        "dynamic.outcome_changes_later_allocation": any(bool(row.get("feedback_influenced_allocation")) for row in per_example),
        "allocation.complete_evc": (
            bool(allocations) and len(outcomes) == expected_allocations
            and all(
                row.get("predicted_evc") is not None
                and row.get("evc_components_raw") and row.get("evc_components_normalized")
                and row.get("requested_budget") and row.get("pre_state_summary")
                and {"terminal_gap", "terminal_proximity"}.issubset(row["evc_components_raw"])
                for row in allocations
            )
        ),
        "allocation.actual_cost_utility_delta": (
            bool(outcomes) and all(
                row.get("actual_cost") and row.get("post_state_summary")
                and row.get("state_delta") and row.get("actual_utility_components_raw")
                and row.get("actual_utility_components_normalized")
                and "terminal_gap_reduction" in row["actual_utility_components_raw"]
                for row in outcomes
            )
        ),
        "allocation.terminal_gap_trace": float(dynamic.get("complete_terminal_gap_trace_rate", 0.0)) == 1.0,
        "allocation.matched_budget_curve": (
            curve.get("schema_version") == "dynamic-hypergraph-v2.2-budget-curve-v1"
            and bool(curve.get("complete")) and int(curve.get("point_count", 0)) >= 2
        ),
        "allocation.pareto_against_both": pareto["passed"],
        "termination.no_unsupported_answer": int(dynamic.get("unsupported_answer_count", -1)) == 0,
        "termination.separated_outcomes": tests["passed"] and termination_consistent,
        "termination.terminal_belief_and_sufficient_chain": terminal["passed"] and float(dynamic.get("complete_terminal_belief_readout_rate", 0.0)) == 1.0,
    }
    return {
        "schema_version": "dynamic-hypergraph-v2.2-gate-evaluation-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "v1_run": str(v1_run), "v1_metrics": _json(v1_run / "metrics.json"),
            "adaptive_run": str(adaptive_run), "control_runs": [str(path) for path in controls],
            "adaptive_code_version": code_version,
            "adaptive_source_tree_sha256": source_tree_sha256,
            "test_gate": tests, "leaked_keys": leaked_keys,
            "join_cases": join_cases, "downstream_nary_cases": nary_cases,
            "revision_evaluation": str(revision_eval), "revision_metrics": revision,
            "campaign_audit": campaign, "terminal_audit": terminal,
            "allocation_candidate_count": len(allocations),
            "allocation_outcome_count": len(outcomes),
            "expected_allocation_count": expected_allocations,
            "termination_consistent": termination_consistent,
            "observed_termination_outcomes": sorted(set(terminal_by_qid.values())),
            "budget_curve": curve, "pareto": pareto,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-run", type=Path, required=True)
    parser.add_argument("--adaptive-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, action="append", required=True)
    parser.add_argument("--revision-eval", type=Path, required=True)
    parser.add_argument("--campaign-ledger", type=Path, required=True)
    parser.add_argument("--budget-curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.v1_run, args.adaptive_run, args.control_run,
        args.revision_eval, args.campaign_ledger, args.budget_curve,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
