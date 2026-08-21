#!/usr/bin/env python3
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _nested_keys(value: Any) -> set[str]:
    """Return every mapping key in a nested trace payload."""
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for child in value.values():
            keys.update(_nested_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_nested_keys(child))
        return keys
    return set()


def _test_gate() -> dict[str, Any]:
    command = [
        "python", "-m", "pytest", "-q",
        "tests_research/test_dynamic_v2_core.py",
        "tests_research/test_dynamic_v2_integration.py",
        "tests_research/test_data.py",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "passed": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-5:],
        "stderr_tail": completed.stderr.splitlines()[-5:],
    }


def _mean_normalized_cost(
    run: Path, reference_caps: tuple[int, int, int] = (16, 16_000, 8),
) -> float:
    cost = _json(run / "cost_summary.json")
    metrics = _json(run / "metrics.json")
    count = max(1, int(metrics.get("count", 0)))
    call_cap, token_cap, retrieval_cap = reference_caps
    return (
        float(cost.get("llm_calls", 0)) / count / call_cap
        + float(cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0)) / count / token_cap
        + float(cost.get("retrieval_calls", 0)) / count / retrieval_cap
    ) / 3.0


def _pareto(v2_run: Path, controls: list[Path], gate: dict[str, Any]) -> dict[str, Any]:
    v2_metrics = _json(v2_run / "metrics.json")
    v2_f1 = float(v2_metrics.get("f1", 0.0))
    v2_cost = _mean_normalized_cost(v2_run)
    policy = gate["computation_allocation"]
    comparisons = []
    passed = False
    for control in controls:
        metrics = _json(control / "metrics.json")
        control_f1 = float(metrics.get("f1", 0.0))
        control_cost = _mean_normalized_cost(control)
        v2_qids = {str(row.get("qid")) for row in _jsonl(v2_run / "predictions.jsonl")}
        control_qids = {str(row.get("qid")) for row in _jsonl(control / "predictions.jsonl")}
        comparable = bool(v2_qids) and v2_qids == control_qids
        cost_reduction = (control_cost - v2_cost) / max(control_cost, 1e-12)
        cost_difference = abs(v2_cost - control_cost) / max(control_cost, 1e-12)
        quality_difference = v2_f1 - control_f1
        noninferior_cheaper = (
            quality_difference >= -float(policy["pareto_noninferior_f1_tolerance"])
            and cost_reduction >= float(policy["pareto_cost_reduction_at_noninferior_f1"])
        )
        matched_cost_better = (
            cost_difference <= float(policy["pareto_matched_cost_tolerance"])
            and quality_difference >= float(policy["pareto_quality_gain_at_matched_cost"])
        )
        point_passed = comparable and (noninferior_cheaper or matched_cost_better)
        passed = passed or point_passed
        comparisons.append({
            "control_run": str(control),
            "v2_f1": v2_f1, "control_f1": control_f1,
            "v2_normalized_cost": v2_cost, "control_normalized_cost": control_cost,
            "quality_difference": quality_difference,
            "relative_cost_reduction": cost_reduction,
            "relative_cost_difference": cost_difference,
            "same_qid_set": comparable,
            "passed": point_passed,
        })
    return {"passed": passed, "comparisons": comparisons}


def evaluate(v1_run: Path, v2_run: Path, controls: list[Path]) -> dict[str, Any]:
    gate = _json(Path("configs/dynamic_v2_hard_gate.json"))
    tests = _test_gate()
    v1_metrics = _json(v1_run / "metrics.json")
    v1_dynamic = _json(v1_run / "dynamic_metrics.json")
    v2_metrics = _json(v2_run / "metrics.json")
    v2_dynamic = _json(v2_run / "dynamic_v2_metrics.json")
    v2_rows = _jsonl(v2_run / "dynamic_v2_per_example_metrics.jsonl")
    config = yaml.safe_load((v2_run / "resolved_config.yaml").read_text(encoding="utf-8"))
    reasoning = _jsonl(v2_run / "reasoning_traces.jsonl")
    predictions = _jsonl(v2_run / "predictions.jsonl")
    forbidden = {
        "answers", "gold_document_ids", "gold_titles", "oracle_decomposition",
        "supporting_paragraphs", "hop_count",
    }
    leaked_keys = sorted(forbidden.intersection(_nested_keys(reasoning)))
    allocation_events = [
        row for row in reasoning if row.get("event") == "allocation_reconciled"
    ]
    allocation_candidates = [
        candidate
        for row in reasoning if row.get("event") == "meta_decision"
        for candidate in row.get("allocation_candidates", [])
    ]
    expected_allocation_count = sum(int(row.get("allocation_count", 0)) for row in v2_rows)
    allocations_fully_reconciled = (
        bool(allocation_events) and len(allocation_events) == expected_allocation_count
    )
    predicted_evc_recorded = bool(allocation_candidates) and all(
        row.get("predicted_evc") is not None
        and bool(row.get("evc_components_raw"))
        and bool(row.get("evc_components_normalized"))
        and bool(row.get("requested_budget"))
        for row in allocation_candidates
    )
    actual_cost_recorded = bool(allocation_events) and all(
        bool(row.get("actual_cost")) for row in allocation_events
    )
    join_cases = sum(bool(row.get("auditable_three_or_four_hop_join_case")) for row in v2_rows)
    natural_count = sum(int(row.get("natural_revision_count", 0)) for row in v2_rows)
    correct = sum(int(row.get("natural_revision_correct", 0)) for row in v2_rows)
    wrong = sum(int(row.get("natural_revision_wrong", 0)) for row in v2_rows)
    natural_precision = correct / max(1, correct + wrong)
    terminal_by_qid = {
        str(row.get("qid")): str(row.get("termination_outcome")) for row in v2_rows
    }
    status_to_terminal = {
        "answer": "ANSWER", "abstain": "ABSTAIN",
        "budget_exhausted": "BUDGET_EXHAUSTED",
    }
    termination_rows_consistent = bool(predictions) and len(predictions) == len(v2_rows) and all(
        status_to_terminal.get(str(row.get("status"))) == terminal_by_qid.get(str(row.get("qid")))
        and ((str(row.get("status")) == "answer") == bool(row.get("answer")))
        for row in predictions
    )
    observed_outcomes = set(terminal_by_qid.values())
    required_outcomes = set(gate["termination"]["required_outcomes"])
    checks = {
        "infrastructure.zero_leakage": (
            not leaked_keys and not config.get("oracle_evidence") and not config.get("oracle_decomposition")
        ),
        "infrastructure.zero_invariant_violation": (
            tests["passed"] and float(v2_metrics.get("infrastructure_failure_rate", 1.0)) == 0.0
        ),
        "infrastructure.controller_only_mutation": (
            tests["passed"] and float(v2_dynamic.get("controller_state_hash_present_rate", 0.0)) == 1.0
        ),
        "reasoning.candidate_presence_gain": (
            float(v2_dynamic.get("candidate_presence_rate", 0.0))
            >= float(v1_dynamic.get("gold_candidate_generated_rate", 0.0))
            + float(gate["reasoning_capability"]["candidate_presence_absolute_gain_over_v1"])
        ),
        "reasoning.full_chain_gain": (
            float(v2_metrics.get("full_chain_completion_rate", 0.0))
            >= float(v1_metrics.get("full_chain_completion_rate", 0.0))
            + float(gate["reasoning_capability"]["full_chain_completion_absolute_gain_over_v1"])
        ),
        "reasoning.auditable_join_cases": (
            join_cases >= int(gate["reasoning_capability"]["minimum_auditable_three_or_four_hop_join_cases"])
        ),
        "dynamic.adversarial_revision": tests["passed"],
        "dynamic.natural_revision_precision": (
            natural_count >= int(gate["dynamic_behavior"]["minimum_natural_revision_count"])
            and natural_precision >= float(gate["dynamic_behavior"]["minimum_natural_revision_precision"])
        ),
        "dynamic.non_uniform_allocation": any(bool(row.get("non_uniform_allocation")) for row in v2_rows),
        "allocation.complete_evc_trace": (
            bool(v2_rows)
            and allocations_fully_reconciled
            and all(bool(row.get("complete_evc_trace")) for row in v2_rows)
        ),
        "allocation.predicted_evc_recorded": predicted_evc_recorded,
        "allocation.actual_cost_recorded": actual_cost_recorded,
        "termination.no_unsupported_answer": int(v2_dynamic.get("unsupported_answer_count", -1)) == 0,
        "termination.separated_outcomes": (
            tests["passed"] and termination_rows_consistent
            and required_outcomes.issubset(observed_outcomes)
        ),
    }
    pareto = _pareto(v2_run, controls, gate)
    checks["allocation.pareto_improvement"] = pareto["passed"]
    return {
        "schema_version": "dynamic-hypergraph-v2-gate-evaluation-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "v1_run": str(v1_run), "v2_run": str(v2_run),
            "control_runs": [str(value) for value in controls],
            "test_gate": tests, "leaked_keys": leaked_keys,
            "auditable_three_or_four_hop_join_cases": join_cases,
            "natural_revision_count": natural_count,
            "natural_revision_precision": natural_precision,
            "allocation_event_count": len(allocation_events),
            "allocation_candidate_count": len(allocation_candidates),
            "expected_allocation_count": expected_allocation_count,
            "allocations_fully_reconciled": allocations_fully_reconciled,
            "predicted_evc_recorded": predicted_evc_recorded,
            "actual_cost_recorded": actual_cost_recorded,
            "termination_rows_consistent": termination_rows_consistent,
            "observed_termination_outcomes": sorted(observed_outcomes),
            "pareto": pareto,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Dynamic Hypergraph v2 hard gate")
    parser.add_argument("--v1-run", type=Path, required=True)
    parser.add_argument("--v2-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.v1_run, args.v2_run, args.control_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
