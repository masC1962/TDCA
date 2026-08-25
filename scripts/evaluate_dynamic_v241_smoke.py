#!/usr/bin/env python3
"""Evaluate the preregistered v2.4.1 smoke-A gate without inference calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_dynamic_v23_offline import analyze
from scripts.verify_artifact import verify
from tdca_research.utils import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(run: Path, preregistration: Path) -> dict[str, Any]:
    prereg = _json(preregistration)
    limits = prereg["adaptive_smoke_a20_hard_gates"]
    artifact = verify(run, expected_count=20)
    metrics = _json(run / "metrics.json")
    dynamic = _json(run / "dynamic_v2_metrics.json")
    cost = _json(run / "cost_summary.json")
    config = _json(run / "resolved_config.json") if (run / "resolved_config.json").exists() else {}
    if not config:
        import yaml
        config = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    reasoning = _jsonl(run / "reasoning_traces.jsonl")
    failures = (run / "failures.jsonl").read_text(encoding="utf-8")
    offline = analyze(run)
    overall_corr = offline["allocation_calibration"]["overall"].get(
        "spearman_predicted_evc_actual_utility"
    )
    choice_corr = offline["allocation_calibration"]["choice_conditioned"].get(
        "spearman_predicted_evc_actual_utility"
    )
    no_diff_editor = sum(
        row.get("event") == "allocation_reconciled"
        and str((row.get("allocation") or {}).get("operation_family", "")).startswith("expand:")
        and not bool(row.get("progressed"))
        for row in reasoning
    )
    evidence = {
        "artifact_verified": bool(artifact["verified"]),
        "zero_gold_or_oracle_inference": not bool(config.get("oracle_evidence"))
        and not bool(config.get("oracle_decomposition")),
        "infrastructure_failure_count": int(artifact["infrastructure_failures"]),
        "graph_invariant_violation_count": failures.count("GraphInvariantError"),
        "controller_only_mutation_violation_count": failures.count("outside the V2 controller"),
        "unsupported_answer_count": int(dynamic["unsupported_answer_count"]),
        "selected_infeasible_join_count": float(
            dynamic.get("mean_selected_infeasible_join_count", 0.0)
        ) * int(dynamic["count"]),
        "repeated_same_fingerprint_extraction_count": float(
            dynamic.get("mean_repeated_same_fingerprint_extraction_count", 0.0)
        ) * int(dynamic["count"]),
        "no_diff_editor_allocation_count": no_diff_editor,
        "candidate_presence_rate": float(dynamic["candidate_presence_rate"]),
        "legacy_execution_plan_completion_rate": float(metrics.get(
            "execution_plan_completion_rate", metrics["full_chain_completion_rate"],
        )),
        "graph_proof_completion_rate": float(dynamic.get("graph_proof_completion_rate", 0.0)),
        "f1": float(metrics["f1"]),
        "logical_llm_calls": int(cost["llm_calls"]),
        "logical_total_tokens": int(cost["prompt_tokens"]) + int(cost["completion_tokens"]),
        "provider_attempts": int(cost.get("provider_attempts", cost["provider_calls"])),
        "provider_reported_tokens": int(cost.get("provider_prompt_tokens", 0))
        + int(cost.get("provider_completion_tokens", 0)),
        "budget_exhaustion_rate": float(metrics["budget_exhaustion_rate"]),
        "evc_utility_correlation": overall_corr,
        "choice_conditioned_evc_correlation": choice_corr,
        "choice_conditioned_count": int(
            offline["allocation_calibration"]["choice_conditioned"]["count"]
        ),
        "complete_evc_trace_rate": float(dynamic["complete_evc_trace_rate"]),
        "non_uniform_allocation_rate": float(dynamic["non_uniform_allocation_rate"]),
        "proof_gap_diagnosis_count": sum(
            row.get("event") == "proof_gap_diagnosed" for row in reasoning
        ),
        "editor_preallocation_filtered_count": sum(
            row.get("event") == "graph_editor_preallocation_filtered" for row in reasoning
        ),
        "termination_outcomes": dynamic["termination_outcomes"],
    }
    checks = {
        "artifact_complete": evidence["artifact_verified"],
        "zero_leakage": evidence["zero_gold_or_oracle_inference"],
        "zero_infrastructure_failure": evidence["infrastructure_failure_count"] <= limits["infrastructure_failure_count_max"],
        "zero_graph_invariant_violation": evidence["graph_invariant_violation_count"] <= limits["graph_invariant_violation_count_max"],
        "controller_only_mutation": evidence["controller_only_mutation_violation_count"] <= limits["controller_only_mutation_violation_count_max"],
        "zero_unsupported_answer": evidence["unsupported_answer_count"] <= limits["unsupported_answer_count_max"],
        "zero_selected_infeasible_join": evidence["selected_infeasible_join_count"] <= limits["selected_infeasible_join_count_max"],
        "zero_repeated_extraction_fingerprint": evidence["repeated_same_fingerprint_extraction_count"] <= limits["repeated_same_fingerprint_extraction_count_max"],
        "zero_no_diff_editor_allocation": evidence["no_diff_editor_allocation_count"] <= limits["no_diff_editor_allocation_count_max"],
        "candidate_presence": evidence["candidate_presence_rate"] >= limits["candidate_presence_rate_min"],
        "legacy_execution_plan_completion": evidence["legacy_execution_plan_completion_rate"] >= limits["legacy_execution_plan_completion_rate_min"],
        "graph_proof_completion": evidence["graph_proof_completion_rate"] >= limits["graph_proof_completion_rate_min"],
        "f1_non_regression": evidence["f1"] >= limits["f1_min"],
        "bounded_logical_calls": evidence["logical_llm_calls"] <= limits["logical_llm_calls_max"],
        "bounded_logical_tokens": evidence["logical_total_tokens"] <= limits["logical_total_tokens_max"],
        "bounded_budget_exhaustion": evidence["budget_exhaustion_rate"] <= limits["budget_exhaustion_rate_max"],
        "positive_evc_calibration": overall_corr is not None and overall_corr >= limits["evc_utility_correlation_min"],
        "positive_choice_conditioned_evc": choice_corr is not None and choice_corr > limits["choice_conditioned_evc_correlation_exclusive_min"],
        "complete_evc_trace": evidence["complete_evc_trace_rate"] >= limits["complete_evc_trace_rate_min"],
        "non_uniform_allocation": evidence["non_uniform_allocation_rate"] > 0.0,
        "terminal_outcomes_typed": set(evidence["termination_outcomes"]).issubset({
            "ANSWER", "ABSTAIN", "BUDGET_EXHAUSTED",
        }),
    }
    passed = all(checks.values())
    return {
        "schema_version": "dynamic-hypergraph-v2.4.1-adaptive-smoke-a-gate-v1",
        "inference_calls_made_by_gate": 0,
        "run": str(run),
        "preregistration": str(preregistration),
        "evidence": evidence,
        "checks": checks,
        "passed": passed,
        "decision": "GO_PAIRED_SHADOW_B" if passed else "SAFE_STOP",
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("configs/dynamic_v241_preregistration.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.run, args.preregistration)
    write_json(args.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "failed_checks": report["failed_checks"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
