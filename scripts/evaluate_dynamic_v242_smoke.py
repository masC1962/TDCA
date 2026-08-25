#!/usr/bin/env python3
"""Evaluate the preregistered v2.4.2 smoke-A gate without inference calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_dynamic_v242_offline import analyze
from scripts.verify_artifact import verify
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.utils import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(run: Path, preregistration: Path, config_path: Path) -> dict[str, Any]:
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
    offline = analyze(run, DynamicV2ResearchConfig.from_yaml(config_path))
    overall = offline["overall"]
    choice = offline["choice_conditioned"]
    recovery = offline["proof_gap_recovery"]
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
        "immediate_evc_correlation": overall[
            "spearman_predicted_immediate_actual_immediate"
        ],
        "delayed_evc_correlation": overall[
            "spearman_predicted_delayed_actual_delayed"
        ],
        "choice_conditioned_delayed_correlation": choice[
            "spearman_predicted_delayed_actual_delayed"
        ],
        "choice_conditioned_count": int(choice["count"]),
        "proof_gap_recovery": recovery,
        "complete_evc_trace_rate": float(dynamic["complete_evc_trace_rate"]),
        "complete_delayed_credit_trace_rate": float(
            dynamic.get("complete_delayed_credit_trace_rate", 0.0)
        ),
        "non_uniform_allocation_rate": float(dynamic["non_uniform_allocation_rate"]),
        "termination_outcomes": dynamic["termination_outcomes"],
    }
    success_mean = float(recovery["mean_successful_delayed_return"])
    failed_mean = float(recovery["mean_failed_delayed_return"])
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
        "positive_immediate_calibration": evidence["immediate_evc_correlation"] is not None and evidence["immediate_evc_correlation"] >= limits["immediate_evc_correlation_min"],
        "positive_delayed_calibration": evidence["delayed_evc_correlation"] is not None and evidence["delayed_evc_correlation"] >= limits["delayed_evc_correlation_min"],
        "positive_choice_conditioned_delayed_calibration": evidence["choice_conditioned_delayed_correlation"] is not None and evidence["choice_conditioned_delayed_correlation"] > limits["choice_conditioned_delayed_correlation_exclusive_min"],
        "choice_conditioned_sample_size": evidence["choice_conditioned_count"] >= limits["choice_conditioned_count_min"],
        "successful_proof_gap_has_positive_delayed_return": recovery["successful_count"] > 0 and success_mean > limits["successful_proof_gap_mean_delayed_return_exclusive_min"],
        "successful_proof_gap_exceeds_failed": success_mean > failed_mean,
        "complete_evc_trace": evidence["complete_evc_trace_rate"] >= limits["complete_evc_trace_rate_min"],
        "complete_delayed_credit_trace": evidence["complete_delayed_credit_trace_rate"] >= limits["complete_delayed_credit_trace_rate_min"],
        "non_uniform_allocation": evidence["non_uniform_allocation_rate"] > 0.0,
        "terminal_outcomes_typed": set(evidence["termination_outcomes"]).issubset({
            "ANSWER", "ABSTAIN", "BUDGET_EXHAUSTED",
        }),
    }
    passed = all(checks.values())
    return {
        "schema_version": "dynamic-hypergraph-v2.4.2-adaptive-smoke-a-gate-v1",
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
        default=Path("configs/dynamic_v242_preregistration.json"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/dynamic_hypergraph_v242_qwen_smoke20.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.run, args.preregistration, args.config)
    write_json(args.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "failed_checks": report["failed_checks"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
