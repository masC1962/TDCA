#!/usr/bin/env python3
"""Paired offline v2.4.1 versus v2.4.3 shadow-B safety gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_dynamic_v242_offline import analyze
from scripts.evaluate_dynamic_v242_shadow import _summary
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.utils import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(
    v241_run: Path, v243_run: Path, preregistration: Path, config_path: Path,
) -> dict[str, Any]:
    limits = _json(preregistration)["paired_shadow_b20_hard_gates"]
    baseline = _summary(v241_run)
    candidate = _summary(v243_run)
    dynamic = _json(v243_run / "dynamic_v2_metrics.json")
    candidate["complete_proof_obligation_trace_rate"] = float(
        dynamic.get("complete_proof_obligation_trace_rate", 0.0)
    )
    candidate["no_executable_without_certificate_count"] = int(
        dynamic.get("no_executable_without_certificate_count", 0)
    )
    candidate["abstain_has_exhaustion_evidence_rate"] = float(
        dynamic.get("abstain_has_exhaustion_evidence_rate", 0.0)
    )
    calibration = analyze(v243_run, DynamicV2ResearchConfig.from_yaml(config_path))
    delayed_corr = calibration["overall"]["spearman_predicted_delayed_actual_delayed"]
    safety_fields = (
        "infrastructure_failure_count", "graph_invariant_violation_count",
        "controller_only_mutation_violation_count", "unsupported_answer_count",
        "selected_infeasible_join_count", "repeated_same_fingerprint_extraction_count",
    )
    checks = {
        "both_artifacts_complete": baseline["artifact_verified"] and candidate["artifact_verified"],
        "both_zero_safety_violations": all(
            baseline[field] == 0 and candidate[field] == 0 for field in safety_fields
        ),
        "v243_candidate_presence_non_regression": (
            candidate["candidate_presence_rate"] >= baseline["candidate_presence_rate"]
        ),
        "v243_execution_plan_completion_non_regression": (
            candidate["execution_plan_completion_rate"]
            >= baseline["execution_plan_completion_rate"]
        ),
        "v243_graph_proof_completion_non_regression": (
            candidate["graph_proof_completion_rate"]
            >= baseline["graph_proof_completion_rate"]
        ),
        "v243_f1_non_regression": candidate["f1"] >= baseline["f1"],
        "v243_complete_proof_obligation_trace": (
            candidate["complete_proof_obligation_trace_rate"]
            >= limits["v243_complete_proof_obligation_trace_rate_min"]
        ),
        "v243_positive_delayed_calibration": (
            delayed_corr is not None
            and delayed_corr >= limits["v243_delayed_evc_correlation_min"]
        ),
        "v243_zero_uncertified_no_executable": (
            candidate["no_executable_without_certificate_count"] == 0
        ),
        "v243_all_abstains_certified": (
            candidate["abstain_has_exhaustion_evidence_rate"] >= 1.0
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3-paired-shadow-b-gate-v1",
        "inference_calls_made_by_gate": 0,
        "baseline_v241": baseline,
        "candidate_v243": candidate,
        "v243_delayed_evc_correlation": delayed_corr,
        "delta_v243_minus_v241": {
            field: candidate[field] - baseline[field]
            for field in (
                "candidate_presence_rate", "execution_plan_completion_rate",
                "graph_proof_completion_rate", "f1",
            )
        },
        "checks": checks,
        "passed": passed,
        "decision": "GO_V244_IMPLEMENTATION" if passed else "SAFE_STOP",
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v241-run", type=Path, required=True)
    parser.add_argument("--v243-run", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("configs/dynamic_v243_preregistration.json"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/dynamic_hypergraph_v243_qwen_shadow20.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.v241_run, args.v243_run, args.preregistration, args.config,
    )
    write_json(args.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "failed_checks": report["failed_checks"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
