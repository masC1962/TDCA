#!/usr/bin/env python3
"""Paired, offline v2.4 versus v2.4.1 shadow-B safety gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.verify_artifact import verify
from tdca_research.utils import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summary(run: Path) -> dict[str, Any]:
    artifact = verify(run, expected_count=20)
    metrics = _json(run / "metrics.json")
    dynamic = _json(run / "dynamic_v2_metrics.json")
    failures = (run / "failures.jsonl").read_text(encoding="utf-8")
    reasoning = _jsonl(run / "reasoning_traces.jsonl")
    return {
        "run": str(run),
        "artifact_verified": bool(artifact["verified"]),
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
        "no_diff_editor_allocation_count": sum(
            row.get("event") == "allocation_reconciled"
            and str((row.get("allocation") or {}).get("operation_family", "")).startswith("expand:")
            and not bool(row.get("progressed"))
            for row in reasoning
        ),
        "candidate_presence_rate": float(dynamic["candidate_presence_rate"]),
        "execution_plan_completion_rate": float(metrics.get(
            "execution_plan_completion_rate", metrics["full_chain_completion_rate"],
        )),
        "graph_proof_completion_rate": float(dynamic.get("graph_proof_completion_rate", 0.0)),
        "f1": float(metrics["f1"]),
    }


def evaluate(v24_run: Path, v241_run: Path, preregistration: Path) -> dict[str, Any]:
    prereg = _json(preregistration)
    limits = prereg["paired_shadow_b20_hard_gates"]
    baseline = _summary(v24_run)
    candidate = _summary(v241_run)
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
        "v241_candidate_presence_non_regression": candidate["candidate_presence_rate"] >= baseline["candidate_presence_rate"],
        "v241_execution_plan_completion_non_regression": candidate["execution_plan_completion_rate"] >= baseline["execution_plan_completion_rate"],
        "v241_graph_proof_completion_non_regression": candidate["graph_proof_completion_rate"] >= baseline["graph_proof_completion_rate"],
        "v241_f1_non_regression": candidate["f1"] >= baseline["f1"],
        "v241_zero_no_diff_editor_allocation": candidate["no_diff_editor_allocation_count"] <= limits["v241_no_diff_editor_allocation_count_max"],
    }
    passed = all(checks.values())
    return {
        "schema_version": "dynamic-hypergraph-v2.4.1-paired-shadow-b-gate-v1",
        "inference_calls_made_by_gate": 0,
        "baseline_v24": baseline,
        "candidate_v241": candidate,
        "delta_v241_minus_v24": {
            field: candidate[field] - baseline[field]
            for field in (
                "candidate_presence_rate", "execution_plan_completion_rate",
                "graph_proof_completion_rate", "f1",
            )
        },
        "checks": checks,
        "passed": passed,
        "decision": "GO_SMOKE_CONTROLS" if passed else "SAFE_STOP",
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v24-run", type=Path, required=True)
    parser.add_argument("--v241-run", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("configs/dynamic_v241_preregistration.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.v24_run, args.v241_run, args.preregistration)
    write_json(args.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "failed_checks": report["failed_checks"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
