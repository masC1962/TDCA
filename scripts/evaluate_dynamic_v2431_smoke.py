#!/usr/bin/env python3
"""Evaluate the preregistered v2.4.3.1 Smoke-A gate without inference calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_dynamic_v243_smoke import evaluate as evaluate_v243
from tdca_research.utils import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(run: Path, preregistration: Path, config_path: Path) -> dict[str, Any]:
    report = evaluate_v243(run, preregistration, config_path)
    limits = _json(preregistration)["adaptive_smoke_a20_hard_gates"]
    allocations = [
        allocation
        for example in _jsonl(run / "dynamic_v2_graphs.jsonl")
        for allocation in example["graph"].get("allocation_history", [])
    ]
    call_matches = [
        int(row.get("predicted_provider_calls", -1))
        == int((row.get("requested_budget") or {}).get("llm_calls", -2))
        for row in allocations
    ]
    high_rows = [row for row in allocations if row.get("fidelity_level") == "high"]
    unjustified_high = [
        row for row in high_rows
        if float(row.get("predicted_marginal_evc", 0.0)) <= 0.0
        or not bool(row.get("reserve_feasible", False))
    ]
    target_audits = [
        bool(isinstance(row.get("obligation_estimate"), dict))
        and sorted(row.get("target_obligation_ids", []))
        == sorted((row.get("obligation_estimate") or {}).get(
            "target_obligation_ids", []
        ))
        and all(
            value in row.get("target_obligation_ids", [])
            for value in row.get("actual_closed_target_ids", [])
        )
        for row in allocations
    ]
    closure_trace = [
        bool(row.get("pre_target_obligation_statuses"))
        and "actual_target_closure_rate" in row
        and "actual_obligation_delta" in row
        for row in allocations if row.get("target_obligation_ids")
    ]
    evidence = report["evidence"]
    evidence.update({
        "predicted_verification_calls_match_requested_samples_rate": (
            sum(call_matches) / len(call_matches) if call_matches else 0.0
        ),
        "unjustified_high_fidelity_count": len(unjustified_high),
        "high_fidelity_allocation_count": len(high_rows),
        "operation_target_audit_rate": (
            sum(target_audits) / len(target_audits) if target_audits else 0.0
        ),
        "actual_obligation_closure_trace_rate": (
            sum(closure_trace) / len(closure_trace) if closure_trace else 1.0
        ),
    })
    checks = report["checks"]
    checks.update({
        "exact_requested_provider_call_accounting": (
            evidence["predicted_verification_calls_match_requested_samples_rate"]
            >= limits["predicted_verification_calls_match_requested_samples_rate_min"]
        ),
        "zero_unjustified_high_fidelity": (
            evidence["unjustified_high_fidelity_count"]
            <= limits["unjustified_high_fidelity_count_max"]
        ),
        "complete_operation_target_audit": (
            evidence["operation_target_audit_rate"]
            >= limits["operation_target_audit_rate_min"]
        ),
        "complete_actual_obligation_closure_trace": (
            evidence["actual_obligation_closure_trace_rate"] >= 1.0
        ),
    })
    report.update({
        "schema_version": "dynamic-hypergraph-v2.4.3.1-adaptive-smoke-a-gate-v1",
        "passed": all(checks.values()),
    })
    report["decision"] = "GO_PAIRED_SHADOW_B" if report["passed"] else "SAFE_STOP"
    report["failed_checks"] = [key for key, value in checks.items() if not value]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("configs/dynamic_v2431_preregistration.json"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/dynamic_hypergraph_v2431_qwen_smoke20.yaml"),
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
