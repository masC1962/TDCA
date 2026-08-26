#!/usr/bin/env python3
"""Paired v2.4.1 versus v2.4.3.1 Shadow-B gate; makes no provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_dynamic_v243_shadow import evaluate as evaluate_v243_shadow
from scripts.evaluate_dynamic_v2431_smoke import evaluate as evaluate_v2431_native
from tdca_research.utils import write_json


def evaluate(v241_run: Path, candidate_run: Path, prereg: Path, config: Path) -> dict:
    paired = evaluate_v243_shadow(v241_run, candidate_run, prereg, config)
    native = evaluate_v2431_native(candidate_run, prereg, config)
    checks = dict(paired["checks"])
    for name in (
        "exact_requested_provider_call_accounting",
        "zero_unjustified_high_fidelity",
        "complete_operation_target_audit",
        "complete_actual_obligation_closure_trace",
    ):
        checks[f"v2431_{name}"] = bool(native["checks"][name])
    passed = all(checks.values())
    paired.update({
        "schema_version": "dynamic-hypergraph-v2.4.3.1-paired-shadow-b-gate-v1",
        "candidate_v2431": paired.pop("candidate_v243"),
        "v2431_native_trace_evidence": {
            key: native["evidence"][key] for key in (
                "predicted_verification_calls_match_requested_samples_rate",
                "unjustified_high_fidelity_count", "operation_target_audit_rate",
                "actual_obligation_closure_trace_rate",
            )
        },
        "checks": checks,
        "passed": passed,
        "decision": "FREEZE_ALGORITHM_AND_BEGIN_PAPER_EXPERIMENTS" if passed else "SAFE_STOP",
        "failed_checks": [key for key, value in checks.items() if not value],
    })
    return paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v241-run", type=Path, required=True)
    parser.add_argument("--v2431-run", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("configs/dynamic_v2431_preregistration.json"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/dynamic_hypergraph_v2431_qwen_shadow20.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.v241_run, args.v2431_run, args.preregistration, args.config,
    )
    write_json(args.output, report)
    print(json.dumps({
        "decision": report["decision"], "failed_checks": report["failed_checks"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
