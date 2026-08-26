#!/usr/bin/env python3
"""Evaluate the preregistered v2.4.3.2 Smoke-A gate without inference calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_dynamic_v2431_smoke import evaluate as evaluate_v2431
from tdca_research.utils import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(run: Path, preregistration: Path, config_path: Path) -> dict[str, Any]:
    report = evaluate_v2431(run, preregistration, config_path)
    limits = _json(preregistration)["adaptive_smoke_a20_hard_gates"]
    graph_rows = _jsonl(run / "dynamic_v2_graphs.jsonl")
    allocations = [
        allocation for example in graph_rows
        for allocation in example["graph"].get("allocation_history", [])
    ]
    by_id = {row["allocation_id"]: row for row in allocations}
    transition_trace = [
        isinstance(row.get("transition_certificate"), dict)
        and row["transition_certificate"].get("certificate_version")
        == "certified-transition-option-v2.4.3.2"
        and "predicted_transition_value" in row
        and "actual_transition_value" in row
        and "transition_realized" in row
        for row in allocations
    ]
    mandatory = [
        row for row in allocations
        if (row.get("transition_certificate") or {}).get("mandatory") is True
    ]
    selected_transition_events = [
        event for event in _jsonl(run / "reasoning_traces.jsonl")
        if event.get("event") == "meta_decision"
        and event.get("reason") == "certified_state_transition"
    ]
    invalid_bypass = []
    for event in selected_transition_events:
        row = by_id.get(str(event.get("selected_allocation_id", "")))
        certificate = (row or {}).get("transition_certificate") or {}
        if (
            row is None
            or not certificate.get("mandatory")
            or not certificate.get("deterministic")
            or int(certificate.get("provider_calls", -1)) != 0
            or int((row.get("requested_budget") or {}).get("llm_calls", -1)) != 0
        ):
            invalid_bypass.append(event)
    evidence = report["evidence"]
    evidence.update({
        "complete_transition_trace_rate": (
            sum(transition_trace) / len(transition_trace) if transition_trace else 0.0
        ),
        "certified_transition_allocation_count": len(mandatory),
        "certified_transition_realization_rate": (
            sum(bool(row.get("transition_realized")) for row in mandatory)
            / len(mandatory) if mandatory else 1.0
        ),
        "certified_transition_decision_count": len(selected_transition_events),
        "invalid_transition_bypass_count": len(invalid_bypass),
        "mean_predicted_transition_value": (
            sum(float(row.get("predicted_transition_value", 0.0)) for row in mandatory)
            / len(mandatory) if mandatory else 0.0
        ),
        "mean_actual_transition_value": (
            sum(float(row.get("actual_transition_value", 0.0)) for row in mandatory)
            / len(mandatory) if mandatory else 0.0
        ),
    })
    checks = report["checks"]
    checks.update({
        "complete_transition_trace": (
            evidence["complete_transition_trace_rate"]
            >= limits["complete_transition_trace_rate_min"]
        ),
        "certified_transitions_realized": (
            evidence["certified_transition_realization_rate"]
            >= limits["certified_transition_realization_rate_min"]
        ),
        "zero_invalid_transition_bypass": (
            evidence["invalid_transition_bypass_count"]
            <= limits["invalid_transition_bypass_count_max"]
        ),
    })
    report.update({
        "schema_version": "dynamic-hypergraph-v2.4.3.2-adaptive-smoke-a-gate-v1",
        "passed": all(checks.values()),
    })
    report["decision"] = "GO_PAIRED_SHADOW_B" if report["passed"] else "ITERATE"
    report["failed_checks"] = [key for key, value in checks.items() if not value]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("configs/dynamic_v2432_preregistration.json"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/dynamic_hypergraph_v2432_qwen_smoke20.yaml"),
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
