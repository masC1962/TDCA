#!/usr/bin/env python3
"""Posthoc v2.4.3 safe-stop diagnosis; never used by inference policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.analyze_dynamic_v242_safe_stop import analyze as paired_analyze
from tdca_research.utils import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_qid(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row.get("qid", ""))].append(row)
    return result


def _per_example(run: Path, name: str) -> dict[str, dict[str, Any]]:
    return {str(row["qid"]): row for row in _jsonl(run / name)}


def _mean(rows: list[float]) -> float:
    return mean(rows) if rows else 0.0


def _budget_distribution(run: Path) -> dict[str, Any]:
    rows = [
        row.get("allocation") or {}
        for row in _jsonl(run / "reasoning_traces.jsonl")
        if row.get("event") == "allocation_reconciled"
    ]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("operation_family", ""))].append(
            row.get("requested_budget") or {}
        )
    return {
        family: {
            "count": len(part),
            "mean_max_tokens": _mean([float(row.get("max_tokens", 0)) for row in part]),
            "mean_verification_samples": _mean([
                float(row.get("verification_samples", 0)) for row in part
            ]),
            "high_fidelity_verification_count": sum(
                int(row.get("verification_samples", 0)) >= 2 for row in part
            ),
        }
        for family, part in sorted(by_family.items())
    }


def analyze(baseline: Path, candidate: Path, gate_path: Path) -> dict[str, Any]:
    paired = paired_analyze(baseline, candidate, gate_path)
    gate = _json(gate_path)
    base_dynamic = _per_example(baseline, "dynamic_v2_per_example_metrics.jsonl")
    current_dynamic = _per_example(candidate, "dynamic_v2_per_example_metrics.jsonl")
    traces = _by_qid(_jsonl(candidate / "reasoning_traces.jsonl"))
    graphs = {
        str(row["qid"]): row.get("graph", row)
        for row in _jsonl(candidate / "dynamic_v2_graphs.jsonl")
    }
    candidate_presence = {
        "gained": sorted(
            qid for qid in current_dynamic
            if current_dynamic[qid].get("candidate_presence")
            and not base_dynamic.get(qid, {}).get("candidate_presence")
        ),
        "lost": sorted(
            qid for qid in current_dynamic
            if not current_dynamic[qid].get("candidate_presence")
            and base_dynamic.get(qid, {}).get("candidate_presence")
        ),
    }
    termination_cases = []
    obligation_rows: list[dict[str, Any]] = []
    high_value_failures: list[dict[str, Any]] = []
    family_counts = Counter()
    for qid, qid_traces in traces.items():
        graph = graphs[qid]
        obligations = graph.get("proof_obligations", {})
        finalized_allocations = {
            str(row.get("allocation_id", "")): row
            for row in graph.get("allocation_history", [])
        }
        termination = (graph.get("termination_history") or [{}])[-1]
        if termination.get("outcome") != "ANSWER":
            termination_cases.append({
                "qid": qid,
                "outcome": termination.get("outcome"),
                "reason": termination.get("reason"),
                "remaining_budget": termination.get("remaining_budget"),
                "open_obligations": [
                    row.get("obligation_type") for row in obligations.values()
                    if row.get("status") == "OPEN"
                ],
                "blocked_obligations": [
                    row.get("obligation_type") for row in obligations.values()
                    if row.get("status") == "BLOCKED"
                ],
            })
        for event in qid_traces:
            if event.get("event") != "allocation_reconciled":
                continue
            allocation = event.get("allocation") or {}
            finalized = finalized_allocations.get(
                str(allocation.get("allocation_id", "")), {}
            )
            family = str(allocation.get("operation_family", ""))
            family_counts[family] += 1
            predicted = float(allocation.get("predicted_delayed_proof_return", 0.0))
            actual = float(finalized.get(
                "delayed_realized_proof_return",
                event.get("delayed_realized_proof_return", 0.0),
            ))
            target_ids = [str(value) for value in allocation.get("target_obligation_ids", [])]
            target_types = sorted({
                str(obligations[value].get("obligation_type", "unknown"))
                for value in target_ids if value in obligations
            }) or ["no_explicit_obligation"]
            for obligation_type in target_types:
                obligation_rows.append({
                    "qid": qid,
                    "family": family,
                    "obligation_type": obligation_type,
                    "predicted": predicted,
                    "actual": actual,
                    "progressed": bool(event.get("progressed")),
                })
            if predicted >= 0.70 and actual <= 0.10:
                high_value_failures.append({
                    "qid": qid,
                    "operation_id": allocation.get("operation_id"),
                    "family": family,
                    "target_obligation_types": target_types,
                    "predicted_delayed": predicted,
                    "actual_delayed": actual,
                    "predicted_cost": allocation.get("predicted_normalized_cost"),
                    "progressed": bool(event.get("progressed")),
                    "failure_reason": event.get("failure_reason"),
                })
    by_type: dict[str, dict[str, Any]] = {}
    for obligation_type in sorted({row["obligation_type"] for row in obligation_rows}):
        part = [row for row in obligation_rows if row["obligation_type"] == obligation_type]
        by_type[obligation_type] = {
            "count": len(part),
            "mean_predicted_delayed": _mean([row["predicted"] for row in part]),
            "mean_actual_delayed": _mean([row["actual"] for row in part]),
            "progress_rate": _mean([float(row["progressed"]) for row in part]),
            "overprediction_gap": _mean([
                row["predicted"] - row["actual"] for row in part
            ]),
        }
    base_cost = _json(baseline / "cost_summary.json")
    current_cost = _json(candidate / "cost_summary.json")
    baseline_family_counts = Counter(
        str((row.get("allocation") or {}).get("operation_family", ""))
        for row in _jsonl(baseline / "reasoning_traces.jsonl")
        if row.get("event") == "allocation_reconciled"
    )
    family_keys = sorted(set(baseline_family_counts) | set(family_counts))
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3-safe-stop-diagnostic-v1",
        "inference_calls_made": 0,
        "gold_used_only_for_posthoc_paired_quality_labels": True,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "gate": gate,
        "paired_chain_transitions": paired["transitions"],
        "paired_chain_changes": paired["paired_changes"],
        "candidate_presence_transitions": candidate_presence,
        "termination_cases": sorted(termination_cases, key=lambda row: row["qid"]),
        "allocation_family_counts": {
            "v242": dict(sorted(baseline_family_counts.items())),
            "v243": dict(sorted(family_counts.items())),
            "delta_v243_minus_v242": {
                key: family_counts[key] - baseline_family_counts[key]
                for key in family_keys
            },
        },
        "selected_budget_packet_distribution": {
            "v242": _budget_distribution(baseline),
            "v243": _budget_distribution(candidate),
        },
        "obligation_target_calibration": by_type,
        "high_predicted_delayed_low_realized_count": len(high_value_failures),
        "high_predicted_delayed_low_realized_examples": high_value_failures,
        "compute_delta_v243_minus_v242": {
            "logical_llm_calls": int(current_cost["llm_calls"]) - int(base_cost["llm_calls"]),
            "logical_tokens": (
                int(current_cost["prompt_tokens"]) + int(current_cost["completion_tokens"])
                - int(base_cost["prompt_tokens"]) - int(base_cost["completion_tokens"])
            ),
            "retrieval_calls": int(current_cost["retrieval_calls"]) - int(base_cost["retrieval_calls"]),
        },
        "primary_diagnosis": [
            "Absolute cost eliminated viable-op cost clipping but underpriced repeated computation, causing five budget-exhausted questions.",
            "Proof-obligation severity measures need, not tractable closure probability; treating severity as return reverses delayed-value ranking.",
            "Terminal reachability is mostly a topology constant and cannot distinguish useful from redundant operations inside a reachable region.",
            "MERGE, COMMIT, VERIFY, and EXTRACT require operation-conditioned feasibility and marginal-closure estimates, not only graph deficit magnitude.",
            "Infrastructure, provenance, controller ownership, unsupported-answer, certified-stop, and graph-proof gates passed and should remain frozen.",
        ],
        "next_change_boundary": [
            "Do not enter v2.4.4 diffusion while v2.4.3 allocation is quality-unsafe.",
            "Separate obligation importance from operation-conditioned closure probability and redundancy.",
            "Add a remaining-horizon/opportunity-cost term so absolute resource fractions cannot make nearly every operation positive-EVC.",
            "Use frozen v2.4.3 traces for zero-API counterfactual calibration before another provider run.",
            "Retain absolute ready-set invariance, proof-obligation ledger, and certified meta-stop unchanged.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.baseline, args.candidate, args.gate)
    write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "paired_chain_transitions": report["paired_chain_transitions"],
        "candidate_presence_transitions": report["candidate_presence_transitions"],
        "high_value_failure_count": report[
            "high_predicted_delayed_low_realized_count"
        ],
        "compute_delta": report["compute_delta_v243_minus_v242"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
