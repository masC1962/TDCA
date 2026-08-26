#!/usr/bin/env python3
"""Posthoc v2.4.3.1 SAFE_STOP diagnosis; never imported by inference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from tdca_research.utils import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return mean(rows) if rows else 0.0


def _per_qid(run: Path, filename: str) -> dict[str, dict[str, Any]]:
    return {str(row["qid"]): row for row in _jsonl(run / filename)}


def _trace_by_qid(run: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(run / "reasoning_traces.jsonl"):
        result[str(row.get("qid", ""))].append(row)
    return result


def analyze(baseline: Path, candidate: Path, gate_path: Path) -> dict[str, Any]:
    gate = _json(gate_path)
    baseline_metrics = _per_qid(baseline, "dynamic_v2_per_example_metrics.jsonl")
    candidate_metrics = _per_qid(candidate, "dynamic_v2_per_example_metrics.jsonl")
    traces = _trace_by_qid(candidate)
    graphs = {
        str(row["qid"]): row["graph"]
        for row in _jsonl(candidate / "dynamic_v2_graphs.jsonl")
    }
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    termination_rows = []
    stop_classes = Counter()
    family_counts = Counter()
    fidelity_counts = Counter()
    for qid, graph in graphs.items():
        qid_traces = traces[qid]
        packets = {
            str((event.get("allocation") or {}).get("allocation_id", "")):
            event.get("allocation") or {}
            for event in qid_traces if event.get("event") == "allocation_reconciled"
        }
        for allocation in graph.get("allocation_history", []):
            packet = packets.get(str(allocation.get("allocation_id", "")), {})
            family = str(packet.get("operation_family", "unknown"))
            family_counts[family] += 1
            fidelity_counts[(family, str(allocation.get("fidelity_level", "")))] += 1
            raw = allocation.get("evc_components_raw", {})
            family_rows[family].append({
                "importance": raw.get("obligation_importance", 0.0),
                "closure_probability": raw.get("operation_closure_probability", 0.0),
                "expected_delta": raw.get("expected_obligation_delta", 0.0),
                "terminal_return": raw.get("obligation_terminal_return", 0.0),
                "redundancy": raw.get("operation_redundancy", 0.0),
                "predicted_delayed": allocation.get("predicted_delayed_proof_return", 0.0),
                "predicted_immediate": allocation.get("predicted_immediate_utility", 0.0),
                "predicted_cost": allocation.get("predicted_normalized_cost", 0.0),
                "marginal_evc": allocation.get("predicted_marginal_evc", 0.0),
                "actual_closure": allocation.get("actual_target_closure_rate", 0.0),
                "actual_delayed": allocation.get("delayed_realized_proof_return", 0.0),
                "verification_samples": (
                    allocation.get("requested_budget", {}).get("verification_samples", 0)
                ),
            })
        termination = (graph.get("termination_history") or [{}])[-1]
        if termination.get("outcome") == "ANSWER":
            continue
        meta = [row for row in qid_traces if row.get("event") == "meta_decision"]
        candidates = (meta[-1].get("allocation_candidates", []) if meta else [])
        top = candidates[0] if candidates else {}
        certificate = termination.get("dead_end_certificate") or {}
        certificate_candidates = certificate.get("candidate_operations", [])
        top_kind = str(top.get("operation_family", ""))
        if not top_kind and certificate_candidates:
            top_kind = str(certificate_candidates[0].get("operation_type", "")).lower()
        remaining = termination.get("remaining_budget", {})
        substantial = (
            int(remaining.get("llm_calls", 0)) >= 4
            and int(remaining.get("tokens", 0)) >= 4000
        )
        if top_kind.startswith("commit") or top_kind == "commit":
            stop_class = "certified_commit_below_threshold"
        elif top_kind.startswith("retrieve") and int(termination.get("step", 0)) <= 5:
            stop_class = "early_retrieve_below_threshold"
        elif termination.get("reason") == "no_executable_computation_with_certificate":
            stop_class = "no_executable_with_certificate"
        else:
            stop_class = "other_below_threshold"
        stop_classes[stop_class] += 1
        termination_rows.append({
            "qid": qid,
            "step": termination.get("step"),
            "reason": termination.get("reason"),
            "remaining_budget": remaining,
            "substantial_budget_remaining": substantial,
            "top_operation_family": top_kind,
            "top_predicted_evc": top.get(
                "predicted_evc", termination.get("best_predicted_evc", 0.0)
            ),
            "top_predicted_immediate": top.get("predicted_immediate_utility", 0.0),
            "top_predicted_delayed": top.get("predicted_delayed_proof_return", 0.0),
            "top_predicted_cost": top.get("predicted_normalized_cost", 0.0),
            "open_obligation_count": len(certificate.get("open_obligations", [])),
            "stop_class": stop_class,
        })
    family_calibration = {
        family: {
            "count": len(rows),
            **{
                f"mean_{name}": _mean(row[name] for row in rows)
                for name in rows[0]
            },
        }
        for family, rows in sorted(family_rows.items()) if rows
    }
    paired = {
        "candidate_presence_gained": sorted(
            qid for qid, row in candidate_metrics.items()
            if row.get("candidate_presence")
            and not baseline_metrics.get(qid, {}).get("candidate_presence")
        ),
        "candidate_presence_lost": sorted(
            qid for qid, row in candidate_metrics.items()
            if not row.get("candidate_presence")
            and baseline_metrics.get(qid, {}).get("candidate_presence")
        ),
        "graph_proof_gained": sorted(
            qid for qid, row in candidate_metrics.items()
            if row.get("graph_proof_completion")
            and not baseline_metrics.get(qid, {}).get("graph_proof_completion")
        ),
        "graph_proof_lost": sorted(
            qid for qid, row in candidate_metrics.items()
            if not row.get("graph_proof_completion")
            and baseline_metrics.get(qid, {}).get("graph_proof_completion")
        ),
    }
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3.1-safe-stop-diagnostic-v1",
        "inference_calls_made": 0,
        "gold_used_only_for_posthoc_paired_quality_labels": True,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "gate_decision": gate["decision"],
        "failed_checks": gate["failed_checks"],
        "paired_quality_transitions": paired,
        "termination_count": len(termination_rows),
        "termination_reason_counts": dict(Counter(
            row["reason"] for row in termination_rows
        )),
        "stop_class_counts": dict(stop_classes),
        "substantial_budget_abstain_count": sum(
            row["substantial_budget_remaining"] for row in termination_rows
        ),
        "termination_cases": sorted(termination_rows, key=lambda row: row["qid"]),
        "allocation_family_counts": dict(sorted(family_counts.items())),
        "fidelity_counts": {
            f"{family}|{fidelity}": count
            for (family, fidelity), count in sorted(fidelity_counts.items())
        },
        "family_component_calibration": family_calibration,
        "primary_diagnosis": [
            "v2.4.3.1 fixed sample-cost accounting and eliminated VERIFY high-fidelity collapse, but over-pruned the reasoning horizon.",
            "Deterministic commit:default transitions have no target proof obligation, so their delayed value is forced to zero even though they unlock downstream subgoals.",
            "The global 0.40 immediate / 0.60 delayed mixture makes a certified zero-provider COMMIT fall below the generic meta-stop threshold; this is a semantic category error, not evidence that the commit is worthless.",
            "Multiplicative closure value also discounts early retrieval by current regional obligation mass and terminal distance before the downstream obligation exists, causing early abstention with substantial budget remaining.",
            "Actual operation progress is high for several families while predicted correlations remain negative; mutation progress alone must not become the target, but executable state-transition value is missing from the model.",
        ],
        "next_change_boundary": [
            "Do not run Shadow-B or signed diffusion.",
            "Preserve strict targeting, exact fidelity accounting, controller closure traces, safety and termination typing.",
            "Model certified deterministic state transitions separately from proof-obligation closure; never subject a zero-provider mandatory COMMIT to the generic delayed-value threshold.",
            "Replace single-step obligation-mass dilution with a bounded option/continuation value for operations that deterministically expose the next obligation.",
            "Validate the revised continuation semantics on frozen v2.4.3.1 traces and synthetic graph fixtures before any new API run.",
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
        "stop_class_counts": report["stop_class_counts"],
        "substantial_budget_abstain_count": report["substantial_budget_abstain_count"],
        "paired_quality_transitions": report["paired_quality_transitions"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
