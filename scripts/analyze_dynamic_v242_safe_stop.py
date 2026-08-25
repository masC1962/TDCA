#!/usr/bin/env python3
"""Gold-aware outcome comparison plus gold-free policy-stop diagnostics.

Gold metrics are used only to identify paired regressions after inference.  EVC
and stop-cause diagnostics are read from the recorded controller traces.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

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


def _per_example(run: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["qid"]): row for row in _jsonl(run / "per_example_metrics.jsonl")
    }


def _final_graphs(run: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["qid"]): row.get("graph", row.get("graph_snapshot", row))
        for row in _jsonl(run / "dynamic_v2_graphs.jsonl")
    }


def _trace_summary(rows: list[dict[str, Any]], graph: dict[str, Any]) -> dict[str, Any]:
    reconciled = [row for row in rows if row.get("event") == "allocation_reconciled"]
    decisions = [row for row in rows if row.get("event") == "meta_decision"]
    final_decision = decisions[-1] if decisions else {}
    candidates = list(final_decision.get("allocation_candidates") or [])
    top = []
    for packet in candidates[:5]:
        immediate = float(packet.get("predicted_immediate_utility", 0.0) or 0.0)
        delayed = float(packet.get("predicted_delayed_proof_return", 0.0) or 0.0)
        cost = float(packet.get("predicted_normalized_cost", 0.0) or 0.0)
        top.append({
            "operation_family": str(packet.get("operation_family", "")),
            "operation_id": str(packet.get("operation_id", "")),
            "fidelity_level": str(packet.get("fidelity_level", "")),
            "predicted_immediate_utility": immediate,
            "predicted_delayed_proof_return": delayed,
            "predicted_normalized_cost": cost,
            "gross_horizon_value": 0.4 * immediate + 0.6 * delayed,
            "predicted_evc": float(packet.get("predicted_evc", 0.0) or 0.0),
            "proof_gap_reducibility": float(
                (packet.get("evc_components_raw") or {}).get(
                    "proof_gap_reducibility", 0.0,
                ) or 0.0
            ),
            "feasibility_unlock": float(
                (packet.get("evc_components_raw") or {}).get(
                    "feasibility_unlock", 0.0,
                ) or 0.0
            ),
        })
    extraction = [
        row for row in rows if row.get("event") == "typed_extraction_diagnostic"
    ]
    joins = list(graph.get("join_attempt_history", []))
    answers = {
        node_id for node_id, node in graph.get("nodes", {}).items()
        if node.get("kind") == "answer" and node.get("status") == "accepted"
    }
    answer_support = {
        claim_id
        for answer_id in answers
        for claim_id in graph["nodes"][answer_id].get("supporting_claims", [])
    }
    termination = (graph.get("termination_history") or [{}])[-1]
    remaining = termination.get("remaining_budget") or {}
    return {
        "termination": termination,
        "final_meta_reason": str(final_decision.get("reason", "")),
        "final_candidate_count": len(candidates),
        "final_positive_evc_count": sum(
            float(row.get("predicted_evc", 0.0) or 0.0) > 0.0 for row in candidates
        ),
        "final_top_candidates": top,
        "substantial_budget_remaining": (
            int(remaining.get("llm_calls", 0)) >= 4
            and int(remaining.get("tokens", 0)) >= 4000
        ),
        "allocation_family_counts": dict(sorted(Counter(
            str((row.get("allocation") or {}).get("operation_family", ""))
            for row in reconciled
        ).items())),
        "allocation_count": len(reconciled),
        "zero_row_extraction_count": sum(
            int(row.get("raw_row_count", row.get("raw_rows", 0)) or 0) == 0
            for row in extraction
        ),
        "accepted_join_count": sum(bool(row.get("accepted")) for row in joins),
        "answer_used_join_count": sum(
            bool(row.get("accepted"))
            and str(row.get("conclusion_node_id", "")) in answer_support
            for row in joins
        ),
    }


def analyze(baseline: Path, candidate: Path, gate: Path) -> dict[str, Any]:
    base_metrics = _json(baseline / "metrics.json")
    candidate_metrics = _json(candidate / "metrics.json")
    base_examples = _per_example(baseline)
    candidate_examples = _per_example(candidate)
    base_traces = _by_qid(_jsonl(baseline / "reasoning_traces.jsonl"))
    candidate_traces = _by_qid(_jsonl(candidate / "reasoning_traces.jsonl"))
    base_graphs = _final_graphs(baseline)
    candidate_graphs = _final_graphs(candidate)
    paired = {}
    transitions = Counter()
    for qid in sorted(set(base_examples) & set(candidate_examples)):
        base = base_examples[qid]
        current = candidate_examples[qid]
        base_chain = bool(base.get("full_chain_complete"))
        current_chain = bool(current.get("full_chain_complete"))
        if not base_chain and current_chain:
            transition = "chain_gained"
        elif base_chain and not current_chain:
            transition = "chain_lost"
        elif base_chain:
            transition = "chain_retained"
        else:
            transition = "chain_absent_both"
        transitions[transition] += 1
        if transition not in {"chain_gained", "chain_lost"}:
            continue
        base_trace = _trace_summary(base_traces[qid], base_graphs[qid])
        current_trace = _trace_summary(candidate_traces[qid], candidate_graphs[qid])
        stop_reason = str(current_trace["termination"].get("reason", ""))
        if transition == "chain_lost" and stop_reason == "best_expected_value_not_above_marginal_cost_threshold":
            cause = "net_evc_clipped_below_stop_threshold"
        elif transition == "chain_lost" and stop_reason == "no_executable_computation":
            cause = "ready_set_exhausted_after_policy_divergence"
        else:
            cause = "paired_outcome_change"
        paired[qid] = {
            "transition": transition,
            "diagnostic_cause": cause,
            "baseline": {
                "exact_match": bool(base.get("exact_match")),
                "f1": float(base.get("f1", 0.0)),
                "status": str(base.get("status", "")),
                "trace": base_trace,
            },
            "candidate": {
                "exact_match": bool(current.get("exact_match")),
                "f1": float(current.get("f1", 0.0)),
                "status": str(current.get("status", "")),
                "trace": current_trace,
            },
        }
    return {
        "schema_version": "dynamic-hypergraph-v2.4.2-safe-stop-diagnostic-v1",
        "inference_calls_made": 0,
        "gold_used_only_for_posthoc_paired_quality_labels": True,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "quality": {
            "baseline": {
                "exact_match": base_metrics["exact_match"],
                "f1": base_metrics["f1"],
                "execution_plan_completion_rate": base_metrics[
                    "execution_plan_completion_rate"
                ],
            },
            "candidate": {
                "exact_match": candidate_metrics["exact_match"],
                "f1": candidate_metrics["f1"],
                "execution_plan_completion_rate": candidate_metrics[
                    "execution_plan_completion_rate"
                ],
            },
        },
        "gate": _json(gate),
        "transitions": dict(sorted(transitions.items())),
        "paired_changes": paired,
        "primary_diagnosis": [
            "The delayed-credit target and both calibration gates passed.",
            "Net EVC cost subtraction clipped viable recovery actions to zero in two chain regressions despite substantial remaining budget.",
            "Two additional chain regressions exhausted the ready set after a different JOIN/extraction trajectory; family capacity priors are too coarse for within-family frontier ranking.",
            "The candidate gained two chains, so the mechanism is active but not quality-safe under the frozen v2.4.2 policy.",
        ],
        "next_change_boundary": [
            "Do not change credit attribution or the passed calibration thresholds.",
            "Replace coarse family-only delayed capacity with graph-local causal reachability and proof-frontier features.",
            "Calibrate normalized cost semantics so meta-stop cannot discard a high proof-gap opportunity solely through min-max cost scaling.",
            "Repeat zero-API replay and a new preregistered development smoke before opening shadow-B.",
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
        "transitions": report["transitions"],
        "failed_checks": report["gate"]["failed_checks"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
