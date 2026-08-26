#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from tdca_research.utils import write_json


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _corr(rows: list[tuple[float, float]]) -> float:
    if len(rows) < 2:
        return 0.0
    xs, ys = zip(*rows)
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in rows)
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def analyze(run: Path) -> dict[str, Any]:
    graphs = {str(row["qid"]): row["graph"] for row in _jsonl(run / "dynamic_v2_graphs.jsonl")}
    quality = {str(row["qid"]): row for row in _jsonl(run / "per_example_metrics.jsonl")}
    structural = {
        str(row["qid"]): row for row in _jsonl(run / "dynamic_v2_per_example_metrics.jsonl")
    }
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(run / "reasoning_traces.jsonl"):
        traces[str(row.get("qid", ""))].append(row)

    immediate: dict[str, list[tuple[float, float]]] = defaultdict(list)
    component_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for qid, graph in graphs.items():
        family_by_allocation = {
            str((event.get("allocation") or {}).get("allocation_id", "")):
            str((event.get("allocation") or {}).get("operation_family", "unknown"))
            for event in traces[qid] if event.get("event") == "allocation_reconciled"
        }
        for row in graph.get("allocation_history", []):
            actual_immediate = float(row.get("actual_immediate_utility", 0.0))
            immediate[family_by_allocation.get(str(row["allocation_id"]), "unknown")].append((
                float(row.get("predicted_immediate_utility", 0.0)),
                actual_immediate,
            ))
            for name, value in row.get("evc_components_raw", {}).items():
                component_pairs[str(name)].append((float(value), actual_immediate))
            raw = row.get("evc_components_raw", {})
            normalized = row.get("evc_components_normalized", {})
            closure_mass = (
                float(raw.get("operation_closure_probability", 0.0))
                * float(raw.get("expected_obligation_delta", 0.0))
            )
            component_pairs["closure_mass"].append((closure_mass, actual_immediate))
            one_step_components = [
                float(raw.get("evidence_novelty", 0.0)),
                float(raw.get("obligation_closure", 0.0)),
                closure_mass,
                float(raw.get("terminal_gap", 0.0)),
                float(raw.get("answer_impact", 0.0)),
            ]
            component_pairs["one_step_progress_equal"].append((
                sum(one_step_components) / len(one_step_components), actual_immediate,
            ))
            component_pairs["one_step_progress_max"].append((
                max(one_step_components), actual_immediate,
            ))
            normalized_closure_mass = (
                float(normalized.get("operation_closure_probability", 0.0))
                * float(normalized.get("expected_obligation_delta", 0.0))
            )
            normalized_one_step = [
                float(normalized.get("evidence_novelty", 0.0)),
                float(normalized.get("obligation_closure", 0.0)),
                normalized_closure_mass,
                float(normalized.get("terminal_gap", 0.0)),
                float(normalized.get("answer_impact", 0.0)),
            ]
            component_pairs["normalized_one_step_progress_equal"].append((
                sum(normalized_one_step) / len(normalized_one_step), actual_immediate,
            ))

    abstains = []
    for qid, graph in graphs.items():
        termination = (graph.get("termination_history") or [{}])[-1]
        if termination.get("outcome") == "ANSWER":
            continue
        nodes = graph.get("nodes", {})
        subgoals = []
        for node_id, node in nodes.items():
            if "dependencies" not in node or "question_template" not in node:
                continue
            subgoals.append({
                "node_id": node_id,
                "status": node.get("status"),
                "dependencies": node.get("dependencies", []),
                "terminal": node.get("terminal", False),
                "claim_status_counts": dict(Counter(
                    claim.get("status") for claim in nodes.values()
                    if claim.get("target_subgoal") == node_id and "evidence_refs" in claim
                )),
                "retrieval_attempts": sum(
                    row.get("target_subgoal") == node_id
                    for row in graph.get("retrieval_attempt_history", [])
                ),
            })
        open_obligations = [
            {
                key: row.get(key) for key in (
                    "target_subgoal", "branch_id", "obligation_type", "status",
                    "severity", "reason_codes", "required_node_ids",
                )
            }
            for row in graph.get("proof_obligations", {}).values()
            if row.get("status") in {"OPEN", "BLOCKED"}
        ]
        recent = []
        for event in traces[qid][-18:]:
            allocation = event.get("allocation") or {}
            recent.append({
                "event": event.get("event"),
                "step": event.get("step", event.get("step_id")),
                "operation_family": allocation.get("operation_family"),
                "failure_reason": event.get("failure_reason"),
                "outcome": event.get("outcome"),
                "reason": event.get("reason"),
            })
        abstains.append({
            "qid": qid,
            "hop_count": structural[qid].get("hop_count"),
            "candidate_presence": structural[qid].get("candidate_presence"),
            "graph_proof_completion": structural[qid].get("graph_proof_completion"),
            "execution_plan_completion": quality[qid].get("execution_plan_completion"),
            "termination": termination,
            "open_obligation_type_counts": dict(Counter(
                row["obligation_type"] for row in open_obligations
            )),
            "open_obligations": open_obligations,
            "subgoals": sorted(subgoals, key=lambda row: row["node_id"]),
            "active_branches": [
                {"branch_id": key, **value}
                for key, value in graph.get("branches", {}).items()
                if value.get("status") == "active"
            ],
            "recent_trace": recent,
        })

    incorrect_answers = [
        {
            "qid": qid,
            "answer": row.get("answer"),
            "confidence": row.get("confidence"),
            "exact_match": row.get("exact_match"),
            "f1": row.get("f1"),
            "candidate_presence": structural[qid].get("candidate_presence"),
            "graph_proof_completion": structural[qid].get("graph_proof_completion"),
        }
        for qid, row in quality.items()
        if row.get("status") == "answer" and float(row.get("f1", 0.0)) < 1.0
    ]
    return {
        "schema_version": "dynamic-v2.4.3.2-safe-stop-diagnostic-v1",
        "inference_calls_made": 0,
        "immediate_calibration_by_family": {
            family: {
                "count": len(rows),
                "pearson": _corr(rows),
                "mean_predicted": mean(x for x, _ in rows),
                "mean_actual": mean(y for _, y in rows),
            }
            for family, rows in sorted(immediate.items())
        },
        "actual_immediate_component_correlations": {
            name: _corr(rows) for name, rows in sorted(component_pairs.items())
        },
        "abstain_count": len(abstains),
        "abstains": sorted(abstains, key=lambda row: row["qid"]),
        "incorrect_answer_count": len(incorrect_answers),
        "incorrect_answers": sorted(incorrect_answers, key=lambda row: row["qid"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run)
    write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "abstain_count": report["abstain_count"],
        "abstain_summary": [{
            "qid": row["qid"],
            "candidate_presence": row["candidate_presence"],
            "graph_proof_completion": row["graph_proof_completion"],
            "reason": row["termination"].get("reason"),
            "open_obligation_type_counts": row["open_obligation_type_counts"],
            "subgoals": [{
                key: subgoal[key] for key in (
                    "node_id", "status", "claim_status_counts", "retrieval_attempts",
                )
            } for subgoal in row["subgoals"]],
            "active_branch_count": len(row["active_branches"]),
        } for row in report["abstains"]],
        "incorrect_answers": report["incorrect_answers"],
        "immediate_calibration_by_family": report["immediate_calibration_by_family"],
        "actual_immediate_component_correlations": report[
            "actual_immediate_component_correlations"
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
