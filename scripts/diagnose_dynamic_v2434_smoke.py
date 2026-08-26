#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_dynamic_v2432_safe_stop import analyze as analyze_base
from tdca_research.utils import write_json


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze(run: Path) -> dict:
    report = analyze_base(run)
    predictions = {
        str(row["qid"]): row for row in _jsonl(run / "predictions.jsonl")
    }
    quality = {
        str(row["qid"]): row for row in _jsonl(run / "per_example_metrics.jsonl")
    }
    structural = {
        str(row["qid"]): row
        for row in _jsonl(run / "dynamic_v2_per_example_metrics.jsonl")
    }
    graph_payloads = {
        str(row["qid"]): row["graph"]
        for row in _jsonl(run / "dynamic_v2_graphs.jsonl")
    }
    failed_transitions = []
    terminal_certificates = []
    for example in _jsonl(run / "dynamic_v2_graphs.jsonl"):
        qid = str(example["qid"])
        for row in example["graph"].get("allocation_history", []):
            certificate = row.get("transition_certificate") or {}
            if not certificate.get("mandatory"):
                continue
            summary = {
                "qid": qid,
                "allocation_id": str(row.get("allocation_id", "")),
                "operation_id": str(row.get("operation_id", "")),
                "certificate_version": str(certificate.get("certificate_version", "")),
                "kind": str(certificate.get("kind", "")),
                "predicted_transition_value": float(
                    row.get("predicted_transition_value", 0.0)
                ),
                "actual_transition_value": float(row.get("actual_transition_value", 0.0)),
                "transition_realized": bool(row.get("transition_realized", False)),
                "transition_observations": row.get("transition_observations", {}),
            }
            if summary["kind"] == "accepted_terminal_materialization":
                terminal_certificates.append(summary)
            if not summary["transition_realized"]:
                failed_transitions.append(summary)

    blocked_after_accepted_readout = []
    by_qid: dict[str, list[dict]] = {}
    for row in _jsonl(run / "reasoning_traces.jsonl"):
        by_qid.setdefault(str(row.get("qid", "")), []).append(row)
    for qid, events in by_qid.items():
        for index, event in enumerate(events):
            if (
                event.get("event") != "terminal_belief_readout"
                or not event.get("accepted_answer_node_ids")
            ):
                continue
            meta = next((
                row for row in events[index + 1:]
                if row.get("event") == "meta_decision"
            ), {})
            if meta.get("outcome") != "CONTINUE":
                blocked_after_accepted_readout.append({
                    "qid": qid,
                    "step": event.get("step"),
                    "answer_node_ids": event.get("accepted_answer_node_ids"),
                    "meta_outcome": meta.get("outcome"),
                    "meta_reason": meta.get("reason"),
                    "selected_allocation_id": meta.get("selected_allocation_id"),
                })

    report.update({
        "schema_version": "dynamic-v2.4.3.4-smoke-diagnostic-v1",
        "certified_terminal_materialization_count": len(terminal_certificates),
        "certified_terminal_materializations": terminal_certificates,
        "failed_certified_transition_count": len(failed_transitions),
        "failed_certified_transitions": failed_transitions,
        "blocked_after_accepted_terminal_readout_count": len(
            blocked_after_accepted_readout
        ),
        "blocked_after_accepted_terminal_readout": blocked_after_accepted_readout,
        "per_example_outcomes": [{
            "qid": qid,
            "question": str(predictions[qid].get("question", "")),
            "status": str(predictions[qid].get("status", "")),
            "answer": predictions[qid].get("answer"),
            "f1": float(quality[qid].get("f1", 0.0)),
            "candidate_presence": bool(structural[qid].get("candidate_presence")),
            "execution_plan_completion": bool(
                quality[qid].get("execution_plan_completion")
            ),
            "graph_proof_completion": bool(
                structural[qid].get("graph_proof_completion")
            ),
        } for qid in sorted(predictions)],
        "recovery_retrievals": {
            qid: [row for row in graph_payloads[qid].get(
                "retrieval_attempt_history", []
            ) if str(row.get("query", "")).lower().startswith(
                "independent source"
            )]
            for qid in sorted(graph_payloads)
        },
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run)
    write_json(args.output, report)
    print(json.dumps({
        "failed_certified_transitions": report["failed_certified_transitions"],
        "certified_terminal_materialization_count": report[
            "certified_terminal_materialization_count"
        ],
        "blocked_after_accepted_terminal_readout": report[
            "blocked_after_accepted_terminal_readout"
        ],
        "incorrect_answers": report["incorrect_answers"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
