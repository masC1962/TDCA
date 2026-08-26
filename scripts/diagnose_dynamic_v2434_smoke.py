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
    proof_gap_allocations = {}
    for qid, payload in graph_payloads.items():
        retrievals = list(payload.get("retrieval_attempt_history", []))
        allocations = list(payload.get("allocation_history", []))
        outcomes = {
            str(row.get("allocation_id", "")): row
            for row in payload.get("operation_outcome_history", [])
        }
        rows = []
        for allocation_index, allocation in enumerate(allocations):
            raw = allocation.get("evc_components_raw") or {}
            if not (
                float(raw.get("proof_gap_reducibility", 0.0)) > 0.0
                or float(raw.get("feasibility_unlock", 0.0)) > 0.0
            ):
                continue
            operation_id = str(allocation.get("operation_id", ""))
            retrieval = next((
                row for row in retrievals
                if str(row.get("operation_id", "")) == operation_id
                or str(row.get("operation_id", "")).startswith(
                    f"{operation_id}_allocation_"
                )
            ), None)
            rows.append({
                "allocation_id": str(allocation.get("allocation_id", "")),
                "operation_id": operation_id,
                "operation_family": str(allocation.get("operation_family", "")),
                "outcome": outcomes.get(
                    str(allocation.get("allocation_id", "")), {}
                ),
                "target_obligation_ids": list(
                    allocation.get("target_obligation_ids") or []
                ),
                "predicted_evc": float(allocation.get("predicted_evc", 0.0)),
                "predicted_delayed_proof_return": float(
                    allocation.get("predicted_delayed_proof_return", 0.0)
                ),
                "delayed_realized_proof_return": float(
                    allocation.get("delayed_realized_proof_return", 0.0)
                ),
                "actual_closed_target_ids": list(
                    allocation.get("actual_closed_target_ids") or []
                ),
                "retrieval": retrieval,
                "next_allocations": [{
                    "allocation_id": str(value.get("allocation_id", "")),
                    "operation_id": str(value.get("operation_id", "")),
                    "predicted_evc": float(value.get("predicted_evc", 0.0)),
                    "actual_utility": float(value.get("actual_utility", 0.0)),
                } for value in allocations[allocation_index + 1:allocation_index + 5]],
            })
        proof_gap_allocations[qid] = rows
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
            qid: [
                row["retrieval"] for row in proof_gap_allocations[qid]
                if row["retrieval"] is not None
            ]
            for qid in sorted(graph_payloads)
        },
        "proof_gap_allocations": proof_gap_allocations,
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
        "proof_gap_allocations": {
            qid: rows for qid, rows in report["proof_gap_allocations"].items()
            if rows
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
