#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tdca_research.utils import write_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(run: Path) -> dict[str, Any]:
    graph_rows = _jsonl(run / "dynamic_v2_graphs.jsonl")
    trace_rows: dict[str, list[dict[str, Any]]] = {}
    for row in _jsonl(run / "reasoning_traces.jsonl"):
        trace_rows.setdefault(str(row["qid"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for graph_row in graph_rows:
        qid = str(graph_row["qid"])
        graph = graph_row["graph"]
        allocations = {
            str(row["allocation_id"]): row
            for row in graph.get("allocation_history", [])
        }
        allocation_by_operation = {
            str(row["operation_id"]): row
            for row in graph.get("allocation_history", [])
        }
        credits_by_allocation: dict[str, list[dict[str, Any]]] = {}
        for credit in graph.get("credit_assignment_history", []):
            credits_by_allocation.setdefault(
                str(credit["allocation_id"]), []
            ).append(credit)
        events = trace_rows.get(qid, [])
        for retrieval in graph.get("retrieval_attempt_history", []):
            target_ids = list(retrieval.get("recovery_target_obligation_ids") or [])
            if (
                retrieval.get("recovery_policy") != "proof_gap_recovery_v1"
                or not target_ids
            ):
                continue
            operation_id = str(retrieval.get("operation_id", ""))
            allocation = allocation_by_operation.get(operation_id)
            if allocation is None:
                allocation = next((
                    value for candidate_id, value in allocation_by_operation.items()
                    if operation_id.startswith(f"{candidate_id}_allocation_")
                ), None)
            allocation_id = str((allocation or {}).get("allocation_id", ""))
            credits = credits_by_allocation.get(allocation_id, [])
            final_credit = credits[-1] if credits else {}
            priority_events = [
                event for event in events
                if event.get("event") == "proof_recovery_extraction_priority"
                and event.get("retrieval_attempt_id") == retrieval.get("attempt_id")
            ]
            rows.append({
                "qid": qid,
                "retrieval_attempt_id": str(retrieval.get("attempt_id", "")),
                "allocation_id": allocation_id,
                "target_obligation_ids": target_ids,
                "new_evidence_count": int(retrieval.get("new_evidence_count", 0)),
                "priority_event_count": len(priority_events),
                "priority_events": priority_events,
                "predicted_delayed_proof_return": float(
                    (allocation or {}).get("predicted_delayed_proof_return", 0.0)
                ),
                "delayed_realized_proof_return": float(
                    (allocation or {}).get("delayed_realized_proof_return", 0.0)
                ),
                "actual_closed_target_ids": list(
                    (allocation or {}).get("actual_closed_target_ids") or []
                ),
                "seed_node_ids": list(final_credit.get("seed_node_ids") or []),
                "causal_descendant_ids": list(
                    final_credit.get("causal_descendant_ids") or []
                ),
                "causal_event_ids": list(final_credit.get("causal_event_ids") or []),
                "delayed_components_raw": dict(
                    final_credit.get("delayed_components_raw") or {}
                ),
                "credit_finalized": bool(
                    (allocation or {}).get("credit_finalized", False)
                ),
            })
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3.8-recovery-audit-v1",
        "source_run": str(run),
        "gold_used": False,
        "targeted_recovery_count": len(rows),
        "controller_provenance_complete_count": sum(
            bool(row["target_obligation_ids"]) for row in rows
        ),
        "freshness_priority_event_count": sum(
            row["priority_event_count"] for row in rows
        ),
        "positive_causal_return_count": sum(
            row["delayed_realized_proof_return"] > 0.0 for row in rows
        ),
        "end_to_end_recovery_count": sum(
            row["priority_event_count"] > 0
            and row["delayed_realized_proof_return"] > 0.0
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run)
    write_json(args.output, report)
    print(json.dumps({
        key: report[key] for key in (
            "targeted_recovery_count",
            "controller_provenance_complete_count",
            "freshness_priority_event_count",
            "positive_causal_return_count",
            "end_to_end_recovery_count",
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
