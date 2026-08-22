#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qid", required=True)
    args = parser.parse_args()
    output = []
    for row in rows(args.run / "reasoning_traces.jsonl"):
        if str(row.get("qid")) != args.qid:
            continue
        event = str(row.get("event", ""))
        if event == "meta_decision":
            candidates = row.get("allocation_candidates", [])
            output.append({
                "event": event, "step": row.get("step"), "outcome": row.get("outcome"),
                "reason": row.get("reason"), "best_evc": row.get("best_predicted_evc"),
                "candidates": [{
                    "operation_id": value.get("operation_id"),
                    "family": value.get("operation_family"),
                    "evc": value.get("predicted_evc"),
                    "raw": value.get("evc_components_raw"),
                    "normalized": value.get("evc_components_normalized"),
                    "prior": value.get("feedback_prior"),
                    "budget": value.get("requested_budget"),
                } for value in candidates[:4]],
            })
        elif event == "allocation_reconciled":
            allocation = row.get("allocation", {})
            output.append({
                "event": event,
                "operation_id": allocation.get("operation_id"),
                "family": allocation.get("operation_family"),
                "progressed": row.get("progressed"),
                "actual_utility": row.get("actual_utility"),
                "raw_utility": row.get("actual_utility_components_raw"),
                "statistics_after": row.get("statistics_after"),
                "failure_reason": row.get("failure_reason"),
            })
        elif event in {"graph_operation", "join_rejected", "typed_extraction_rejected", "termination"}:
            output.append({
                key: row.get(key) for key in (
                    "event", "step_id", "operation", "operation_id", "target_id",
                    "join_signature", "diagnostics", "outcome", "reason",
                ) if key in row
            })
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
