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
    parser.add_argument("--final-proof", action="store_true")
    args = parser.parse_args()
    if args.final_proof:
        graph_row = next(
            row for row in rows(args.run / "dynamic_v2_graphs.jsonl")
            if str(row.get("qid")) == args.qid
        )
        graph = graph_row["graph"]
        claims = {
            node_id: {
                "subject": node.get("subject"), "relation": node.get("relation"),
                "value": node.get("value"), "target_subgoal": node.get("target_subgoal"),
                "status": node.get("status"),
                "dependencies": node.get("dependency_claim_ids", []),
                "support": (node.get("score") or {}).get("absolute_support"),
                "raw_score": (node.get("score") or {}).get("raw", {}),
                "relative_weight": (node.get("score") or {}).get("relative_weight"),
                "evidence_gap": (node.get("score") or {}).get("evidence_gap"),
                "answers_subgoal": (node.get("provenance") or {}).get("metadata", {}).get("answers_subgoal"),
                "join_depth": (graph.get("claim_semantics", {}).get(node_id) or {}).get("join_depth"),
            }
            for node_id, node in graph.get("nodes", {}).items()
            if node.get("kind") == "claim"
        }
        print(json.dumps({
            "question": graph.get("question"),
            "subgoals": {
                node_id: node for node_id, node in graph.get("nodes", {}).items()
                if node.get("kind") == "subgoal"
            },
            "query_graph": graph.get("query_graph", {}),
            "claims": claims,
            "branches": graph.get("branches", {}),
            "join_attempts": graph.get("join_attempt_history", []),
            "termination": graph.get("termination_history", []),
        }, ensure_ascii=False, indent=2))
        return
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
