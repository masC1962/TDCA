#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(run: Path) -> dict[str, dict]:
    metrics = {str(row["qid"]): row for row in jsonl(run / "dynamic_v2_per_example_metrics.jsonl")}
    predictions = {str(row["qid"]): row for row in jsonl(run / "predictions.jsonl")}
    traces: dict[str, list[dict]] = defaultdict(list)
    for row in jsonl(run / "reasoning_traces.jsonl"):
        traces[str(row.get("qid", ""))].append(row)
    rows = {}
    for qid in sorted(set(predictions) | set(metrics)):
        metric = metrics.get(qid, {})
        events = traces.get(qid, [])
        rows[qid] = {
            "status": predictions.get(qid, {}).get("status"),
            "stop_reason": predictions.get(qid, {}).get("stop_reason"),
            "candidate": bool(metric.get("candidate_presence")),
            "chain": bool(metric.get("auditable_three_or_four_hop_join_case")),
            "allocation_count": int(metric.get("allocation_count", 0)),
            "nary_attempts": int(metric.get("nary_join_attempt_count", 0)),
            "nary_accepted": int(metric.get("nary_join_accepted_count", 0)),
            "events": Counter(str(row.get("event", "")) for row in events),
            "operations": Counter(
                str(row.get("operation", "")) for row in events if row.get("event") == "graph_operation"
            ),
            "model_failures": Counter(
                str(row.get("error_type", "")) for row in events
                if row.get("event") in {"recoverable_model_failure", "provider_refusal"}
            ),
            "join_rejections": sum(row.get("event") == "join_rejected" for row in events),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    before, after = summarize(args.before), summarize(args.after)
    changes = []
    for qid in sorted(set(before) | set(after)):
        left, right = before.get(qid, {}), after.get(qid, {})
        if (
            left.get("candidate") != right.get("candidate")
            or left.get("chain") != right.get("chain")
            or left.get("status") != right.get("status")
        ):
            changes.append({"qid": qid, "before": left, "after": right})
    print(json.dumps({
        "changed_count": len(changes),
        "candidate_lost": [row["qid"] for row in changes if row["before"].get("candidate") and not row["after"].get("candidate")],
        "candidate_gained": [row["qid"] for row in changes if row["after"].get("candidate") and not row["before"].get("candidate")],
        "chain_lost": [row["qid"] for row in changes if row["before"].get("chain") and not row["after"].get("chain")],
        "chain_gained": [row["qid"] for row in changes if row["after"].get("chain") and not row["before"].get("chain")],
        "changes": changes,
    }, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
