#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze(run: Path) -> dict[str, Any]:
    official = {str(row["qid"]): row for row in _jsonl(run / "per_example_metrics.jsonl")}
    dynamic = {
        str(row["qid"]): row for row in _jsonl(run / "dynamic_v2_per_example_metrics.jsonl")
    }
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(run / "reasoning_traces.jsonl"):
        traces[str(row.get("qid", ""))].append(row)
    cases = []
    for qid in sorted(set(official) | set(dynamic)):
        metric = official.get(qid, {})
        dyn = dynamic.get(qid, {})
        events = traces.get(qid, [])
        structured = sum(
            row.get("event") == "recoverable_model_failure"
            and row.get("error_type") == "StructuredOutputError"
            for row in events
        )
        retrieval_ready = bool(metric.get("all_gold_recalled"))
        answer_visible = bool(metric.get("answer_in_context"))
        candidate = bool(dyn.get("candidate_presence"))
        chain = bool(metric.get("full_chain_complete"))
        joined = int(dyn.get("join_count", 0) or 0) > 0
        if not retrieval_ready:
            frontier = "retrieval"
        elif structured:
            frontier = "structured_protocol"
        elif answer_visible and not candidate:
            frontier = "extraction"
        elif candidate and not chain and not joined:
            frontier = "join_discovery"
        elif candidate and not chain:
            frontier = "proof_closure"
        elif chain and float(metric.get("f1", 0.0) or 0.0) < 1.0:
            frontier = "terminal_synthesis"
        else:
            frontier = "closed_or_unattributed"
        cases.append({
            "qid": qid,
            "hop_count": metric.get("hop_count", dyn.get("hop_count")),
            "retrieval_ready": retrieval_ready,
            "answer_visible": answer_visible,
            "candidate_present": candidate,
            "join_materialized": joined,
            "full_chain_complete": chain,
            "structured_output_failure_count": structured,
            "first_observed_loss_frontier": frontier,
            "counterfactual_bounds": {
                "gold_passage_would_bypass_retrieval": not retrieval_ready,
                "oracle_atomic_claim_would_bypass_extraction": answer_visible and not candidate,
                "oracle_join_would_bypass_join_frontier": candidate and not chain,
            },
        })
    counts = Counter(row["first_observed_loss_frontier"] for row in cases)
    count = max(1, len(cases))
    return {
        "schema_version": "dynamic-v2.1-stage-bound-analysis-v1",
        "source_run": str(run),
        "methodology": {
            "kind": "offline_trace_counterfactual_bound",
            "zero_leakage": (
                "Gold-derived fields are read only after inference from evaluator artifacts; "
                "they are never passed to the reasoner, retriever, graph, or allocator."
            ),
            "causal_limit": (
                "The bounds identify the earliest observed missing capability. They are not an "
                "estimate of the final accuracy obtained by jointly replacing multiple stages."
            ),
        },
        "count": len(cases),
        "frontier_counts": dict(sorted(counts.items())),
        "rates": {
            "retrieval_ready": sum(row["retrieval_ready"] for row in cases) / count,
            "answer_visible": sum(row["answer_visible"] for row in cases) / count,
            "candidate_present": sum(row["candidate_present"] for row in cases) / count,
            "join_materialized": sum(row["join_materialized"] for row in cases) / count,
            "full_chain_complete": sum(row["full_chain_complete"] for row in cases) / count,
            "unrecovered_structured_output": sum(
                row["structured_output_failure_count"] > 0 for row in cases
            ) / count,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute zero-leakage stage bounds from a completed Dynamic v2.1 run",
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "count": result["count"], "frontier_counts": result["frontier_counts"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
