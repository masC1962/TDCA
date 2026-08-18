from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import write_json


def _jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    identifiers = [str(row["qid"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate qids in analysis input: {path}")
    return {str(row["qid"]): row for row in rows}


def categorize(row: dict[str, Any], prediction: dict[str, Any]) -> str:
    if float(row.get("exact_match", 0.0)) == 1.0:
        return "correct"
    if prediction.get("status") == "infrastructure_failure":
        return "infrastructure_failure"
    if "budget" in str(prediction.get("stop_reason", "")) or "budget_exhausted" in prediction.get("rejection_reasons", []):
        return "budget_exhaustion"
    if float(row.get("all_gold_recalled", 0.0)) == 0.0:
        return "retrieval_miss"
    node_accuracy = row.get("decomposition_node_accuracy")
    if node_accuracy is not None and float(node_accuracy) < 0.5:
        return "decomposition_error"
    binding_accuracy = row.get("variable_binding_accuracy")
    if binding_accuracy is not None and float(binding_accuracy) < 1.0:
        return "binding_error"
    claim_precision = row.get("verified_claim_precision")
    if claim_precision is not None and float(claim_precision) < 0.5 and int(row.get("verified_claim_count", 0)):
        return "claim_hallucination_or_wrong_extraction"
    if prediction.get("status") == "abstain" and float(row.get("answer_in_context", 0.0)) == 1.0:
        return "verification_false_reject_or_missing_terminal"
    if float(row.get("full_chain_complete", 0.0)) == 1.0:
        return "final_synthesis_error"
    return "incomplete_reasoning_chain"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create aligned, gold-dependent failure categories after a completed run")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--baseline_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics = _jsonl(run_dir / "per_example_metrics.jsonl")
    predictions = _jsonl(run_dir / "predictions.jsonl")
    if set(metrics) != set(predictions):
        raise ValueError("prediction and metric qids must match exactly")
    rows = []
    for qid in sorted(metrics):
        row = metrics[qid]
        category = categorize(row, predictions[qid])
        rows.append({
            "qid": qid,
            "category": category,
            "hop_count": row.get("hop_count"),
            "exact_match": row.get("exact_match"),
            "f1": row.get("f1"),
            "status": row.get("status"),
            "all_gold_recalled": row.get("all_gold_recalled"),
            "answer_in_context": row.get("answer_in_context"),
            "stop_reason": predictions[qid].get("stop_reason"),
        })

    by_hop: dict[str, dict[str, int]] = {}
    for hop in sorted({str(row["hop_count"]) for row in rows}):
        subset = [row for row in rows if str(row["hop_count"]) == hop]
        by_hop[hop] = dict(Counter(row["category"] for row in subset))
    result: dict[str, Any] = {
        "count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "category_counts_by_hop": by_hop,
        "rows": rows,
        "note": "Categories are deterministic post-hoc diagnostics and never enter inference prompts.",
    }
    if args.baseline_dir:
        baseline = _jsonl(Path(args.baseline_dir) / "per_example_metrics.jsonl")
        if set(metrics) != set(baseline):
            raise ValueError("baseline and main metric qids must match exactly")
        aligned = sorted(metrics)
        result["paired_outcomes"] = dict(Counter(
            f"base_{int(float(baseline[qid]['exact_match']))}->new_{int(float(metrics[qid]['exact_match']))}"
            for qid in aligned
        ))
        result["paired_outcomes_by_hop"] = {
            hop: dict(Counter(
                f"base_{int(float(baseline[qid]['exact_match']))}->new_{int(float(metrics[qid]['exact_match']))}"
                for qid in aligned if str(metrics[qid].get("hop_count")) == hop
            ))
            for hop in sorted({str(metrics[qid].get("hop_count")) for qid in aligned})
        }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
