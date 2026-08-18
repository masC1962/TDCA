#!/usr/bin/env python3
"""Post-hoc alignment of predicted plan length to evaluation hop labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    return {str(row["qid"]): row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    run = Path(args.run_dir)
    predictions = load(run / "predictions.jsonl")
    metrics = load(run / "per_example_metrics.jsonl")
    by_hop = {}
    for hop in sorted({int(row["hop_count"]) for row in metrics.values() if row.get("hop_count") is not None}):
        ids = [qid for qid, row in metrics.items() if int(row["hop_count"]) == hop]
        lengths = [len(predictions[qid].get("plan", {}).get("slots", [])) for qid in ids]
        by_hop[str(hop)] = {
            "count": len(ids),
            "mean_predicted_slots": sum(lengths) / len(lengths),
            "slot_count_histogram": {str(key): value for key, value in sorted(Counter(lengths).items())},
            "under_decomposed_rate": sum(length_ < hop for length_ in lengths) / len(lengths),
            "exact_length_rate": sum(length_ == hop for length_ in lengths) / len(lengths),
            "over_decomposed_rate": sum(length_ > hop for length_ in lengths) / len(lengths),
        }
    report = {"count": len(metrics), "by_hop": by_hop, "note": "Gold hop labels are used only post hoc."}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
