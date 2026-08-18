#!/usr/bin/env python3
"""Compare executed slot orders across aligned scheduler ablations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def orders(run_dir: str) -> dict[str, list[str]]:
    values: dict[str, list[tuple[int, str]]] = defaultdict(list)
    path = Path(run_dir) / "reasoning_traces.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        values[str(row["qid"])].append((int(row["step"]), str(row["slot_id"])))
    return {qid: [slot for _, slot in sorted(rows)] for qid, rows in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    aligned = {label: orders(path) for label, path in args.run}
    labels = list(aligned)
    qids = sorted(set.intersection(*(set(values) for values in aligned.values())))
    differences = []
    for qid in qids:
        per_label = {label: aligned[label][qid] for label in labels}
        if len({tuple(order) for order in per_label.values()}) > 1:
            differences.append({"qid": qid, "orders": per_label})
    report = {
        "labels": labels,
        "aligned_qids_with_reasoning_trace": len(qids),
        "identical_order_count": len(qids) - len(differences),
        "different_order_count": len(differences),
        "different_order_rate": len(differences) / len(qids) if qids else None,
        "differences": differences,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
