#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics")
    parser.add_argument("--output")
    args = parser.parse_args()
    grouped = defaultdict(list)
    for line in Path(args.metrics).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[round(float(row["confidence"]), 2)].append(float(row["exact_match"]))
    rows = [
        {"confidence": confidence, "count": len(values), "accuracy": sum(values) / len(values)}
        for confidence, values in sorted(grouped.items())
    ]
    report = {"values": rows, "note": "Rounded only for post-hoc reporting."}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
