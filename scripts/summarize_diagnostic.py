#!/usr/bin/env python3
"""Print compact case-analysis or plan-structure diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    keys = (
        "count", "category_counts", "category_counts_by_hop", "paired_outcomes",
        "paired_outcomes_by_hop", "prediction_count", "valid_plan_count", "plan_types",
        "slot_counts", "root_counts", "max_ready_widths", "branching_plan_rate",
    )
    print(json.dumps({key: data[key] for key in keys if key in data}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
