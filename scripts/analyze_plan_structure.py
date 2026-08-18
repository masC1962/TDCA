#!/usr/bin/env python3
"""Measure whether saved plans expose meaningful scheduling choices."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    plan_types = Counter()
    slot_counts = Counter()
    root_counts = Counter()
    max_widths = Counter()
    branching = 0
    valid = 0
    details = []
    for row in rows:
        plan = row.get("plan")
        if not isinstance(plan, dict):
            continue
        slots = plan.get("slots", [])
        if not isinstance(slots, list) or not slots:
            continue
        valid += 1
        plan_types[str(plan.get("plan_type", "unknown"))] += 1
        slot_counts[len(slots)] += 1
        dependencies = {str(slot["slot_id"]): set(map(str, slot.get("dependencies", []))) for slot in slots}
        roots = sum(not values for values in dependencies.values())
        root_counts[roots] += 1
        complete: set[str] = set()
        max_ready = 0
        while len(complete) < len(dependencies):
            ready = [slot_id for slot_id, deps in dependencies.items() if slot_id not in complete and deps <= complete]
            if not ready:
                break
            max_ready = max(max_ready, len(ready))
            complete.add(ready[0])
        max_widths[max_ready] += 1
        branching += int(max_ready > 1)
        details.append({"qid": row.get("qid"), "slots": len(slots), "roots": roots, "max_ready_width": max_ready})
    report = {
        "prediction_count": len(rows),
        "valid_plan_count": valid,
        "plan_types": dict(plan_types),
        "slot_counts": {str(key): value for key, value in sorted(slot_counts.items())},
        "root_counts": {str(key): value for key, value in sorted(root_counts.items())},
        "max_ready_widths": {str(key): value for key, value in sorted(max_widths.items())},
        "branching_plan_rate": branching / valid if valid else None,
        "details": details,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
