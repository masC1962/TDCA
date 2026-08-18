#!/usr/bin/env python3
"""Summarize per-example resource use and proximity to frozen budget limits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics")
    parser.add_argument("--max-llm-calls", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.metrics).read_text(encoding="utf-8").splitlines() if line.strip()]
    calls = [float(row["llm_calls"]) for row in rows]
    tokens = [float(row["total_tokens"]) for row in rows]
    report = {
        "count": len(rows),
        "llm_calls": {
            "mean": sum(calls) / len(calls), "p50": quantile(calls, 0.5),
            "p90": quantile(calls, 0.9), "max": max(calls),
            "at_limit": sum(value >= args.max_llm_calls for value in calls),
        },
        "total_tokens": {
            "mean": sum(tokens) / len(tokens), "p50": quantile(tokens, 0.5),
            "p90": quantile(tokens, 0.9), "max": max(tokens),
            "at_limit": sum(value >= args.max_total_tokens for value in tokens),
        },
        "stop_reasons": {},
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
