#!/usr/bin/env python3
"""Build a transparent quality/cost table from completed run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(label: str, directory: str) -> dict[str, object]:
    run = Path(directory)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    cost = json.loads((run / "cost_summary.json").read_text(encoding="utf-8"))
    count = int(metrics["count"])
    return {
        "label": label,
        "count": count,
        "exact_match": metrics["exact_match"],
        "f1": metrics["f1"],
        "answered_rate": metrics["answered_rate"],
        "selective_accuracy": metrics["selective_accuracy"],
        "mean_total_tokens": (cost["prompt_tokens"] + cost["completion_tokens"]) / count,
        "mean_llm_calls": cost["llm_calls"] / count,
        "mean_provider_calls": cost.get("provider_calls", cost["llm_calls"]) / count,
        "mean_provider_tokens": (
            cost.get("provider_prompt_tokens", cost["prompt_tokens"])
            + cost.get("provider_completion_tokens", cost["completion_tokens"])
        ) / count,
        "mean_retrieval_calls": cost["retrieval_calls"] / count,
        "mean_wall_seconds": cost["wall_seconds"] / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [load(label, directory) for label, directory in args.run]
    report = {"runs": rows, "note": "Wall seconds are summed per example and are not batch makespan."}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
