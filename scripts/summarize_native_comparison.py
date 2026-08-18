#!/usr/bin/env python3
"""Print the compact reportable subset of a native paired comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    metric_keys = (
        "exact_match", "f1", "answered_rate", "selective_accuracy",
        "support_recall", "all_gold_document_recall", "answer_in_context_rate", "ece",
    )
    cost_keys = (
        "llm_calls", "provider_calls", "provider_attempts", "retrieval_calls",
        "prompt_tokens", "completion_tokens", "provider_prompt_tokens",
        "provider_completion_tokens", "wall_seconds", "cache_hits",
    )
    print(json.dumps({
        "count": summary["count"],
        "dataset": summary["dataset"],
        "main_metrics": {key: summary["main_metrics"].get(key) for key in metric_keys},
        "baseline_metrics": {key: summary["baseline_metrics"].get(key) for key in metric_keys},
        "main_by_hop": summary["main_by_hop"],
        "baseline_by_hop": summary["baseline_by_hop"],
        "cost": {
            label: {key: values.get(key) for key in cost_keys}
            for label, values in summary["cost"].items()
        },
        "status_changes": summary["paired_comparison"]["status_changes"],
        "paired_em": summary["paired_comparison"]["paired_bootstrap_exact_match"],
        "paired_f1": summary["paired_comparison"]["paired_bootstrap_f1"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
