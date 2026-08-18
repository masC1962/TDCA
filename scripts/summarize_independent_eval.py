#!/usr/bin/env python3
"""Print compact aggregate and grouped metrics from an independent evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = (
    "count", "exact_match", "f1", "answered_rate", "abstention_rate",
    "infrastructure_failure_rate", "selective_accuracy", "support_recall",
    "all_gold_document_recall", "answer_in_context_rate", "ece",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()
    data = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    print(json.dumps({
        "metrics": {key: data["metrics"].get(key) for key in KEYS},
        "metrics_by_hop": data.get("metrics_by_hop", {}),
        "metrics_by_type": data.get("metrics_by_type", {}),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
