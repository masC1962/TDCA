#!/usr/bin/env python3
"""Print the aggregate section of a retrieval probe without per-row payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()
    report = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    summary = {
        "dataset": report["dataset"],
        "split": report["split"],
        "count": len(report["sample_ids"]),
        "top_k": report["top_k"],
        "retrievers": {
            name: {
                "support_recall": values["support_recall"],
                "all_gold_recalled": values["all_gold_recalled"],
                "wall_seconds": values["wall_seconds"],
                "by_hop": values.get("by_hop", {}),
            }
            for name, values in report["retrievers"].items()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
