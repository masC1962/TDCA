#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()
    data = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    print(json.dumps({
        "count": data["count"],
        "category_counts": data["category_counts"],
        "category_counts_by_hop": data.get("category_counts_by_hop", {}),
        "paired_outcomes": data.get("paired_outcomes"),
        "paired_outcomes_by_hop": data.get("paired_outcomes_by_hop"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
