#!/usr/bin/env python3
"""Print the compact, reportable subset of a HippoRAG comparison artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    print(json.dumps({
        "count": summary["count"],
        "hipporag_metrics": summary["hipporag_metrics"],
        "hipporag_by_hop": summary["hipporag_by_hop"],
        "hipporag_wall_seconds": summary["hipporag_wall_seconds"],
        "hipporag_cache_delta": summary["hipporag_cache_delta"],
        "status_changes": summary["paired_comparison"]["status_changes"],
        "paired_em": summary["paired_comparison"]["paired_bootstrap_exact_match"],
        "paired_f1": summary["paired_comparison"]["paired_bootstrap_f1"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
