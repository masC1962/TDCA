#!/usr/bin/env python3
"""Compactly report completion/progress for a known run directory or exit file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--exit-file")
    args = parser.parse_args()
    report: dict[str, object] = {}
    if args.exit_file:
        path = Path(args.exit_file)
        report["exit_file_present"] = path.exists()
        report["exit_code"] = path.read_text(encoding="utf-8").strip() if path.exists() else None
    if args.run_dir:
        run = Path(args.run_dir)
        report["run_dir_present"] = run.exists()
        for name in ("predictions.jsonl", "per_example_metrics.jsonl", "failures.jsonl"):
            path = run / name
            report[f"{name}_lines"] = sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else None
        manifest_path = run / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report["completed_at_utc"] = manifest.get("completed_at_utc")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
