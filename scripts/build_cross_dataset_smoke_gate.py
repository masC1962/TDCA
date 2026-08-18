#!/usr/bin/env python3
"""Open the cross-dataset tuning gate only from four audited smoke artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from tdca_research.utils import write_json


EXPECTED = {
    ("hotpotqa", "structured_tdca", "dense"),
    ("hotpotqa", "ircot", "bm25"),
    ("2wikimultihopqa", "structured_tdca", "dense"),
    ("2wikimultihopqa", "ircot", "bm25"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=4)
    parser.add_argument(
        "--output", default="research_outputs/cross_dataset_smoke_gate.json",
    )
    args = parser.parse_args()

    audits = []
    observed = set()
    for raw in args.run_dirs:
        run_dir = Path(raw)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_artifact.py")),
                str(run_dir),
                "--expected-count",
                "20",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        audit = json.loads(completed.stdout)
        config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        identity = (str(config.get("dataset")), str(config.get("method")), str(config.get("retriever")))
        if config.get("split") != "smoke" or config.get("setting") != "distractor":
            raise ValueError(f"not a cross-dataset distractor smoke artifact: {run_dir}")
        if audit["infrastructure_failures"] != 0:
            raise ValueError(f"smoke artifact contains infrastructure failures: {run_dir}")
        if identity in observed:
            raise ValueError(f"duplicate smoke method identity: {identity}")
        observed.add(identity)
        audits.append({**audit, "dataset": identity[0], "method": identity[1], "retriever": identity[2]})
    if observed != EXPECTED:
        raise ValueError(f"smoke artifact identities mismatch: expected {sorted(EXPECTED)}, found {sorted(observed)}")

    write_json(args.output, {
        "status": "open",
        "expected_count": 20,
        "reason": "All four pre-registered cross-dataset smoke artifacts passed strict audit with zero infrastructure failures.",
        "artifact_audits": sorted(audits, key=lambda row: (row["dataset"], row["method"])),
    })
    print(args.output)


if __name__ == "__main__":
    main()
