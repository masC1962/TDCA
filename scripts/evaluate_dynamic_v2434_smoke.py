#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_dynamic_v2432_smoke import evaluate as evaluate_v2432
from tdca_research.utils import write_json


def evaluate(run: Path, preregistration: Path, config_path: Path) -> dict:
    report = evaluate_v2432(run, preregistration, config_path)
    report["schema_version"] = (
        "dynamic-hypergraph-v2.4.3.4-adaptive-smoke-a-gate-v1"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("configs/dynamic_v2434_preregistration.json"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/dynamic_hypergraph_v2434_qwen_smoke20.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.run, args.preregistration, args.config)
    write_json(args.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "failed_checks": report["failed_checks"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
