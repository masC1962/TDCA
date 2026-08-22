#!/usr/bin/env python3
"""Aggregate actual provider usage for one declared Dynamic v2 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage(path: Path) -> dict[str, Any]:
    recorded_path = str(path)
    if path.is_dir():
        cost_path = path / "cost_summary.json"
        if not cost_path.exists():
            return {"path": recorded_path, "type": "run", "complete": False,
                    "error": "missing cost_summary.json"}
        cost = read_json(cost_path)
        required = {"provider_calls", "provider_prompt_tokens", "provider_completion_tokens"}
        complete = required.issubset(cost)
        return {
            "path": recorded_path,
            "type": "qa_run",
            "complete": complete,
            "usage_sha256": digest(cost_path),
            "provider_calls": int(cost.get("provider_calls", 0)),
            "provider_reported_tokens": int(cost.get("provider_prompt_tokens", 0))
            + int(cost.get("provider_completion_tokens", 0)),
        }
    report = read_json(path)
    metrics = report.get("metrics", {})
    required = {"provider_calls", "provider_reported_tokens", "complete_predictions"}
    complete = required.issubset(metrics) and bool(metrics.get("complete_predictions"))
    return {
        "path": recorded_path,
        "type": "revision_evaluation",
        "complete": complete,
        "usage_sha256": digest(path),
        "provider_calls": int(metrics.get("provider_calls", 0)),
        "provider_reported_tokens": int(metrics.get("provider_reported_tokens", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-call-cap", type=int, default=1500)
    parser.add_argument("--provider-token-cap", type=int, default=1_500_000)
    args = parser.parse_args()
    unique = {str(path.resolve()): path for path in args.artifact}
    rows = [usage(unique[key]) for key in sorted(unique)]
    provider_calls = sum(int(row.get("provider_calls", 0)) for row in rows)
    provider_tokens = sum(int(row.get("provider_reported_tokens", 0)) for row in rows)
    payload = {
        "schema_version": "dynamic-v2-campaign-usage-v1",
        "artifacts": rows,
        "artifact_count": len(rows),
        "provider_calls": provider_calls,
        "provider_reported_tokens": provider_tokens,
        "provider_call_cap": args.provider_call_cap,
        "provider_reported_token_cap": args.provider_token_cap,
        "within_caps": (
            provider_calls <= args.provider_call_cap
            and provider_tokens <= args.provider_token_cap
        ),
        "complete": bool(rows) and all(bool(row.get("complete")) for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "provider_calls": provider_calls,
        "provider_reported_tokens": provider_tokens,
        "within_caps": payload["within_caps"],
        "complete": payload["complete"],
    }, indent=2))
    if not payload["within_caps"] or not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
