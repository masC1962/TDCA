#!/usr/bin/env python3
"""Build a fail-closed matched-compute budget-curve report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_MODES = {"adaptive_evc", "uniform", "fixed_order"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(path: Path) -> dict[str, Any]:
    required = (
        "metrics.json", "dynamic_v2_metrics.json", "cost_summary.json",
        "run_manifest.json", "resolved_config.yaml", "predictions.jsonl",
        "partial_progress.json",
    )
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise ValueError(f"budget-curve run {path} is missing {missing}")
    config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    manifest = _json(path / "run_manifest.json")
    metrics = _json(path / "metrics.json")
    dynamic = _json(path / "dynamic_v2_metrics.json")
    cost = _json(path / "cost_summary.json")
    progress = _json(path / "partial_progress.json")
    predictions = _jsonl(path / "predictions.jsonl")
    return {
        "path": str(path),
        "allocator_mode": str(config.get("allocator_mode", "")),
        "budget": {
            "max_llm_calls": int(config.get("max_llm_calls", 0)),
            "max_total_tokens": int(config.get("max_total_tokens", 0)),
            "max_retrieval_calls": int(config.get("max_retrieval_calls", 0)),
        },
        "sample_ids": [str(row.get("qid")) for row in predictions],
        "dataset_sha256": str(manifest.get("dataset_sha256", "")),
        "model": str(manifest.get("model", "")),
        "prompt_version": str(manifest.get("prompt_version", "")),
        "split_seed": int(manifest.get("split_seed", -1)),
        "config_except_allocator_and_budget": {
            key: value for key, value in config.items()
            if key not in {
                "allocator_mode", "max_llm_calls", "max_total_tokens",
                "max_retrieval_calls",
            }
        },
        "complete": (
            progress.get("status") == "complete"
            and int(metrics.get("count", 0)) == len(predictions)
            and bool(predictions)
        ),
        "quality": {
            "f1": float(metrics.get("f1", 0.0)),
            "exact_match": float(metrics.get("exact_match", 0.0)),
            "candidate_presence": float(dynamic.get("candidate_presence_rate", 0.0)),
            "full_chain_completion": float(metrics.get("full_chain_completion_rate", 0.0)),
        },
        "cost": {
            "provider_calls": int(cost.get("provider_calls", 0)),
            "provider_reported_tokens": (
                int(cost.get("provider_prompt_tokens", 0))
                + int(cost.get("provider_completion_tokens", 0))
            ),
            "logical_llm_calls": int(cost.get("llm_calls", 0)),
            "logical_tokens": (
                int(cost.get("prompt_tokens", 0)) + int(cost.get("completion_tokens", 0))
            ),
            "retrieval_calls": int(cost.get("retrieval_calls", 0)),
            "cache_hits": int(cost.get("cache_hits", 0)),
        },
        "artifact_sha256": {
            name: _sha256(path / name) for name in required
        },
    }


def build(paths: list[Path]) -> dict[str, Any]:
    rows = [_run(path) for path in paths]
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        budget = row["budget"]
        key = (
            budget["max_llm_calls"], budget["max_total_tokens"],
            budget["max_retrieval_calls"],
        )
        grouped.setdefault(key, []).append(row)
    points = []
    common_samples = rows[0]["sample_ids"] if rows else []
    common_identity = None
    for budget, point_rows in sorted(grouped.items()):
        modes = [row["allocator_mode"] for row in point_rows]
        identities = {
            json.dumps({
                "sample_ids": row["sample_ids"],
                "dataset_sha256": row["dataset_sha256"],
                "model": row["model"],
                "prompt_version": row["prompt_version"],
                "split_seed": row["split_seed"],
                "config": row["config_except_allocator_and_budget"],
            }, sort_keys=True)
            for row in point_rows
        }
        identity = next(iter(identities), "")
        if common_identity is None:
            common_identity = identity
        point_complete = (
            len(point_rows) == len(REQUIRED_MODES)
            and set(modes) == REQUIRED_MODES
            and len(modes) == len(set(modes))
            and len(identities) == 1
            and all(row["complete"] for row in point_rows)
        )
        points.append({
            "budget": {
                "max_llm_calls": budget[0], "max_total_tokens": budget[1],
                "max_retrieval_calls": budget[2],
            },
            "modes": sorted(modes),
            "matched_identity": len(identities) == 1,
            "complete": point_complete,
            "runs": sorted(point_rows, key=lambda row: row["allocator_mode"]),
        })
    complete = (
        len(points) >= 2
        and all(point["complete"] for point in points)
        and all(row["sample_ids"] == common_samples for row in rows)
        and all(
            json.dumps({
                "sample_ids": row["sample_ids"],
                "dataset_sha256": row["dataset_sha256"],
                "model": row["model"],
                "prompt_version": row["prompt_version"],
                "split_seed": row["split_seed"],
                "config": row["config_except_allocator_and_budget"],
            }, sort_keys=True) == common_identity
            for row in rows
        )
    )
    return {
        "schema_version": "dynamic-hypergraph-v2.2-budget-curve-v1",
        "complete": complete,
        "required_allocator_modes": sorted(REQUIRED_MODES),
        "point_count": len(points),
        "sample_count": len(common_samples),
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "complete": report["complete"]}))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
