#!/usr/bin/env python3
"""Verify completion, required files, checksums and row counts for a native run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdca_research.experiments import ArtifactWriter
from tdca_research.utils import sha256_file


def verify(
    run_dir: str | Path, expected_count: int | None = None, *, allow_legacy_schema: bool = False,
) -> dict[str, object]:
    run = Path(run_dir)
    missing = [name for name in ArtifactWriter.FILES if not (run / name).exists()]
    legacy_optional = {"metrics_by_type.json", "partial_progress.json"}
    if missing and not (allow_legacy_schema and set(missing).issubset(legacy_optional)):
        raise ValueError(f"required artifacts missing: {missing}")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    progress_path = run / "partial_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
    if not manifest.get("completed_at_utc") or (progress is not None and progress.get("status") != "complete"):
        raise ValueError("run is not marked complete")
    checksums_path = run / "artifact_checksums.json"
    if not checksums_path.exists():
        raise ValueError("artifact_checksums.json is missing")
    expected = json.loads(checksums_path.read_text(encoding="utf-8"))
    mismatches = [
        name for name, digest in expected.items()
        if not (run / name).is_file() or sha256_file(run / name) != digest
    ]
    if mismatches:
        raise ValueError(f"artifact checksum mismatch: {mismatches}")
    prediction_ids = [
        str(json.loads(line)["qid"])
        for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metric_ids = [
        str(json.loads(line)["qid"])
        for line in (run / "per_example_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_ids = [str(value) for value in manifest.get("sample_ids", [])]
    if prediction_ids != sample_ids or metric_ids != sample_ids:
        raise ValueError("prediction/metric qids do not match the ordered manifest sample IDs")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("duplicate prediction qids")
    if expected_count is not None and len(prediction_ids) != expected_count:
        raise ValueError(f"expected {expected_count} predictions, found {len(prediction_ids)}")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if int(metrics.get("count", -1)) != len(prediction_ids):
        raise ValueError("metrics count does not match predictions")
    return {
        "run_dir": str(run), "verified": True, "count": len(prediction_ids),
        "schema": "legacy_pre_checkpoint" if missing else "current",
        "missing_current_schema_files": missing,
        "checksum_count": len(expected), "infrastructure_failures": sum(
            1 for line in (run / "failures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--allow-legacy-schema", action="store_true",
        help="Allow completed pre-checkpoint artifacts missing only metrics_by_type/partial_progress",
    )
    args = parser.parse_args()
    print(json.dumps(
        verify(args.run_dir, args.expected_count, allow_legacy_schema=args.allow_legacy_schema),
        indent=2, ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
