#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _latest_timestamped_dir(base_output_dir: Path) -> Path:
    candidates = sorted(
        base_output_dir.parent.glob(f"{base_output_dir.name}_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No timestamped output directory found for {base_output_dir}")
    return candidates[0]


def main() -> None:
    dataset_path = ROOT / "data" / "hotpot_dev_distractor_v1.jsonl"
    output_base = ROOT / "outputs" / "test_sparse_mock"

    cmd = [
        sys.executable,
        "scripts/run_baseline.py",
        "--baseline",
        "sparse_rag",
        "--dataset_path",
        str(dataset_path),
        "--dataset_name",
        "hotpotqa",
        "--limit",
        "2",
        "--top_k",
        "5",
        "--llm_backend",
        "mock",
        "--output_dir",
        str(output_base),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    run_dir = _latest_timestamped_dir(output_base)
    if not re.search(r"_\d{8}_\d{6}_sparse_rag_", run_dir.name):
        raise AssertionError(f"Output directory is not timestamped as expected: {run_dir.name}")

    pred_path = run_dir / "predictions.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 2:
        raise AssertionError(f"Expected 2 rows, found {len(rows)}")

    required_fields = {
        "pred",
        "gold",
        "retrieved_titles",
        "raw_generation",
        "generation_empty",
        "llm_finish_reason",
        "llm_usage",
        "llm_error",
    }
    for field in required_fields:
        if field not in rows[0]:
            raise AssertionError(f"Missing field in predictions row: {field}")

    if not str(rows[0].get("pred", "")).strip():
        raise AssertionError("Mock sparse_rag prediction is empty")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("timestamped_output"):
        raise AssertionError("Manifest should record timestamped_output=true")
    if int(manifest.get("max_new_tokens_answer", 0)) != 1200:
        raise AssertionError("Baseline default max_new_tokens_answer should be 1200")

    print(json.dumps(
        {
            "status": "ok",
            "output_dir": str(run_dir),
            "num_rows": len(rows),
            "sample_pred": rows[0]["pred"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
