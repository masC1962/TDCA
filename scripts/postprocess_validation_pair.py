#!/usr/bin/env python3
"""Fail-closed independent scoring and paired comparison for validation runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from tdca_research.compare import compare
from tdca_research.utils import sha256_file


def qids(path: Path) -> list[str]:
    return [str(json.loads(line)["qid"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def artifact_audit(run_dir: Path, expected_count: int, *, allow_legacy_schema: bool) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).with_name("verify_artifact.py")),
        str(run_dir),
        "--expected-count",
        str(expected_count),
    ]
    if allow_legacy_schema:
        command.append("--allow-legacy-schema")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def cost_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "cost_summary.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--main-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-legacy-schema",
        action="store_true",
        help="Allow only the explicitly supported pre-checkpoint schema in artifact auditing",
    )
    args = parser.parse_args()
    main_run, baseline_run = Path(args.main_run), Path(args.baseline_run)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    audits = {
        "main": artifact_audit(
            main_run, args.expected_count, allow_legacy_schema=args.allow_legacy_schema,
        ),
        "baseline": artifact_audit(
            baseline_run, args.expected_count, allow_legacy_schema=args.allow_legacy_schema,
        ),
    }

    main_ids = qids(main_run / "predictions.jsonl")
    baseline_ids = qids(baseline_run / "predictions.jsonl")
    if len(main_ids) != args.expected_count or len(set(main_ids)) != args.expected_count:
        raise ValueError(f"main artifact is partial or has duplicate qids: {len(main_ids)}")
    if main_ids != baseline_ids:
        raise ValueError("paired artifacts do not contain identical ordered qids")
    resolved = yaml.safe_load((main_run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    split = str(resolved.get("split", ""))
    independent_paths = {}
    for label, run_dir in (("main", main_run), ("baseline", baseline_run)):
        target = output / f"{label}_independent_eval.json"
        subprocess.run([
            sys.executable, "-m", "tdca_research.evaluate",
            "--dataset_path", args.dataset_path,
            "--dataset", args.dataset,
            "--predictions", str(run_dir / "predictions.jsonl"),
            "--output", str(target),
            "--expected-qids", str(main_run / "split_manifest.json"),
            "--split", split,
        ], check=True)
        independent_paths[label] = target

    main_eval = json.loads(independent_paths["main"].read_text(encoding="utf-8"))
    baseline_eval = json.loads(independent_paths["baseline"].read_text(encoding="utf-8"))
    main_rows = output / "main_rows.jsonl"
    baseline_rows = output / "baseline_rows.jsonl"
    for target, rows in ((main_rows, main_eval["rows"]), (baseline_rows, baseline_eval["rows"])):
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    comparison = compare(baseline_rows, main_rows, seed=520, samples=10000)
    (output / "paired_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "count": args.expected_count,
        "dataset": args.dataset,
        "dataset_sha256": sha256_file(args.dataset_path),
        "runs": {"main": str(main_run), "baseline": str(baseline_run)},
        "artifact_audits": audits,
        "main_metrics": main_eval["metrics"],
        "baseline_metrics": baseline_eval["metrics"],
        "main_by_hop": main_eval["metrics_by_hop"],
        "baseline_by_hop": baseline_eval["metrics_by_hop"],
        "main_by_type": main_eval["metrics_by_type"],
        "baseline_by_type": baseline_eval["metrics_by_type"],
        "cost": {"main": cost_summary(main_run), "baseline": cost_summary(baseline_run)},
        "paired_comparison": comparison,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "count": summary["count"],
        "main_em": main_eval["metrics"]["exact_match"],
        "main_f1": main_eval["metrics"]["f1"],
        "baseline_em": baseline_eval["metrics"]["exact_match"],
        "baseline_f1": baseline_eval["metrics"]["f1"],
        "main_infrastructure_failures": audits["main"]["infrastructure_failures"],
        "baseline_infrastructure_failures": audits["baseline"]["infrastructure_failures"],
        "paired_em": comparison["paired_bootstrap_exact_match"],
        "paired_f1": comparison.get("paired_bootstrap_f1"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
