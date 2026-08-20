#!/usr/bin/env python3
"""Fail-closed independent HippoRAG scoring and paired native-run comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from tdca_research.compare import compare
from tdca_research.data import load_examples
from tdca_research.utils import sha256_file


def _jsonl_qids(path: Path) -> list[str]:
    return [str(json.loads(line)["qid"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _audit_native(run_dir: Path, expected_count: int, *, allow_legacy_schema: bool) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).with_name("verify_artifact.py")),
        str(run_dir),
        "--expected-count",
        str(expected_count),
    ]
    if allow_legacy_schema:
        command.append("--allow-legacy-schema")
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--main-run", required=True)
    parser.add_argument("--hipporag-artifact", required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=520)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-legacy-schema",
        action="store_true",
        help="Allow only the explicitly supported pre-checkpoint native artifact schema",
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main_run = Path(args.main_run)
    hippo_path = Path(args.hipporag_artifact)
    main_audit = _audit_native(
        main_run, args.expected_count, allow_legacy_schema=args.allow_legacy_schema,
    )
    if hippo_path.name.endswith(".partial"):
        raise ValueError("partial HippoRAG checkpoints must never be scored or compared")
    hippo = json.loads(hippo_path.read_text(encoding="utf-8"))
    hippo_ids = [str(row["qid"]) for row in hippo.get("rows", [])]
    main_ids = _jsonl_qids(main_run / "predictions.jsonl")
    if len(main_ids) != args.expected_count or len(set(main_ids)) != args.expected_count:
        raise ValueError("main artifact is partial or contains duplicate qids")
    if (
        hippo.get("sample_count") != args.expected_count
        or len(hippo_ids) != args.expected_count
        or len(set(hippo_ids)) != args.expected_count
    ):
        raise ValueError("HippoRAG artifact is partial")
    if set(hippo_ids) != set(main_ids):
        raise ValueError("HippoRAG and main artifacts do not contain identical qid sets")
    resolved = yaml.safe_load((main_run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    split = str(resolved.get("split", ""))

    main_eval_path = output / "main_independent_eval.json"
    subprocess.run([
        sys.executable, "-m", "tdca_research.evaluate",
        "--dataset_path", args.dataset_path, "--dataset", args.dataset,
        "--predictions", str(main_run / "predictions.jsonl"), "--output", str(main_eval_path),
        "--expected-qids", str(main_run / "split_manifest.json"), "--split", split,
    ], check=True)
    hippo_eval_path = output / "hipporag_independent_eval.json"
    subprocess.run([
        sys.executable, "external_baselines/evaluate_hipporag_artifact.py",
        "--input", str(hippo_path), "--output", str(hippo_eval_path),
    ], check=True)

    main_eval = json.loads(main_eval_path.read_text(encoding="utf-8"))
    hippo_eval = json.loads(hippo_eval_path.read_text(encoding="utf-8"))
    main_evaluated_ids = [str(row["qid"]) for row in main_eval["rows"]]
    hippo_evaluated_ids = [str(row["qid"]) for row in hippo_eval["rows"]]
    if len(set(main_evaluated_ids)) != args.expected_count or set(main_evaluated_ids) != set(main_ids):
        raise ValueError("independent main evaluation changed the qid set")
    if len(set(hippo_evaluated_ids)) != args.expected_count or set(hippo_evaluated_ids) != set(hippo_ids):
        raise ValueError("independent HippoRAG evaluation changed the qid set")

    examples = {example.qid: example for example in load_examples(args.dataset_path, args.dataset)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in hippo_eval["rows"]:
        grouped[str(examples[str(row["qid"])].hop_count)].append(row)
    hippo_by_hop = {
        hop: {
            "count": len(rows),
            "exact_match": sum(float(row["exact_match"]) for row in rows) / len(rows),
            "f1": sum(float(row["f1"]) for row in rows) / len(rows),
            "answered_rate": sum(row["status"] == "answer" for row in rows) / len(rows),
        }
        for hop, rows in sorted(grouped.items())
    }

    main_rows_path = output / "main_rows.jsonl"
    hippo_rows_path = output / "hipporag_rows.jsonl"
    main_rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in main_eval["rows"]), encoding="utf-8",
    )
    hippo_rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in hippo_eval["rows"]), encoding="utf-8",
    )
    paired = compare(hippo_rows_path, main_rows_path, seed=args.bootstrap_seed, samples=10000)
    _write_json(output / "paired_comparison.json", paired)

    before = hippo.get("shared_cache_before", {})
    after = hippo.get("shared_cache_after", {})
    cache_delta = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in ("calls", "prompt_tokens", "completion_tokens")
    }
    summary = {
        "count": args.expected_count,
        "source_provenance": {
            "main_predictions": str(main_run / "predictions.jsonl"),
            "main_predictions_sha256": sha256_file(main_run / "predictions.jsonl"),
            "hipporag_artifact": str(hippo_path),
            "hipporag_artifact_sha256": sha256_file(hippo_path),
            "hipporag_repository_commit": hippo.get("repository_commit"),
        },
        "main_artifact_audit": main_audit,
        "main_cost": json.loads((main_run / "cost_summary.json").read_text(encoding="utf-8")),
        "main_metrics": main_eval["metrics"],
        "main_by_hop": main_eval["metrics_by_hop"],
        "hipporag_metrics": {key: hippo_eval[key] for key in (
            "exact_match", "f1", "parser_recoveries", "parser_failures",
        )},
        "hipporag_by_hop": hippo_by_hop,
        "hipporag_upstream_qa_metrics": hippo.get("qa_metrics", {}),
        "hipporag_upstream_retrieval_metrics": hippo.get("retrieval_metrics", {}),
        "hipporag_wall_seconds": hippo.get("wall_seconds"),
        "hipporag_cache_delta": cache_delta,
        "paired_comparison": paired,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({
        "count": summary["count"],
        "main_em": main_eval["metrics"]["exact_match"],
        "hipporag_em": hippo_eval["exact_match"],
        "main_f1": main_eval["metrics"]["f1"],
        "hipporag_f1": hippo_eval["f1"],
        "paired_em": paired["paired_bootstrap_exact_match"],
        "paired_f1": paired.get("paired_bootstrap_f1"),
        "cache_delta": cache_delta,
    }, indent=2))


if __name__ == "__main__":
    main()
