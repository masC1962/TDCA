#!/usr/bin/env python3
"""Build a zero-API matched smoke comparison and an explicit Go/No-Go decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_dynamic_v23_offline import analyze
from scripts.verify_artifact import verify


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _spearman(report: dict[str, Any]) -> float | None:
    return report["allocation_calibration"]["overall"][
        "spearman_predicted_evc_actual_utility"
    ]


def summarize(label: str, run: Path) -> dict[str, Any]:
    artifact = verify(run, expected_count=20)
    metrics = _json(run / "metrics.json")
    dynamic = _json(run / "dynamic_v2_metrics.json")
    cost = _json(run / "cost_summary.json")
    offline = analyze(run)
    join_audit = offline["join_frontier_audit"]
    per_metrics = {str(row["qid"]): row for row in _jsonl(run / "per_example_metrics.jsonl")}
    per_dynamic = {
        str(row["qid"]): row
        for row in _jsonl(run / "dynamic_v2_per_example_metrics.jsonl")
    }
    return {
        "label": label,
        "run": str(run),
        "artifact_verified": bool(artifact["verified"]),
        "infrastructure_failures": int(artifact["infrastructure_failures"]),
        "exact_match": float(metrics["exact_match"]),
        "f1": float(metrics["f1"]),
        "candidate_presence": float(dynamic["candidate_presence_rate"]),
        "full_chain_completion": float(metrics["full_chain_completion_rate"]),
        "selective_accuracy": float(metrics["selective_accuracy"]),
        "budget_exhaustion": float(metrics["budget_exhaustion_rate"]),
        "unsupported_answers": int(dynamic["unsupported_answer_count"]),
        "llm_calls": int(cost["llm_calls"]),
        "tokens": int(cost["prompt_tokens"]) + int(cost["completion_tokens"]),
        "retrieval_calls": int(cost["retrieval_calls"]),
        "mean_claim_count": float(dynamic["mean_claim_count"]),
        "mean_allocation_count": float(dynamic["mean_allocation_count"]),
        "evc_utility_spearman": _spearman(offline),
        "real_operation_choice_rate": float(
            offline["ready_set_audit"]["real_operation_choice_rate"]
        ),
        "extraction_bottleneck_count": int(
            offline["terminal_bottlenecks"]["bottleneck_counts"].get(
                "context_to_candidate_extraction", 0,
            )
        ),
        "join_attempt_count": int(join_audit["attempt_count"]),
        "join_accepted_count": int(join_audit["accepted_count"]),
        "join_charged_count": int(join_audit["charged_count"]),
        "join_answer_used_count": int(join_audit["answer_used_count"]),
        "join_llm_calls": int(join_audit["llm_calls"]),
        "per_example": {
            qid: {
                "hop_count": int(row.get("hop_count", 0) or 0),
                "status": str(row.get("status", "")),
                "exact_match": bool(float(row.get("exact_match", 0.0))),
                "full_chain": bool(float(row.get("full_chain_complete", 0.0))),
                "candidate_presence": bool(
                    per_dynamic.get(qid, {}).get("candidate_presence", False)
                ),
                "llm_calls": int(row.get("llm_calls", 0) or 0),
                "tokens": int(row.get("total_tokens", 0) or 0),
                "join": dict(join_audit["by_qid"].get(qid, {})),
            }
            for qid, row in per_metrics.items()
        },
    }


def compare(baseline: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        row["delta_vs_baseline"] = {
            key: row[key] - baseline[key]
            for key in (
                "exact_match", "f1", "candidate_presence", "full_chain_completion",
                "selective_accuracy", "budget_exhaustion", "llm_calls", "tokens",
                "retrieval_calls", "mean_claim_count", "mean_allocation_count",
                "real_operation_choice_rate", "extraction_bottleneck_count",
                "join_attempt_count", "join_accepted_count", "join_charged_count",
                "join_answer_used_count", "join_llm_calls",
            )
        }
        current_corr = row["evc_utility_spearman"]
        base_corr = baseline["evc_utility_spearman"]
        row["delta_vs_baseline"]["evc_utility_spearman"] = (
            None if current_corr is None or base_corr is None else current_corr - base_corr
        )
        base_examples = baseline.get("per_example", {})
        current_examples = row.get("per_example", {})
        common = sorted(set(base_examples) & set(current_examples))
        row["paired_vs_baseline"] = {
            "common_qids": len(common),
            "chain_gained": [
                qid for qid in common
                if current_examples[qid]["full_chain"] and not base_examples[qid]["full_chain"]
            ],
            "chain_lost": [
                qid for qid in common
                if base_examples[qid]["full_chain"] and not current_examples[qid]["full_chain"]
            ],
            "candidate_gained": [
                qid for qid in common
                if current_examples[qid]["candidate_presence"]
                and not base_examples[qid]["candidate_presence"]
            ],
            "candidate_lost": [
                qid for qid in common
                if base_examples[qid]["candidate_presence"]
                and not current_examples[qid]["candidate_presence"]
            ],
            "exact_match_gained": [
                qid for qid in common
                if current_examples[qid]["exact_match"] and not base_examples[qid]["exact_match"]
            ],
            "exact_match_lost": [
                qid for qid in common
                if base_examples[qid]["exact_match"] and not current_examples[qid]["exact_match"]
            ],
            "budget_exhausted_qids": [
                qid for qid in common
                if current_examples[qid]["status"] == "budget_exhausted"
            ],
        }
    selected = rows[-1]
    checks = {
        "artifact_complete": selected["artifact_verified"],
        "zero_infrastructure_failure": selected["infrastructure_failures"] == 0,
        "zero_unsupported_answer": selected["unsupported_answers"] == 0,
        "candidate_presence_plus_0_10": (
            selected["candidate_presence"] >= baseline["candidate_presence"] + 0.10
        ),
        "full_chain_non_regression": (
            selected["full_chain_completion"] >= baseline["full_chain_completion"]
        ),
        "f1_non_regression": selected["f1"] >= baseline["f1"],
        "positive_evc_calibration": (
            selected["evc_utility_spearman"] is not None
            and selected["evc_utility_spearman"] > 0.0
        ),
        "bounded_llm_call_growth": selected["llm_calls"] <= 1.10 * baseline["llm_calls"],
        "bounded_graph_growth": (
            selected["mean_claim_count"] <= 1.25 * baseline["mean_claim_count"]
            and selected["mean_allocation_count"] <= 1.25 * baseline["mean_allocation_count"]
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "dynamic-hypergraph-v2.3-smoke-comparison-v1",
        "inference_calls_made": 0,
        "baseline": baseline,
        "candidates": rows,
        "selected_candidate": selected["label"],
        "checks": checks,
        "passed": passed,
        "decision": "GO_MATCHED_CONTROLS" if passed else "NO_GO_FIX_BEFORE_CONTROLS",
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def render(report: dict[str, Any]) -> str:
    rows = [report["baseline"], *report["candidates"]]
    lines = [
        "# Dynamic Hypergraph TDCA v2.3 smoke Go/No-Go",
        "",
        f"- Decision: **{report['decision']}**",
        f"- Selected candidate: `{report['selected_candidate']}`",
        "- Provider/LLM calls made by this comparison: 0",
        f"- Failed checks: `{report['failed_checks']}`",
        "",
        "| Run | EM | F1 | Candidate | Full chain | EVC↔utility | Calls | Tokens | Retrieval | Budget exhausted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        corr = row["evc_utility_spearman"]
        corr_text = "n/a" if corr is None else f"{corr:.4f}"
        lines.append(
            f"| {row['label']} | {row['exact_match']:.3f} | {row['f1']:.3f} | "
            f"{row['candidate_presence']:.3f} | {row['full_chain_completion']:.3f} | "
            f"{corr_text} | {row['llm_calls']} | {row['tokens']} | "
            f"{row['retrieval_calls']} | {row['budget_exhaustion']:.3f} |"
        )
    lines += ["", "## Gate checks", ""]
    lines.extend(
        f"- [{'x' if value else ' '}] {key}"
        for key, value in report["checks"].items()
    )
    selected = report["candidates"][-1].get("paired_vs_baseline", {})
    baseline_examples = report["baseline"].get("per_example", {})
    candidate_examples = report["candidates"][-1].get("per_example", {})
    lines += [
        "",
        "## Selected candidate paired transitions",
        "",
        f"- Chain gained/lost: {len(selected.get('chain_gained', []))}/"
        f"{len(selected.get('chain_lost', []))}",
        f"- Candidate gained/lost: {len(selected.get('candidate_gained', []))}/"
        f"{len(selected.get('candidate_lost', []))}",
        f"- Exact match gained/lost: {len(selected.get('exact_match_gained', []))}/"
        f"{len(selected.get('exact_match_lost', []))}",
        f"- Chain-lost qids: `{selected.get('chain_lost', [])}`",
        f"- Budget-exhausted qids: `{selected.get('budget_exhausted_qids', [])}`",
    ]
    for qid in selected.get("chain_lost", []):
        before = baseline_examples.get(qid, {})
        after = candidate_examples.get(qid, {})
        before_join = before.get("join", {})
        after_join = after.get("join", {})
        lines.append(
            f"- `{qid}`: status {before.get('status')} -> {after.get('status')}; "
            f"calls {before.get('llm_calls')} -> {after.get('llm_calls')}; "
            f"JOIN attempts/accepted/charged "
            f"{before_join.get('attempt_count', 0)}/{before_join.get('accepted_count', 0)}/"
            f"{before_join.get('charged_count', 0)} -> "
            f"{after_join.get('attempt_count', 0)}/{after_join.get('accepted_count', 0)}/"
            f"{after_join.get('charged_count', 0)}"
        )
    lines += [
        "",
        "The smoke split is development-only and too small for a paper claim. "
        "A failed check blocks matched controls and larger runs; it does not invalidate "
        "the mechanism-level improvements recorded above.",
        "",
    ]
    return "\n".join(lines)


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=_labeled_path, required=True)
    parser.add_argument("--run", type=_labeled_path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = summarize(*args.baseline)
    candidates = [summarize(*row) for row in args.run]
    report = compare(baseline, candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "comparison.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"],
        "failed_checks": report["failed_checks"],
        "output": str(args.output_dir),
        "inference_calls_made": 0,
    }))


if __name__ == "__main__":
    main()
