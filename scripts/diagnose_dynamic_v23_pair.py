#!/usr/bin/env python3
"""Zero-API paired diagnosis for Dynamic Hypergraph TDCA run transitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.analyze_dynamic_v23_offline import _jsonl, analyze
from scripts.compare_dynamic_v23_smoke import summarize


def _run_detail(run: Path) -> dict[str, dict[str, Any]]:
    offline = analyze(run)
    events: dict[str, list[dict[str, Any]]] = {}
    terminal: dict[str, dict[str, Any]] = {}
    for row in _jsonl(run / "reasoning_traces.jsonl"):
        qid = str(row.get("qid", ""))
        if row.get("event") == "allocation_reconciled":
            packet = row.get("allocation") or {}
            events.setdefault(qid, []).append({
                "step": int(row.get("step", 0) or 0),
                "family": str(packet.get("operation_family", "")),
                "region": str(packet.get("region_key", "")),
                "evc": float(packet.get("predicted_evc", 0.0) or 0.0),
                "utility": float(row.get("actual_utility", 0.0) or 0.0),
                "progressed": bool(row.get("progressed", False)),
                "failure": str(row.get("failure_reason", "")),
                "cost": dict(row.get("actual_cost") or {}),
            })
        elif row.get("event") == "meta_decision":
            terminal[qid] = {
                "outcome": str(row.get("outcome", "")),
                "reason": str(row.get("reason", "")),
                "best_predicted_evc": float(row.get("best_predicted_evc", 0.0) or 0.0),
            }

    retrievals: dict[str, list[dict[str, Any]]] = {}
    for row in offline["retrieval_marginal_utility"]["per_retrieval"]:
        retrievals.setdefault(str(row["qid"]), []).append(row)
    extractions: dict[str, list[dict[str, Any]]] = {}
    for row in offline["extraction_diagnostics"]["per_attempt"]:
        extractions.setdefault(str(row["qid"]), []).append(row)
    joins: dict[str, list[dict[str, Any]]] = {}
    for row in offline["join_frontier_audit"]["per_attempt"]:
        joins.setdefault(str(row["qid"]), []).append(row)

    result: dict[str, dict[str, Any]] = {}
    qids = set(events) | set(terminal) | set(retrievals) | set(extractions) | set(joins)
    for qid in sorted(qids):
        allocations = events.get(qid, [])
        join_rows = joins.get(qid, [])
        result[qid] = {
            "terminal": terminal.get(qid, {}),
            "allocation_family_counts": dict(Counter(row["family"] for row in allocations)),
            "allocation_timeline": allocations,
            "retrieval": [{
                key: row.get(key) for key in (
                    "subgoal_id", "subgoal_round", "hit_count",
                    "new_unique_passage_count", "final_supported_claim_count",
                    "used_by_accepted_answer", "actual_utility",
                )
            } for row in retrievals.get(qid, [])],
            "extraction": [{
                key: row.get(key) for key in (
                    "target_id", "focus_mode", "accepted", "raw_rows",
                    "accepted_rows", "rejections",
                )
            } for row in extractions.get(qid, [])],
            "join": join_rows,
            "join_rejection_reasons": dict(Counter(
                row["rejection_reason"] or "unspecified"
                for row in join_rows if not row["accepted"]
            )),
        }
    return result


def diagnose(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = summarize("baseline", baseline_path)
    candidate = summarize("candidate", candidate_path)
    base_detail = _run_detail(baseline_path)
    candidate_detail = _run_detail(candidate_path)
    common = sorted(set(baseline["per_example"]) & set(candidate["per_example"]))
    transitions = {
        "chain_gained": [
            qid for qid in common
            if candidate["per_example"][qid]["full_chain"]
            and not baseline["per_example"][qid]["full_chain"]
        ],
        "chain_lost": [
            qid for qid in common
            if baseline["per_example"][qid]["full_chain"]
            and not candidate["per_example"][qid]["full_chain"]
        ],
    }
    selected = sorted(set(transitions["chain_gained"] + transitions["chain_lost"]))
    return {
        "schema_version": "dynamic-v2.3-paired-diagnostic-v1",
        "inference_calls_made": 0,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "transitions": transitions,
        "per_example": {
            qid: {
                "baseline_metrics": baseline["per_example"][qid],
                "candidate_metrics": candidate["per_example"][qid],
                "baseline_trace": base_detail.get(qid, {}),
                "candidate_trace": candidate_detail.get(qid, {}),
            }
            for qid in selected
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Dynamic Hypergraph TDCA paired chain diagnostic", "",
        "- Provider/LLM calls made: 0",
        f"- Chain gained: `{report['transitions']['chain_gained']}`",
        f"- Chain lost: `{report['transitions']['chain_lost']}`", "",
    ]
    for qid in report["transitions"]["chain_lost"]:
        row = report["per_example"][qid]
        before = row["baseline_trace"]
        after = row["candidate_trace"]
        compact_joins = [{
            "kind": value.get("join_kind"),
            "premises": value.get("premise_count"),
            "alignment": value.get("goal_alignment"),
            "accepted": value.get("accepted"),
            "charged": value.get("charged"),
            "answer_used": value.get("used_by_accepted_answer"),
            "reason": value.get("rejection_reason"),
        } for value in after.get("join", [])]
        lines += [
            f"## {qid}", "",
            f"- Status/calls: {row['baseline_metrics']['status']}/"
            f"{row['baseline_metrics']['llm_calls']} -> "
            f"{row['candidate_metrics']['status']}/{row['candidate_metrics']['llm_calls']}",
            f"- Allocation families: `{before.get('allocation_family_counts', {})}` -> "
            f"`{after.get('allocation_family_counts', {})}`",
            f"- Terminal: `{before.get('terminal', {})}` -> `{after.get('terminal', {})}`",
            f"- Extraction: `{after.get('extraction', [])}`",
            f"- JOIN rejection reasons: `{after.get('join_rejection_reasons', {})}`",
            f"- JOIN attempts: `{compact_joins}`",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.baseline, args.candidate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_diagnostic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "paired_diagnostic.md").write_text(
        render(report), encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output_dir), "inference_calls_made": 0,
        "chain_gained": len(report["transitions"]["chain_gained"]),
        "chain_lost": len(report["transitions"]["chain_lost"]),
    }))


if __name__ == "__main__":
    main()
