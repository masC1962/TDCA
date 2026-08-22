#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_ORDER = (
    "provider_or_infrastructure_failure",
    "retrieval_miss",
    "query_formulation_or_missing_binding_miss",
    "claim_extraction_miss",
    "type_or_binding_mismatch",
    "join_verification_rejection",
    "join_expressivity_failure",
    "candidate_commit_or_survival_failure",
    "final_synthesis_failure",
    "premature_stop",
    "evc_misallocation",
    "budget_exhaustion",
    "correct_or_no_observed_failure",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["qid"]): row for row in rows if row.get("qid") is not None}


def _events_by_qid(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("qid") is not None:
            grouped[str(row["qid"])].append(row)
    return grouped


def _normalized_query(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def analyze(run_dir: Path, baseline_commit: str) -> dict[str, Any]:
    metrics = _by_qid(_jsonl(run_dir / "per_example_metrics.jsonl"))
    dynamic = _by_qid(_jsonl(run_dir / "dynamic_v2_per_example_metrics.jsonl"))
    predictions = _by_qid(_jsonl(run_dir / "predictions.jsonl"))
    retrieval = _events_by_qid(_jsonl(run_dir / "retrieval_traces.jsonl"))
    reasoning = _events_by_qid(_jsonl(run_dir / "reasoning_traces.jsonl"))
    qids = sorted(set(metrics) | set(dynamic) | set(predictions))
    cases: list[dict[str, Any]] = []

    for qid in qids:
        metric = metrics.get(qid, {})
        dyn = dynamic.get(qid, {})
        pred = predictions.get(qid, {})
        traces = reasoning.get(qid, [])
        retrieval_rows = retrieval.get(qid, [])
        status = str(pred.get("status", metric.get("status", "")))
        f1 = float(metric.get("f1", 0.0) or 0.0)
        full_chain = bool(metric.get("full_chain_complete", 0.0))
        answer_in_context = bool(metric.get("answer_in_context", 0.0))
        all_gold_recalled = bool(metric.get("all_gold_recalled", 0.0))
        candidate_presence = bool(dyn.get("candidate_presence", False))
        candidate_survival = bool(dyn.get("candidate_survival", False))
        hop_count = int(metric.get("hop_count", dyn.get("hop_count", 0)) or 0)
        join_count = int(dyn.get("join_count", 0) or 0)
        allocation_count = int(dyn.get("allocation_count", 0) or 0)
        join_rejections = [row for row in traces if row.get("event") == "join_rejected"]
        model_failures = [
            row for row in traces
            if row.get("event") in {"recoverable_model_failure", "infrastructure_failure"}
        ]
        termination = next(
            (row for row in reversed(traces) if row.get("event") == "termination"), {}
        )
        queries = [_normalized_query(row.get("query")) for row in retrieval_rows]
        generic_queries = [
            query for query in queries
            if "find a missing relation" in query or "existing typed claims" in query
        ]
        labels: list[dict[str, Any]] = []

        def add(label: str, rule: str, evidence: dict[str, Any]) -> None:
            labels.append({"label": label, "inference_rule": rule, "evidence": evidence})

        if model_failures or float(metric.get("infrastructure_failure", 0.0) or 0.0) > 0:
            add(
                "provider_or_infrastructure_failure",
                "A provider/infrastructure trace event or metric was recorded.",
                {
                    "event_count": len(model_failures),
                    "error_types": sorted({str(row.get("error_type", "")) for row in model_failures}),
                },
            )
        if not all_gold_recalled:
            add(
                "retrieval_miss",
                "The official per-example metric reports all_gold_recalled=false.",
                {
                    "retrieval_calls": int(metric.get("retrieval_calls", len(retrieval_rows)) or 0),
                    "retrieval_queries": queries,
                    "ordered_evidence_path_recall": metric.get("ordered_evidence_path_recall"),
                },
            )
            if generic_queries or len(set(queries)) < len(queries):
                add(
                    "query_formulation_or_missing_binding_miss",
                    "Gold retrieval failed and the trace contains a generic fallback or repeated query.",
                    {
                        "generic_fallback_queries": generic_queries,
                        "repeated_query_count": max(0, len(queries) - len(set(queries))),
                    },
                )
        if answer_in_context and not candidate_presence:
            add(
                "claim_extraction_miss",
                "The answer occurs in retrieved context but no gold-answer candidate was materialized.",
                {
                    "answer_in_context": True,
                    "candidate_presence": False,
                    "verified_claim_count": metric.get("verified_claim_count"),
                },
            )
        binding_accuracy = float(metric.get("variable_binding_accuracy", 1.0) or 0.0)
        if binding_accuracy < 1.0:
            add(
                "type_or_binding_mismatch",
                "The official variable_binding_accuracy is below 1.0.",
                {"variable_binding_accuracy": binding_accuracy},
            )
        if join_rejections:
            add(
                "join_verification_rejection",
                "At least one proposed JOIN was explicitly rejected in the reasoning trace.",
                {
                    "count": len(join_rejections),
                    "reason_codes": [
                        row.get("diagnostics", {}).get("reason_codes", []) for row in join_rejections
                    ],
                    "signatures": [row.get("join_signature") for row in join_rejections],
                },
            )
        if hop_count >= 3 and not full_chain and join_count < max(1, hop_count - 1):
            add(
                "join_expressivity_failure",
                "A 3/4-hop example lacks a full chain and materialized fewer than hop_count-1 JOIN states.",
                {"hop_count": hop_count, "join_count": join_count, "full_chain_complete": False},
            )
        if candidate_presence and not candidate_survival:
            add(
                "candidate_commit_or_survival_failure",
                "A gold-answer candidate was created but no such candidate remained active.",
                {"candidate_presence": True, "candidate_survival": False},
            )
        if candidate_survival and (status != "answer" or f1 < 1.0):
            add(
                "final_synthesis_failure",
                "A gold-answer candidate survived but the terminal answer was absent or not fully correct.",
                {"status": status, "f1": f1, "stop_reason": pred.get("stop_reason")},
            )
        if status == "abstain" and (answer_in_context or candidate_presence) and not model_failures:
            add(
                "premature_stop",
                "The run abstained despite retrieved answer text or a gold-answer candidate.",
                {
                    "answer_in_context": answer_in_context,
                    "candidate_presence": candidate_presence,
                    "termination_reason": termination.get("reason", pred.get("stop_reason")),
                },
            )
        if status == "budget_exhausted":
            if not full_chain or not candidate_presence:
                add(
                    "evc_misallocation",
                    "Budget was exhausted before candidate/chain completion; this is an allocation symptom, not proof of sole causality.",
                    {
                        "allocation_count": allocation_count,
                        "candidate_presence": candidate_presence,
                        "full_chain_complete": full_chain,
                        "termination_reason": termination.get("reason", pred.get("stop_reason")),
                    },
                )
            add(
                "budget_exhaustion",
                "The terminal status is budget_exhausted.",
                {
                    "llm_calls": metric.get("llm_calls"),
                    "total_tokens": metric.get("total_tokens"),
                    "retrieval_calls": metric.get("retrieval_calls"),
                    "allocation_count": allocation_count,
                },
            )
        if not labels and f1 >= 1.0:
            add(
                "correct_or_no_observed_failure",
                "No listed failure rule fired and official F1 is 1.0.",
                {"status": status, "f1": f1},
            )
        if not labels:
            add(
                "final_synthesis_failure",
                "The answer is not fully correct and no earlier trace-grounded failure rule fired.",
                {"status": status, "f1": f1, "stop_reason": pred.get("stop_reason")},
            )
        ranks = {name: index for index, name in enumerate(LABEL_ORDER)}
        labels.sort(key=lambda row: ranks[row["label"]])
        cases.append({
            "qid": qid,
            "hop_count": hop_count,
            "status": status,
            "f1": f1,
            "main_cause": labels[0]["label"],
            "secondary_causes": [row["label"] for row in labels[1:]],
            "labels": labels,
        })

    main_counts = Counter(row["main_cause"] for row in cases)
    all_counts = Counter(label["label"] for row in cases for label in row["labels"])
    return {
        "schema_version": "dynamic-v2-failure-taxonomy-v1",
        "source_run": str(run_dir),
        "baseline_commit": baseline_commit,
        "methodology": {
            "kind": "deterministic_trace-derived_failure_attribution",
            "causal_warning": (
                "Labels are reproducible symptoms inferred from frozen metrics and traces; "
                "evc_misallocation and query-formulation labels are hypotheses, not sole-cause proofs."
            ),
            "label_order_for_main_cause": list(LABEL_ORDER),
        },
        "count": len(cases),
        "main_cause_counts": dict(main_counts),
        "all_label_counts": dict(all_counts),
        "cases": cases,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Dynamic Hypergraph TDCA v2 pre-change failure taxonomy",
        "",
        f"- Source run: `{result['source_run']}`",
        f"- Frozen baseline commit: `{result['baseline_commit']}`",
        f"- Cases: {result['count']}",
        "- Method: deterministic rules over frozen official metrics and reasoning/retrieval traces.",
        "- Caution: query/EVC labels are trace-grounded hypotheses, not claims of unique causality.",
        "",
        "## Counts",
        "",
        "| Label | Main cause | Any label |",
        "|---|---:|---:|",
    ]
    for label in LABEL_ORDER:
        lines.append(
            f"| {label} | {result['main_cause_counts'].get(label, 0)} | "
            f"{result['all_label_counts'].get(label, 0)} |"
        )
    lines.extend([
        "",
        "## Per-example attribution",
        "",
        "| QID | Hop | Status | F1 | Main cause | Secondary causes |",
        "|---|---:|---|---:|---|---|",
    ])
    for row in result["cases"]:
        lines.append(
            f"| `{row['qid']}` | {row['hop_count']} | {row['status']} | {row['f1']:.3f} | "
            f"{row['main_cause']} | {', '.join(row['secondary_causes']) or '-'} |"
        )
    lines.extend([
        "",
        "The companion JSON contains every inference rule and its supporting trace fields.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a frozen Dynamic v2 failure taxonomy")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.baseline_commit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "failure_taxonomy.json"
    jsonl_path = args.output_dir / "failure_taxonomy.jsonl"
    md_path = args.output_dir / "failure_taxonomy.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result["cases"]),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "count": result["count"],
        "main_cause_counts": result["main_cause_counts"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
