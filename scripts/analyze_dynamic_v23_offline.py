#!/usr/bin/env python3
"""Offline, post-hoc audit for Dynamic Hypergraph TDCA allocation research.

This script never calls an LLM and never participates in inference.  It reads a
completed (or explicitly partial) Dynamic v2 artifact and diagnoses five questions:

1. Does predicted EVC rank selected operations by their realized utility?
2. What did every retrieval add immediately and to the final proof graph?
3. Which observable bottleneck explains each non-answer?
4. How often did the allocator face a real operation/region choice rather than only
   choosing among fidelity variants of one operation?
5. Which JOIN proposals consumed compute, were rejected, or reached the final answer?

The report deliberately keeps post-hoc gold-aware diagnoses separate from runtime
signals.  Gold-aware fields are used only in ``terminal_bottlenecks`` and are never
written back into configs or inference state.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_qid(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["qid"]): row
        for row in rows
        if row.get("qid") is not None
    }


def _events_by_qid(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("qid") is not None:
            grouped[str(row["qid"])].append(row)
    return grouped


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _ranks(values: list[float]) -> list[float]:
    """Average ranks for ties, zero based; sufficient for Spearman auditing."""

    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = _ranks(left)
    y = _ranks(right)
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_scale = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_scale = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_scale == 0.0 or y_scale == 0.0:
        return None
    value = numerator / (x_scale * y_scale)
    if abs(value - 1.0) < 1e-12:
        return 1.0
    if abs(value + 1.0) < 1e-12:
        return -1.0
    return value


def _calibration_bins(rows: list[dict[str, Any]], count: int = 4) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: (row["predicted_evc"], row["qid"], row["operation_id"]))
    bins: list[dict[str, Any]] = []
    for bin_index in range(min(count, len(ordered))):
        start = bin_index * len(ordered) // min(count, len(ordered))
        end = (bin_index + 1) * len(ordered) // min(count, len(ordered))
        part = ordered[start:end]
        if not part:
            continue
        bins.append({
            "bin": bin_index + 1,
            "count": len(part),
            "min_predicted_evc": min(row["predicted_evc"] for row in part),
            "max_predicted_evc": max(row["predicted_evc"] for row in part),
            "mean_predicted_evc": _mean(row["predicted_evc"] for row in part),
            "mean_actual_utility": _mean(row["actual_utility"] for row in part),
            "progress_rate": _rate(sum(row["progressed"] for row in part), len(part)),
            "mean_terminal_gap_reduction": _mean(
                row["terminal_gap_reduction"] for row in part
            ),
            "mean_chain_progress": _mean(row["answer_chain_progress"] for row in part),
        })
    return bins


def allocation_calibration(reasoning_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    pending_choices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in reasoning_rows:
        if event.get("event") == "meta_decision" and str(event.get("outcome", "")) == "CONTINUE":
            candidates = list(event.get("allocation_candidates") or [])
            if candidates:
                operation_keys = {
                    (
                        str(row.get("operation_id", "")),
                        str(row.get("operation_family", "")),
                        str(row.get("region_key", "")),
                    )
                    for row in candidates
                }
                pending_choices[str(event.get("qid", ""))].append({
                    "selected_operation_id": str(candidates[0].get("operation_id", "")),
                    "real_operation_choice": len(operation_keys) > 1,
                    "choice_size": len(operation_keys),
                })
            continue
        if event.get("event") != "allocation_reconciled":
            continue
        packet = event.get("allocation") or {}
        raw_utility = event.get("actual_utility_components_raw") or {}
        qid = str(event.get("qid", ""))
        choice = pending_choices[qid].pop(0) if pending_choices[qid] else {}
        selected_operation = str(choice.get("selected_operation_id", ""))
        executed_operation = str(packet.get("operation_id", ""))
        choice_matched = bool(
            selected_operation
            and (
                executed_operation == selected_operation
                or executed_operation.startswith(f"{selected_operation}_allocation_")
            )
        )
        selected.append({
            "qid": qid,
            "operation_id": executed_operation,
            "operation_family": str(packet.get("operation_family", "unknown")),
            "region_key": str(packet.get("region_key", "")),
            "fidelity_level": str(packet.get("fidelity_level", "unknown")),
            "predicted_evc": _float(packet.get("predicted_evc")),
            "actual_utility": _float(event.get("actual_utility")),
            "progressed": bool(event.get("progressed")),
            "failure_reason": str(event.get("failure_reason", "")),
            "terminal_gap_reduction": _float(raw_utility.get("terminal_gap_reduction")),
            "answer_chain_progress": _float(raw_utility.get("answer_chain_progress")),
            "actual_cost": {
                str(key): _float(value)
                for key, value in (event.get("actual_cost") or {}).items()
            },
            "real_operation_choice": bool(
                choice_matched and choice.get("real_operation_choice", False)
            ),
            "choice_size": int(choice.get("choice_size", 0)) if choice_matched else 0,
            "choice_trace_matched": choice_matched,
        })

    def summarize(part: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(part),
            "spearman_predicted_evc_actual_utility": spearman(
                [row["predicted_evc"] for row in part],
                [row["actual_utility"] for row in part],
            ),
            "mean_predicted_evc": _mean(row["predicted_evc"] for row in part),
            "mean_actual_utility": _mean(row["actual_utility"] for row in part),
            "positive_utility_rate": _rate(
                sum(row["actual_utility"] > 0.0 for row in part), len(part)
            ),
            "progress_rate": _rate(sum(row["progressed"] for row in part), len(part)),
            "no_op_rate": _rate(sum(not row["progressed"] for row in part), len(part)),
            "mean_terminal_gap_reduction": _mean(
                row["terminal_gap_reduction"] for row in part
            ),
            "mean_answer_chain_progress": _mean(
                row["answer_chain_progress"] for row in part
            ),
        }

    by_family = {
        family: summarize([row for row in selected if row["operation_family"] == family])
        for family in sorted({row["operation_family"] for row in selected})
    }
    by_fidelity = {
        fidelity: summarize([row for row in selected if row["fidelity_level"] == fidelity])
        for fidelity in sorted({row["fidelity_level"] for row in selected})
    }
    choice_conditioned = [row for row in selected if row["real_operation_choice"]]
    if selected:
        high_threshold = sorted(row["predicted_evc"] for row in selected)[
            3 * (len(selected) - 1) // 4
        ]
    else:
        high_threshold = 0.0
    high_value_failures = [
        row for row in selected
        if row["predicted_evc"] >= high_threshold
        and (not row["progressed"] or row["actual_utility"] <= 0.0)
    ]
    return {
        "overall": summarize(selected),
        "calibration_bins_low_to_high": _calibration_bins(selected),
        "by_operation_family": by_family,
        "by_fidelity": by_fidelity,
        "choice_conditioned": {
            **summarize(choice_conditioned),
            "trace_match_rate": _rate(
                sum(row["choice_trace_matched"] for row in selected), len(selected)
            ),
            "minimum_choice_size": min(
                (row["choice_size"] for row in choice_conditioned), default=0,
            ),
        },
        "high_evc_nonpositive_or_noop_count": len(high_value_failures),
        "high_evc_nonpositive_or_noop_cases": high_value_failures[:50],
    }


def ready_set_audit(reasoning_rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for event in reasoning_rows:
        if event.get("event") != "meta_decision":
            continue
        candidates = list(event.get("allocation_candidates") or [])
        if not candidates:
            continue
        operation_keys = {
            (
                str(row.get("operation_id", "")),
                str(row.get("operation_family", "")),
                str(row.get("region_key", "")),
            )
            for row in candidates
        }
        families = {str(row.get("operation_family", "")) for row in candidates}
        regions = {str(row.get("region_key", "")) for row in candidates}
        operation_ids = {str(row.get("operation_id", "")) for row in candidates}
        continued = str(event.get("outcome", "")) == "CONTINUE"
        selected = candidates[0] if continued else {}
        decisions.append({
            "qid": str(event.get("qid", "")),
            "step": int(event.get("step", 0) or 0),
            "outcome": str(event.get("outcome", "")),
            "packet_count": len(candidates),
            "distinct_operation_count": len(operation_keys),
            "distinct_operation_id_count": len(operation_ids),
            "distinct_family_count": len(families),
            "distinct_region_count": len(regions),
            "real_operation_choice": len(operation_keys) > 1,
            "cross_family_choice": len(families) > 1,
            "cross_region_choice": len(regions) > 1,
            "fidelity_only_choice": len(operation_keys) == 1 and len(candidates) > 1,
            "selected_family": str(selected.get("operation_family", "")),
            "selected_fidelity": str(selected.get("fidelity_level", "")),
            "selected_predicted_evc": _float(selected.get("predicted_evc")),
        })
    count = len(decisions)
    continued = [row for row in decisions if row["outcome"] == "CONTINUE"]
    return {
        "decision_count_with_candidates": count,
        "continued_decision_count": len(continued),
        "real_operation_choice_rate": _rate(
            sum(row["real_operation_choice"] for row in decisions), count
        ),
        "cross_family_choice_rate": _rate(
            sum(row["cross_family_choice"] for row in decisions), count
        ),
        "cross_region_choice_rate": _rate(
            sum(row["cross_region_choice"] for row in decisions), count
        ),
        "fidelity_only_choice_rate": _rate(
            sum(row["fidelity_only_choice"] for row in decisions), count
        ),
        "mean_distinct_operations": _mean(
            row["distinct_operation_count"] for row in decisions
        ),
        "mean_distinct_regions": _mean(row["distinct_region_count"] for row in decisions),
        "selected_family_counts": dict(Counter(
            row["selected_family"] for row in continued if row["selected_family"]
        )),
        "selected_fidelity_counts": dict(Counter(
            row["selected_fidelity"] for row in continued if row["selected_fidelity"]
        )),
        "fidelity_only_selected_counts": dict(Counter(
            row["selected_fidelity"]
            for row in continued
            if row["fidelity_only_choice"] and row["selected_fidelity"]
        )),
        "per_decision": decisions,
    }


def _node_kind(node: dict[str, Any]) -> str:
    value = node.get("kind", "")
    if value:
        return str(value)
    if "document_id" in node and "source_span" in node:
        return "evidence"
    if "candidate_answer" in node:
        return "answer"
    if "evidence_refs" in node:
        return "claim"
    return ""


def retrieval_marginal_utility(
    retrieval_rows: list[dict[str, Any]],
    reasoning_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reconciled = {
        (str(row.get("qid", "")), str((row.get("allocation") or {}).get("operation_id", ""))): row
        for row in reasoning_rows
        if row.get("event") == "allocation_reconciled"
        and str((row.get("allocation") or {}).get("operation_family", "")).startswith("retrieve:")
    }
    graphs = {str(row.get("qid", "")): row.get("graph", {}) for row in graph_rows}
    grouped = _events_by_qid(retrieval_rows)
    cases: list[dict[str, Any]] = []
    for qid, rows in grouped.items():
        seen_passages: set[str] = set()
        region_rounds: Counter[tuple[str, str]] = Counter()
        graph = graphs.get(qid, {})
        nodes = graph.get("nodes") or {}
        claims = {
            node_id: node for node_id, node in nodes.items()
            if _node_kind(node) == "claim"
        }
        accepted_answers = [
            node for node in nodes.values()
            if _node_kind(node) == "answer" and str(node.get("status", "")) == "accepted"
        ]
        for round_index, row in enumerate(rows, start=1):
            operation_id = str(row.get("operation_id", ""))
            region = (str(row.get("branch_id", "")), str(row.get("subgoal_id", "")))
            region_rounds[region] += 1
            passage_ids = [str(hit.get("passage_id", "")) for hit in row.get("hits", [])]
            unique_in_call = list(dict.fromkeys(value for value in passage_ids if value))
            new_passages = [value for value in unique_in_call if value not in seen_passages]
            seen_passages.update(unique_in_call)
            evidence_ids = {
                node_id for node_id, node in nodes.items()
                if _node_kind(node) == "evidence"
                and str((node.get("provenance") or {}).get("operation_id", "")) == operation_id
            }
            supporting_claims = [
                claim_id for claim_id, claim in claims.items()
                if evidence_ids & set(map(str, claim.get("evidence_refs", [])))
            ]
            answer_used = any(
                evidence_ids & set(map(str, answer.get("supporting_evidence", [])))
                for answer in accepted_answers
            )
            outcome = reconciled.get((qid, operation_id), {})
            utility = outcome.get("actual_utility_components_raw") or {}
            cases.append({
                "qid": qid,
                "round": round_index,
                "subgoal_round": region_rounds[region],
                "operation_id": operation_id,
                "subgoal_id": str(row.get("subgoal_id", "")),
                "branch_id": str(row.get("branch_id", "")),
                "query": str(row.get("query", "")),
                "allocated_top_k": int(row.get("allocated_top_k", 0) or 0),
                "hit_count": len(passage_ids),
                "new_unique_passage_count": len(new_passages),
                "duplicate_or_previously_seen_count": len(passage_ids) - len(new_passages),
                "materialized_evidence_count": len(evidence_ids),
                "final_supported_claim_count": len(supporting_claims),
                "used_by_accepted_answer": answer_used,
                "predicted_evc": _float((outcome.get("allocation") or {}).get("predicted_evc")),
                "actual_utility": _float(outcome.get("actual_utility")),
                "progressed": bool(outcome.get("progressed")),
                "immediate_terminal_gap_reduction": _float(
                    utility.get("terminal_gap_reduction")
                ),
                "immediate_chain_progress": _float(utility.get("answer_chain_progress")),
                "zero_unique_passage_yield": not new_passages,
                "no_final_claim_yield": not supporting_claims,
            })
    by_round: dict[str, dict[str, Any]] = {}
    for round_index in sorted({row["round"] for row in cases}):
        part = [row for row in cases if row["round"] == round_index]
        by_round[str(round_index)] = {
            "count": len(part),
            "mean_new_unique_passages": _mean(
                row["new_unique_passage_count"] for row in part
            ),
            "mean_final_supported_claims": _mean(
                row["final_supported_claim_count"] for row in part
            ),
            "answer_use_rate": _rate(sum(row["used_by_accepted_answer"] for row in part), len(part)),
            "no_final_claim_yield_rate": _rate(
                sum(row["no_final_claim_yield"] for row in part), len(part)
            ),
            "mean_actual_utility": _mean(row["actual_utility"] for row in part),
        }
    by_subgoal_round: dict[str, dict[str, Any]] = {}
    for round_index in sorted({row["subgoal_round"] for row in cases}):
        part = [row for row in cases if row["subgoal_round"] == round_index]
        by_subgoal_round[str(round_index)] = {
            "count": len(part),
            "mean_new_unique_passages": _mean(
                row["new_unique_passage_count"] for row in part
            ),
            "mean_final_supported_claims": _mean(
                row["final_supported_claim_count"] for row in part
            ),
            "answer_use_rate": _rate(sum(row["used_by_accepted_answer"] for row in part), len(part)),
            "no_final_claim_yield_rate": _rate(
                sum(row["no_final_claim_yield"] for row in part), len(part)
            ),
            "mean_actual_utility": _mean(row["actual_utility"] for row in part),
        }
    return {
        "retrieval_count": len(cases),
        "mean_new_unique_passages": _mean(row["new_unique_passage_count"] for row in cases),
        "zero_unique_passage_yield_rate": _rate(
            sum(row["zero_unique_passage_yield"] for row in cases), len(cases)
        ),
        "no_final_claim_yield_rate": _rate(
            sum(row["no_final_claim_yield"] for row in cases), len(cases)
        ),
        "accepted_answer_use_rate": _rate(
            sum(row["used_by_accepted_answer"] for row in cases), len(cases)
        ),
        "by_retrieval_round": by_round,
        "by_subgoal_retrieval_round": by_subgoal_round,
        "per_retrieval": cases,
    }


def terminal_bottlenecks(
    predictions: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    dynamic_rows: list[dict[str, Any]],
    reasoning_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _by_qid(metrics_rows)
    dynamic = _by_qid(dynamic_rows)
    terminal_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reasoning_rows:
        if row.get("event") == "terminal_belief_readout" and row.get("qid") is not None:
            terminal_events[str(row["qid"])].append(row)
    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        qid = str(prediction.get("qid", ""))
        status = str(prediction.get("status", ""))
        if status == "answer":
            continue
        metric = metrics.get(qid, {})
        dyn = dynamic.get(qid, {})
        answer_in_context = bool(metric.get("answer_in_context", False))
        all_gold = bool(metric.get("all_gold_recalled", False))
        candidate = bool(dyn.get("candidate_presence", False))
        survival = bool(dyn.get("candidate_survival", False))
        chain = bool(metric.get("full_chain_complete", False))
        diagnostics = [
            candidate_row
            for event in terminal_events.get(qid, [])
            for candidate_row in event.get("candidates", [])
        ]
        rejection_reasons = sorted({
            str(reason)
            for row in diagnostics
            for reason in row.get("rejection_reasons", [])
        })
        if status == "budget_exhausted":
            bottleneck = "budget_exhaustion"
        elif not answer_in_context and not all_gold:
            bottleneck = "retrieval_access"
        elif answer_in_context and not candidate:
            bottleneck = "context_to_candidate_extraction"
        elif candidate and not survival:
            bottleneck = "candidate_survival_or_revision"
        elif candidate and not chain:
            bottleneck = "proof_chain_or_join_completion"
        elif chain:
            bottleneck = "terminal_competition_or_acceptance"
        elif rejection_reasons:
            bottleneck = "terminal_readout_rejection"
        else:
            bottleneck = "no_executable_or_unclassified"
        cases.append({
            "qid": qid,
            "status": status,
            "stop_reason": str(prediction.get("stop_reason", "")),
            "bottleneck": bottleneck,
            "answer_in_context": answer_in_context,
            "all_gold_recalled": all_gold,
            "candidate_presence": candidate,
            "candidate_survival": survival,
            "full_chain_complete": chain,
            "terminal_rejection_reasons": rejection_reasons,
        })
    counts = Counter(row["bottleneck"] for row in cases)
    return {
        "non_answer_count": len(cases),
        "bottleneck_counts": dict(counts),
        "bottleneck_rates": {
            key: _rate(value, len(cases)) for key, value in sorted(counts.items())
        },
        "per_example": cases,
    }


def extraction_diagnostics(reasoning_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize generic extraction failures without reading answer labels."""
    diagnostic_events = [
        row for row in reasoning_rows
        if row.get("event") == "typed_extraction_diagnostic"
    ]
    # v2.2/v2.3.0 traces recorded only rejected calls.  Preserve their audit
    # value and label the denominator explicitly as rejection-only coverage.
    legacy = False
    if not diagnostic_events:
        legacy = True
        diagnostic_events = [
            {
                **row,
                "accepted": False,
                "focus_mode": str(row.get("focus_mode", "unknown")),
            }
            for row in reasoning_rows
            if row.get("event") == "typed_extraction_rejected"
        ]
    rejection_counts: Counter[str] = Counter()
    by_focus: dict[str, Counter[str]] = defaultdict(Counter)
    per_attempt = []
    for row in diagnostic_events:
        diagnostics = row.get("diagnostics") or {}
        accepted = bool(row.get("accepted"))
        focus = str(row.get("focus_mode", "unknown"))
        rejections = {
            str(key): int(value)
            for key, value in (diagnostics.get("rejections") or {}).items()
        }
        rejection_counts.update(rejections)
        by_focus[focus]["attempts"] += 1
        by_focus[focus]["accepted"] += int(accepted)
        by_focus[focus]["raw_rows"] += int(diagnostics.get("raw", 0) or 0)
        by_focus[focus]["accepted_rows"] += int(diagnostics.get("accepted", 0) or 0)
        per_attempt.append({
            "qid": str(row.get("qid", "")),
            "operation_id": str(row.get("operation_id", "")),
            "target_id": str(row.get("target_id", "")),
            "branch_id": str(row.get("branch_id", "")),
            "focus_mode": focus,
            "accepted": accepted,
            "raw_rows": int(diagnostics.get("raw", 0) or 0),
            "accepted_rows": int(diagnostics.get("accepted", 0) or 0),
            "focused_evidence_count": int(
                diagnostics.get("focused_evidence_count", 0) or 0
            ),
            "focused_context_characters": int(
                diagnostics.get("focused_context_characters", 0) or 0
            ),
            "budget_compacted_context": bool(
                diagnostics.get("budget_compacted_context", False)
            ),
            "rejections": rejections,
        })
    return {
        "trace_coverage": "rejections_only" if legacy else "all_attempts",
        "attempt_count": len(per_attempt),
        "accepted_attempt_rate": _rate(
            sum(row["accepted"] for row in per_attempt), len(per_attempt)
        ),
        "zero_raw_rate": _rate(
            sum(row["raw_rows"] == 0 for row in per_attempt), len(per_attempt)
        ),
        "budget_compaction_rate": _rate(
            sum(row["budget_compacted_context"] for row in per_attempt), len(per_attempt)
        ),
        "rejection_reason_counts": dict(rejection_counts),
        "by_focus_mode": {
            key: dict(value) for key, value in sorted(by_focus.items())
        },
        "per_attempt": per_attempt,
    }


def join_frontier_audit(graph_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit JOIN compute, rejection, and accepted-answer proof use."""

    cases: list[dict[str, Any]] = []
    for graph_row in graph_rows:
        qid = str(graph_row.get("qid", ""))
        graph = graph_row.get("graph") or {}
        nodes = graph.get("nodes") or {}
        answer_claims = {
            str(claim_id)
            for node in nodes.values()
            if _node_kind(node) == "answer" and str(node.get("status", "")) == "accepted"
            for claim_id in node.get("supporting_claims", [])
        }
        for attempt in graph.get("join_attempt_history", []):
            cost = attempt.get("creation_cost") or {}
            validation = attempt.get("deterministic_validation") or {}
            conclusion = str(attempt.get("conclusion_node_id", ""))
            cases.append({
                "qid": qid,
                "operation_id": str(attempt.get("operation_id", "")),
                "target_subgoal": str(attempt.get("target_subgoal", "")),
                "branch_id": str(attempt.get("branch_id", "")),
                "join_kind": str(attempt.get("join_kind", "unknown")),
                "premise_count": len(attempt.get("premise_ids", [])),
                "premise_ids": [str(value) for value in attempt.get("premise_ids", [])],
                "goal_alignment": _float(validation.get("goal_alignment")),
                "accepted": bool(attempt.get("accepted", False)),
                "rejection_reason": str(attempt.get("rejection_reason", "")),
                "llm_calls": _float(cost.get("llm_calls")),
                "tokens": _float(cost.get("tokens")),
                "charged": bool(attempt.get("accepted", False))
                or _float(cost.get("llm_calls")) > 0.0,
                "downstream_unlock": _float(attempt.get("downstream_unlock")),
                "conclusion_node_id": conclusion,
                "used_by_accepted_answer": bool(conclusion and conclusion in answer_claims),
            })

    def summarize(part: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "attempt_count": len(part),
            "accepted_count": sum(row["accepted"] for row in part),
            "acceptance_rate": _rate(sum(row["accepted"] for row in part), len(part)),
            "charged_count": sum(row["charged"] for row in part),
            "llm_calls": sum(row["llm_calls"] for row in part),
            "tokens": sum(row["tokens"] for row in part),
            "answer_used_count": sum(row["used_by_accepted_answer"] for row in part),
            "answer_use_rate": _rate(
                sum(row["used_by_accepted_answer"] for row in part), len(part),
            ),
            "mean_goal_alignment": _mean(row["goal_alignment"] for row in part),
            "mean_downstream_unlock": _mean(row["downstream_unlock"] for row in part),
        }

    return {
        **summarize(cases),
        "rejection_reason_counts": dict(Counter(
            row["rejection_reason"] or "unspecified"
            for row in cases if not row["accepted"]
        )),
        "by_join_kind": {
            kind: summarize([row for row in cases if row["join_kind"] == kind])
            for kind in sorted({row["join_kind"] for row in cases})
        },
        "by_qid": {
            qid: {
                **summarize([row for row in cases if row["qid"] == qid]),
                "rejection_reasons": dict(Counter(
                    row["rejection_reason"] or "unspecified"
                    for row in cases
                    if row["qid"] == qid and not row["accepted"]
                )),
            }
            for qid in sorted({row["qid"] for row in cases})
        },
        "per_attempt": cases,
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    required = (
        "reasoning_traces.jsonl",
        "retrieval_traces.jsonl",
        "dynamic_v2_graphs.jsonl",
        "predictions.jsonl",
        "per_example_metrics.jsonl",
        "dynamic_v2_per_example_metrics.jsonl",
    )
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise ValueError(f"run {run_dir} is missing required artifacts: {missing}")
    reasoning = _jsonl(run_dir / "reasoning_traces.jsonl")
    retrieval = _jsonl(run_dir / "retrieval_traces.jsonl")
    graphs = _jsonl(run_dir / "dynamic_v2_graphs.jsonl")
    predictions = _jsonl(run_dir / "predictions.jsonl")
    metrics = _jsonl(run_dir / "per_example_metrics.jsonl")
    dynamic = _jsonl(run_dir / "dynamic_v2_per_example_metrics.jsonl")
    return {
        "schema_version": "dynamic-hypergraph-v2.3-offline-audit-v1",
        "source_run": str(run_dir),
        "sample_count": len(predictions),
        "inference_calls_made": 0,
        "gold_usage_boundary": (
            "Gold-aware per-example fields are used only for post-hoc terminal "
            "failure attribution, never for inference or policy state."
        ),
        "allocation_calibration": allocation_calibration(reasoning),
        "ready_set_audit": ready_set_audit(reasoning),
        "retrieval_marginal_utility": retrieval_marginal_utility(
            retrieval, reasoning, graphs,
        ),
        "terminal_bottlenecks": terminal_bottlenecks(
            predictions, metrics, dynamic, reasoning,
        ),
        "extraction_diagnostics": extraction_diagnostics(reasoning),
        "join_frontier_audit": join_frontier_audit(graphs),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    allocation = report["allocation_calibration"]
    ready = report["ready_set_audit"]
    retrieval = report["retrieval_marginal_utility"]
    terminal = report["terminal_bottlenecks"]
    extraction = report["extraction_diagnostics"]
    joins = report["join_frontier_audit"]
    lines = [
        "# Dynamic Hypergraph TDCA v2.3 offline diagnostic",
        "",
        f"- Source: `{report['source_run']}`",
        f"- Samples: {report['sample_count']}",
        "- Provider/LLM calls made by this audit: 0",
        "- Gold boundary: post-hoc terminal attribution only",
        "",
        "## Allocation calibration",
        "",
        "| Slice | Count | Spearman(EVC, utility) | Mean utility | Progress | No-op |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    slices = {"overall": allocation["overall"], **allocation["by_operation_family"]}
    for name, row in slices.items():
        lines.append(
            f"| {name} | {row['count']} | "
            f"{_fmt(row['spearman_predicted_evc_actual_utility'])} | "
            f"{_fmt(row['mean_actual_utility'])} | {_fmt(row['progress_rate'])} | "
            f"{_fmt(row['no_op_rate'])} |"
        )
    lines += [
        "",
        "## Ready-set choice audit",
        "",
        f"- Decisions with candidates: {ready['decision_count_with_candidates']}",
        f"- Real operation-choice rate: {_fmt(ready['real_operation_choice_rate'])}",
        f"- Cross-family choice rate: {_fmt(ready['cross_family_choice_rate'])}",
        f"- Cross-region choice rate: {_fmt(ready['cross_region_choice_rate'])}",
        f"- Fidelity-only choice rate: {_fmt(ready['fidelity_only_choice_rate'])}",
        f"- Selected families: `{json.dumps(ready['selected_family_counts'], sort_keys=True)}`",
        f"- Selected fidelities: `{json.dumps(ready['selected_fidelity_counts'], sort_keys=True)}`",
        "",
        "## Retrieval marginal utility",
        "",
        f"- Retrievals: {retrieval['retrieval_count']}",
        f"- Mean new unique passages: {_fmt(retrieval['mean_new_unique_passages'])}",
        f"- Zero-unique-yield rate: {_fmt(retrieval['zero_unique_passage_yield_rate'])}",
        f"- No final-claim-yield rate: {_fmt(retrieval['no_final_claim_yield_rate'])}",
        f"- Accepted-answer evidence-use rate: {_fmt(retrieval['accepted_answer_use_rate'])}",
        "",
        "| Retrieval round | Count | New passages | Final supported claims | Answer use | No claim yield | Utility |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for round_index, row in retrieval["by_retrieval_round"].items():
        lines.append(
            f"| {round_index} | {row['count']} | {_fmt(row['mean_new_unique_passages'])} | "
            f"{_fmt(row['mean_final_supported_claims'])} | {_fmt(row['answer_use_rate'])} | "
            f"{_fmt(row['no_final_claim_yield_rate'])} | {_fmt(row['mean_actual_utility'])} |"
        )
    lines += [
        "",
        "### Within-subgoal retrieval rounds",
        "",
        "| Subgoal round | Count | New passages | Final supported claims | Answer use | No claim yield | Utility |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for round_index, row in retrieval["by_subgoal_retrieval_round"].items():
        lines.append(
            f"| {round_index} | {row['count']} | {_fmt(row['mean_new_unique_passages'])} | "
            f"{_fmt(row['mean_final_supported_claims'])} | {_fmt(row['answer_use_rate'])} | "
            f"{_fmt(row['no_final_claim_yield_rate'])} | {_fmt(row['mean_actual_utility'])} |"
        )
    lines += [
        "",
        "## Non-answer bottlenecks",
        "",
        "| Bottleneck | Count | Rate |",
        "|---|---:|---:|",
    ]
    for name, count in sorted(
        terminal["bottleneck_counts"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(
            f"| {name} | {count} | {_fmt(terminal['bottleneck_rates'][name])} |"
        )
    lines += [
        "",
        "## Extraction diagnostics",
        "",
        f"- Trace coverage: {extraction['trace_coverage']}",
        f"- Recorded attempts: {extraction['attempt_count']}",
        f"- Accepted-attempt rate: {_fmt(extraction['accepted_attempt_rate'])}",
        f"- Empty model-output rate: {_fmt(extraction['zero_raw_rate'])}",
        f"- Budget-compacted context rate: {_fmt(extraction['budget_compaction_rate'])}",
        f"- Rejection reasons: `{json.dumps(extraction['rejection_reason_counts'], sort_keys=True)}`",
        "",
        "## JOIN frontier audit",
        "",
        f"- Attempts / accepted / charged: {joins['attempt_count']} / "
        f"{joins['accepted_count']} / {joins['charged_count']}",
        f"- Acceptance rate: {_fmt(joins['acceptance_rate'])}",
        f"- Accepted-answer use rate: {_fmt(joins['answer_use_rate'])}",
        f"- JOIN model calls / tokens: {_fmt(joins['llm_calls'])} / {_fmt(joins['tokens'])}",
        f"- Rejection reasons: `{json.dumps(joins['rejection_reason_counts'], sort_keys=True)}`",
    ]
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Correlation is observational over selected actions, not a counterfactual policy-value estimate.",
        "- Retrieval-to-final-claim attribution uses final graph provenance and does not imply sole causality.",
        "- Gold-aware bottlenecks are evaluation-only and must never become inference features.",
        "- A real allocation claim requires distinct executable operations/regions, not only token-fidelity variation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "offline_diagnostic.json"
    markdown_path = args.output_dir / "offline_diagnostic.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(markdown_path),
        "sample_count": report["sample_count"],
        "inference_calls_made": 0,
    }))


if __name__ == "__main__":
    main()
