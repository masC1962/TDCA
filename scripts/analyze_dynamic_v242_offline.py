#!/usr/bin/env python3
"""Gold-free, two-horizon EVC and causal-credit audit for v2.4.2."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts.analyze_dynamic_v23_offline import spearman
from tdca_research.dynamic_v2.allocator import (
    AdaptiveComputationAllocator,
    EVCSignals,
)
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.credit import delayed_credit_snapshot
from tdca_research.dynamic_v2.graph import DynamicReasoningHypergraphV2
from tdca_research.utils import write_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values) -> float:
    rows = [float(value) for value in values]
    return fmean(rows) if rows else 0.0


def _signals(value: dict[str, Any]) -> EVCSignals:
    allowed = EVCSignals.__dataclass_fields__
    return EVCSignals(**{
        key: float(raw) for key, raw in (value or {}).items() if key in allowed
    })


def _actual_immediate(
    outcome: dict[str, Any], config: DynamicV2ResearchConfig, *, historical: bool = False,
) -> float:
    if not historical and "actual_immediate_utility" in outcome:
        return float(outcome["actual_immediate_utility"])
    normalized = outcome.get("actual_utility_components_normalized") or {}
    weights = {
        "uncertainty_reduction": config.actual_utility_weight_uncertainty,
        "support_gain": config.actual_utility_weight_support,
        "evidence_gap_reduction": config.actual_utility_weight_evidence_gap,
        "entropy_reduction": config.actual_utility_weight_entropy,
        "dependency_unlock_gain": config.actual_utility_weight_unlock,
        "evidence_novelty": config.actual_utility_weight_novelty,
        "answer_chain_progress": config.actual_utility_weight_chain_progress,
        "contradiction_resolution": config.actual_utility_weight_contradiction_resolution,
        "terminal_gap_reduction": config.actual_utility_weight_terminal_gap,
    }
    denominator = max(1e-12, sum(float(value) for value in weights.values()))
    return max(0.0, min(1.0, sum(
        float(weights[name]) * max(0.0, min(1.0, float(normalized.get(name, 0.0))))
        for name in weights
    ) / denominator))


def _choice_map(reasoning: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for event in reasoning:
        qid = str(event.get("qid", ""))
        if event.get("event") == "meta_decision" and event.get("outcome") == "CONTINUE":
            candidates = list(event.get("allocation_candidates") or [])
            keys = {
                (
                    str(row.get("operation_id", "")),
                    str(row.get("operation_family", "")),
                    str(row.get("region_key", "")),
                )
                for row in candidates
            }
            if candidates:
                pending[qid].append({
                    "operation_id": str(candidates[0].get("operation_id", "")),
                    "real_operation_choice": len(keys) > 1,
                    "choice_size": len(keys),
                })
        elif event.get("event") == "allocation_reconciled":
            packet = event.get("allocation") or {}
            choice = pending[qid].pop(0) if pending[qid] else {}
            selected = str(choice.get("operation_id", ""))
            executed = str(packet.get("operation_id", ""))
            matched = bool(selected and (
                executed == selected or executed.startswith(f"{selected}_allocation_")
            ))
            result[(qid, str(packet.get("allocation_id", "")))] = {
                "real_operation_choice": bool(
                    matched and choice.get("real_operation_choice", False)
                ),
                "choice_size": int(choice.get("choice_size", 0)) if matched else 0,
                "choice_trace_matched": matched,
            }
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "spearman_predicted_immediate_actual_immediate": spearman(
            [row["predicted_immediate_utility"] for row in rows],
            [row["actual_immediate_utility"] for row in rows],
        ),
        "spearman_predicted_delayed_actual_delayed": spearman(
            [row["predicted_delayed_proof_return"] for row in rows],
            [row["delayed_realized_proof_return"] for row in rows],
        ),
        "spearman_predicted_evc_combined_utility": spearman(
            [row["predicted_evc"] for row in rows],
            [row["combined_realized_utility"] for row in rows],
        ),
        "mean_predicted_immediate_utility": _mean(
            row["predicted_immediate_utility"] for row in rows
        ),
        "mean_actual_immediate_utility": _mean(
            row["actual_immediate_utility"] for row in rows
        ),
        "mean_predicted_delayed_proof_return": _mean(
            row["predicted_delayed_proof_return"] for row in rows
        ),
        "mean_delayed_realized_proof_return": _mean(
            row["delayed_realized_proof_return"] for row in rows
        ),
    }


def analyze(run: Path, config: DynamicV2ResearchConfig) -> dict[str, Any]:
    reasoning = _jsonl(run / "reasoning_traces.jsonl")
    choices = _choice_map(reasoning)
    rows: list[dict[str, Any]] = []
    for graph_row in _jsonl(run / "dynamic_v2_graphs.jsonl"):
        qid = str(graph_row.get("qid", ""))
        payload = dict(graph_row.get("graph", graph_row.get("graph_snapshot", graph_row)))
        historical = not bool(payload.get("credit_assignment_history"))
        if historical:
            payload["controller_state_hash"] = ""
        graph = DynamicReasoningHypergraphV2.from_dict(payload)
        outcomes = {
            row.allocation_id: row for row in graph.operation_outcome_history
        }
        for allocation in graph.allocation_history:
            outcome = outcomes.get(allocation.allocation_id)
            if outcome is None:
                continue
            normalized = _signals(allocation.evc_components_normalized)
            replay_immediate, replay_delayed, replay_cost = (
                AdaptiveComputationAllocator(config)._horizon_scores(
                    normalized, outcome.operation_family,
                )
            )
            if historical:
                predicted_immediate = replay_immediate
                predicted_delayed = replay_delayed
                predicted_cost = replay_cost
                predicted_evc = max(0.0, (
                    config.evc_immediate_horizon_weight * predicted_immediate
                    + config.evc_delayed_horizon_weight * predicted_delayed
                    - predicted_cost
                ))
                credit = delayed_credit_snapshot(graph, allocation, config)
                delayed_actual = credit.delayed_realized_proof_return
                delayed_components = credit.delayed_components_normalized
            else:
                predicted_immediate = allocation.predicted_immediate_utility
                predicted_delayed = allocation.predicted_delayed_proof_return
                predicted_cost = allocation.predicted_normalized_cost
                predicted_evc = allocation.predicted_evc
                delayed_actual = allocation.delayed_realized_proof_return
                latest = next((
                    value for value in reversed(graph.credit_assignment_history)
                    if value.allocation_id == allocation.allocation_id
                ), None)
                delayed_components = (
                    latest.delayed_components_normalized if latest is not None else {}
                )
            immediate_actual = _actual_immediate(
                outcome.__dict__, config, historical=historical,
            )
            actual_cost = float(getattr(outcome, "actual_normalized_cost", 0.0))
            if historical:
                actual_cost = float(
                    outcome.actual_utility_components_normalized.get("cost", 0.0)
                )
            combined = max(-1.0, min(1.0, (
                config.evc_immediate_horizon_weight * immediate_actual
                + config.evc_delayed_horizon_weight * delayed_actual
                - actual_cost
            )))
            raw_evc = allocation.evc_components_raw or {}
            choice = choices.get((qid, allocation.allocation_id), {})
            successful_recovery = any(float(delayed_components.get(name, 0.0)) > 0.0 for name in (
                "candidate_availability", "successful_join", "supported_terminal_answer",
            ))
            rows.append({
                "qid": qid,
                "allocation_id": allocation.allocation_id,
                "operation_id": allocation.operation_id,
                "operation_family": outcome.operation_family,
                "predicted_evc": predicted_evc,
                "predicted_immediate_utility": predicted_immediate,
                "predicted_delayed_proof_return": predicted_delayed,
                "predicted_normalized_cost": predicted_cost,
                "actual_immediate_utility": immediate_actual,
                "actual_normalized_cost": actual_cost,
                "delayed_realized_proof_return": delayed_actual,
                "combined_realized_utility": combined,
                "proof_gap_operation": (
                    float(raw_evc.get("proof_gap_reducibility", 0.0)) > 0.0
                    or float(raw_evc.get("feasibility_unlock", 0.0)) > 0.0
                ),
                "successful_recovery": successful_recovery,
                **choice,
            })
    choice_rows = [row for row in rows if row.get("real_operation_choice")]
    proof_gap = [row for row in rows if row["proof_gap_operation"]]
    successful = [row for row in proof_gap if row["successful_recovery"]]
    failed = [row for row in proof_gap if not row["successful_recovery"]]
    by_family = {
        family: _summary([row for row in rows if row["operation_family"] == family])
        for family in sorted({row["operation_family"] for row in rows})
    }
    return {
        "schema_version": "dynamic-hypergraph-v2.4.2-horizon-credit-audit-v1",
        "run": str(run),
        "gold_used": False,
        "attribution_rule": "provenance_descendants_only",
        "gamma": config.delayed_credit_gamma,
        "overall": _summary(rows),
        "choice_conditioned": {
            **_summary(choice_rows),
            "trace_match_rate": (
                sum(bool(row.get("choice_trace_matched")) for row in rows) / len(rows)
                if rows else 0.0
            ),
        },
        "by_operation_family": by_family,
        "proof_gap_recovery": {
            "count": len(proof_gap),
            "successful_count": len(successful),
            "failed_count": len(failed),
            "mean_successful_delayed_return": _mean(
                row["delayed_realized_proof_return"] for row in successful
            ),
            "mean_failed_delayed_return": _mean(
                row["delayed_realized_proof_return"] for row in failed
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/dynamic_hypergraph_v242_qwen_smoke20.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = DynamicV2ResearchConfig.from_yaml(args.config)
    config.validate()
    report = analyze(args.run, config)
    write_json(args.output, report)
    print(json.dumps({
        "run": str(args.run), "count": report["overall"]["count"],
        "overall": report["overall"],
        "choice_conditioned": report["choice_conditioned"],
        "proof_gap_recovery": report["proof_gap_recovery"],
        "family_correlations": {
            key: {
                "count": value["count"],
                "immediate": value[
                    "spearman_predicted_immediate_actual_immediate"
                ],
                "delayed": value[
                    "spearman_predicted_delayed_actual_delayed"
                ],
                "mean_predicted_delayed": value[
                    "mean_predicted_delayed_proof_return"
                ],
                "mean_actual_delayed": value[
                    "mean_delayed_realized_proof_return"
                ],
            }
            for key, value in report["by_operation_family"].items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
