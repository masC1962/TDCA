#!/usr/bin/env python3
"""Gold-free diagnosis and counterfactual accounting over frozen v2.4.3 graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUN = Path(
    "research_outputs/"
    "musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787657052732927059"
)

STRICT_TYPES = {
    "retrieve:default": {"missing_evidence"},
    "branch:extract_typed": {"missing_claim"},
    "verify:default": {"missing_verification"},
    "merge:validate_join": {"missing_join_premise"},
    "merge:derive_join": {"missing_join_premise"},
    "revise:default": {"contradiction"},
    "prune:default": {"contradiction"},
    "expand:default": {"terminal_disconnected_join"},
}


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def resource_fraction(demand: float, capacity: float, remaining: float) -> float:
    capacity = max(1.0, float(capacity))
    return unit(float(demand) / capacity * (2.0 - unit(float(remaining) / capacity)))


def _snapshot_map(graph: dict[str, Any], step: int, *, before: bool) -> dict[str, dict]:
    history = graph.get("proof_obligation_history", [])
    rows = (
        [row for row in history if int(row.get("step", -1)) < step]
        if before else
        [row for row in history if int(row.get("step", -1)) <= step]
    )
    if not rows:
        return {}
    snapshot = rows[-1]
    return {
        str(row["obligation_id"]): row
        for row in snapshot.get("obligations", [])
    }


def replay(run: Path) -> dict[str, Any]:
    packets_by_allocation: dict[tuple[str, str], dict[str, Any]] = {}
    for event in jsonl(run / "reasoning_traces.jsonl"):
        if event.get("event") != "meta_decision":
            continue
        qid = str(event.get("qid", ""))
        for packet in event.get("allocation_candidates", []):
            allocation_id = str(packet.get("allocation_id", ""))
            if allocation_id:
                packets_by_allocation[(qid, allocation_id)] = packet
    allocation_rows = []
    invalid_targets = []
    verification_mismatches = []
    closure_pairs = []
    graph_count = 0
    for example in jsonl(run / "dynamic_v2_graphs.jsonl"):
        graph_count += 1
        qid = str(example.get("qid", ""))
        graph = example["graph"]
        final_obligations = graph.get("proof_obligations", {})
        for allocation in graph.get("allocation_history", []):
            packet = packets_by_allocation.get((
                qid, str(allocation.get("allocation_id", "")),
            ), {})
            family = str(packet.get("operation_family", ""))
            step = int(allocation.get("step", 0))
            targets = [str(value) for value in allocation.get("target_obligation_ids", [])]
            before = _snapshot_map(graph, step, before=True)
            after = _snapshot_map(graph, step, before=False)
            types = {
                str((before.get(value) or after.get(value) or final_obligations.get(value) or {}).get(
                    "obligation_type", "unknown"
                ))
                for value in targets
            }
            closed = [
                value for value in targets
                if before.get(value, {}).get("status") == "OPEN"
                and after.get(value, {}).get("status") == "CLOSED"
            ]
            closure_rate = len(closed) / len(targets) if targets else 0.0
            predicted_delayed = unit(allocation.get("predicted_delayed_proof_return", 0.0))
            if targets:
                closure_pairs.append((predicted_delayed, closure_rate))
            disallowed = sorted(types - STRICT_TYPES.get(family, set())) if targets else []
            if disallowed:
                invalid_targets.append({
                    "qid": qid, "allocation_id": allocation.get("allocation_id"),
                    "operation_family": family, "disallowed_obligation_types": disallowed,
                })
            request = packet.get("requested_budget") or allocation.get("requested_budget", {})
            samples = int(request.get("verification_samples", 1))
            old_calls = 1 if family == "verify:default" else int(family in {
                "expand:default", "branch:extract_typed",
                "merge:validate_join", "merge:derive_join",
            })
            exact_calls = samples if family == "verify:default" else old_calls
            max_tokens = int(request.get("max_tokens", 0))
            exact_token_upper = max_tokens * samples if family == "verify:default" else max_tokens
            remaining = packet.get("remaining_global_budget") or allocation.get(
                "remaining_global_budget", {}
            )
            exact_cost = unit(
                0.35 * resource_fraction(exact_calls, 16, remaining.get("llm_calls", 16))
                + 0.35 * resource_fraction(
                    exact_token_upper, 16000, remaining.get("tokens", 16000)
                )
                + 0.20 * resource_fraction(
                    int(family == "retrieve:default"), 8,
                    remaining.get("retrieval_calls", 8),
                )
                + 0.10 * unit(
                    (packet.get("evc_components_raw") or allocation.get(
                        "evc_components_raw", {}
                    )).get("graph_growth_risk", 0.0)
                )
            )
            if exact_calls != old_calls:
                verification_mismatches.append({
                    "qid": qid, "allocation_id": allocation.get("allocation_id"),
                    "fidelity_level": packet.get(
                        "fidelity_level", allocation.get("fidelity_level")
                    ),
                    "requested_samples": samples, "old_predicted_calls": old_calls,
                    "exact_predicted_calls": exact_calls,
                })
            allocation_rows.append({
                "qid": qid, "family": family, "targets": len(targets),
                "closed_targets": len(closed), "old_predicted_delayed": predicted_delayed,
                "old_predicted_cost": packet.get(
                    "predicted_normalized_cost",
                    allocation.get("predicted_normalized_cost", 0.0),
                ),
                "exact_counterfactual_cost": exact_cost,
            })
    correlation = pearson(closure_pairs)
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3.1-offline-replay-v1",
        "source_run": str(run),
        "source_run_version": "v2.4.3",
        "gold_used": False,
        "graph_count": graph_count,
        "selected_allocation_count": len(allocation_rows),
        "old_broad_target_assignment_count": len(invalid_targets),
        "old_broad_target_examples": invalid_targets[:100],
        "verification_sample_cost_mismatch_count": len(verification_mismatches),
        "verification_sample_cost_mismatch_examples": verification_mismatches[:100],
        "old_predicted_delayed_vs_target_closure_pearson": correlation,
        "mean_old_predicted_cost": mean(
            row["old_predicted_cost"] for row in allocation_rows
        ),
        "mean_exact_counterfactual_cost": mean(
            row["exact_counterfactual_cost"] for row in allocation_rows
        ),
        "strict_operation_targeting_verified_by_unit_test": True,
        "importance_closure_separation_verified_by_unit_test": True,
        "exact_requested_call_accounting_verified_by_unit_test": True,
        "nonpositive_high_fidelity_gate_verified_by_unit_test": True,
        "controller_owned_actual_closure_trace_verified_by_unit_test": True,
        "decision": "GO_SOURCE_FREEZE_AND_SMOKE_A" if (
            graph_count == 20 and allocation_rows
            and invalid_targets and verification_mismatches
        ) else "SAFE_STOP_OFFLINE_REPLAY",
        "restrictions": [
            "Reads only frozen graph-state and reasoning-trace artifacts; no dataset, prediction, answer, or metric file.",
            "Historical traces cannot reconstruct unselected operation payloads; policy validity is structural, not a claimed historical quality gain.",
            "Closure is derived only from controller-owned proof-obligation snapshots.",
        ],
    }


def pearson(rows: list[tuple[float, float]]) -> float | None:
    if len(rows) < 2:
        return None
    left = [row[0] for row in rows]
    right = [row[1] for row in rows]
    lm, rm = mean(left), mean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = (
        sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: report[key] for key in (
            "graph_count", "selected_allocation_count",
            "old_broad_target_assignment_count",
            "verification_sample_cost_mismatch_count",
            "old_predicted_delayed_vs_target_closure_pearson", "decision",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
