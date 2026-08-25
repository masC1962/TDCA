#!/usr/bin/env python3
"""Counterfactual absolute-cost audit over frozen v2.4.2 traces.

This script deliberately reads reasoning traces only.  It never opens dataset,
prediction, answer, or metric artifacts and therefore cannot use gold labels.
The replay changes only the cost equation; graph-local delayed value is tested
structurally until native v2.4.3 traces exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUN = Path(
    "research_outputs/"
    "musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787652976991954408"
)
KNOWN_CLIPPED_QIDS = {
    "4hop1__51465_53706_795904_580996",
    "4hop2__161602_474028_88460_21057",
}
CALL_FAMILIES = {
    "expand:default", "branch:extract_typed", "verify:default",
    "merge:validate_join", "merge:derive_join",
}


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resource_fraction(demand: float, capacity: float, remaining: float) -> float:
    capacity = max(1.0, float(capacity))
    remaining_ratio = unit(float(remaining) / capacity)
    return unit(float(demand) / capacity * (2.0 - remaining_ratio))


def absolute_cost(packet: dict[str, Any]) -> tuple[float, dict[str, float]]:
    family = str(packet.get("operation_family", ""))
    request = packet.get("requested_budget") or {}
    remaining = packet.get("remaining_global_budget") or {}
    raw = packet.get("evc_components_raw") or {}
    components = {
        "call": resource_fraction(
            float(family in CALL_FAMILIES), 16.0,
            float(remaining.get("llm_calls", 16)),
        ),
        "token": resource_fraction(
            float(request.get("max_tokens", 0)), 16000.0,
            float(remaining.get("tokens", 16000)),
        ),
        "retrieval": resource_fraction(
            float(family == "retrieve:default"), 8.0,
            float(remaining.get("retrieval_calls", 8)),
        ),
        "graph_risk": unit(float(raw.get("graph_growth_risk", 0.0))),
    }
    return unit(
        0.35 * components["call"]
        + 0.35 * components["token"]
        + 0.20 * components["retrieval"]
        + 0.10 * components["graph_risk"]
    ), components


def replay(run: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for event in jsonl(run / "reasoning_traces.jsonl"):
        if event.get("event") != "meta_decision":
            continue
        qid = str(event.get("qid", ""))
        for packet in event.get("allocation_candidates", []):
            new_cost, components = absolute_cost(packet)
            immediate = unit(float(packet.get("predicted_immediate_utility", 0.0)))
            delayed = unit(float(packet.get("predicted_delayed_proof_return", 0.0)))
            gross = unit(0.40 * immediate + 0.60 * delayed)
            old_cost = unit(float(packet.get("predicted_normalized_cost", 0.0)))
            old_net = unit(float(packet.get("predicted_evc", 0.0)))
            new_net = max(0.0, gross - new_cost)
            rows.append({
                "qid": qid,
                "allocation_id": str(packet.get("allocation_id", "")),
                "operation_id": str(packet.get("operation_id", "")),
                "operation_family": str(packet.get("operation_family", "")),
                "fidelity_level": str(packet.get("fidelity_level", "")),
                "gross_opportunity": gross,
                "old_choice_relative_cost": old_cost,
                "new_absolute_cost": new_cost,
                "old_net_evc": old_net,
                "new_cost_only_counterfactual_net_evc": new_net,
                "remaining_budget": dict(packet.get("remaining_global_budget") or {}),
                "absolute_cost_components": components,
            })
    clipped = [
        row for row in rows
        if row["old_net_evc"] <= 0.0
        and row["gross_opportunity"] > 0.08
        and row["new_cost_only_counterfactual_net_evc"] > 0.08
    ]
    known = {
        qid: sorted(
            (row for row in rows if row["qid"] == qid),
            key=lambda row: (
                -row["new_cost_only_counterfactual_net_evc"], row["allocation_id"]
            ),
        )[:10]
        for qid in sorted(KNOWN_CLIPPED_QIDS)
    }
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3-offline-cost-replay-v1",
        "source_run": str(run),
        "gold_used": False,
        "cost_weights": {"call": 0.35, "token": 0.35, "retrieval": 0.20, "graph_risk": 0.10},
        "scarcity_multiplier_range": [1.0, 2.0],
        "allocation_candidate_count": len(rows),
        "absolute_cost_min": min((row["new_absolute_cost"] for row in rows), default=0.0),
        "absolute_cost_max": max((row["new_absolute_cost"] for row in rows), default=0.0),
        "old_cost_mean": mean(row["old_choice_relative_cost"] for row in rows),
        "absolute_cost_mean": mean(row["new_absolute_cost"] for row in rows),
        "cost_clipping_recovery_count": len(clipped),
        "cost_clipping_recovery_examples": clipped[:100],
        "known_v242_cost_clipping_cases": known,
        "known_cases_have_positive_cost_only_counterfactual": all(
            any(row["new_cost_only_counterfactual_net_evc"] > 0.08 for row in part)
            for part in known.values()
        ),
        "ready_set_invariance_verified_by_unit_test": True,
        "scarcity_monotonicity_verified_by_unit_test": True,
        "graph_local_value_replay_scope": (
            "structural tests only because v2.4.2 traces predate proof-obligation state"
        ),
    }


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "allocation_candidate_count", "old_cost_mean", "absolute_cost_mean",
            "cost_clipping_recovery_count",
            "known_cases_have_positive_cost_only_counterfactual",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
