#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from scripts.analyze_dynamic_v23_offline import spearman
from scripts.analyze_dynamic_v242_offline import analyze as analyze_v242
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.utils import write_json


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def analyze(run: Path, config: DynamicV2ResearchConfig) -> dict:
    base = analyze_v242(run, config)
    allocations = {
        (str(example["qid"]), str(row["allocation_id"])): row
        for example in _jsonl(run / "dynamic_v2_graphs.jsonl")
        for row in example["graph"].get("allocation_history", [])
    }
    rows = []
    for outcome in base["rows"]:
        key = (str(outcome["qid"]), str(outcome["allocation_id"]))
        allocation = allocations[key]
        components = allocation.get("evc_components_normalized") or {}
        certificate = allocation.get("transition_certificate") or {}
        transition = float(allocation.get("predicted_transition_value", 0.0))
        nonterminal_transition = (
            transition
            if certificate.get("kind") != "accepted_terminal_materialization"
            else 0.0
        )
        local_components = [
            float(components.get("obligation_closure", 0.0)),
            float(components.get("terminal_reachability", 0.0)),
            float(components.get("missing_premise_reduction", 0.0)),
            float(components.get("candidate_reachability", 0.0)),
            float(components.get("evidence_path", 0.0)),
        ]
        conditioned = (
            float(components.get("obligation_importance", 0.0))
            * float(components.get("operation_closure_probability", 0.0))
            * float(components.get("expected_obligation_delta", 0.0))
            * float(components.get("obligation_terminal_return", 0.0))
            - float(components.get("operation_redundancy", 0.0))
        )
        candidates = {
            "recorded": float(outcome["predicted_delayed_proof_return"]),
            "recorded_max_nonterminal_transition": max(
                float(outcome["predicted_delayed_proof_return"]),
                nonterminal_transition,
            ),
            "conditioned_plus_nonterminal_transition": max(
                _unit(conditioned), nonterminal_transition,
            ),
            "local_equal_mean": fmean(local_components),
            "local_equal_mean_plus_nonterminal_transition": max(
                fmean(local_components), nonterminal_transition,
            ),
            "closure_mass": _unit(
                float(components.get("operation_closure_probability", 0.0))
                * float(components.get("expected_obligation_delta", 0.0))
            ),
        }
        observed = float(components.get("observed_value", 0.5))
        cooldown = float(components.get("failure_cooldown", 0.0))
        redundancy = float(components.get("operation_redundancy", 0.0))
        dead_end = float(components.get("dead_end_risk", 0.0))
        candidates.update({
            "recorded_x_observed": _unit(
                candidates["recorded"] * observed
            ),
            "recorded_x_feedback": _unit(
                candidates["recorded"] * observed
                * (1.0 - cooldown) * (1.0 - redundancy)
            ),
            "closure_mass_x_observed": _unit(
                candidates["closure_mass"] * observed
            ),
            "closure_mass_x_feedback": _unit(
                candidates["closure_mass"] * observed
                * (1.0 - cooldown) * (1.0 - redundancy)
                * (1.0 - dead_end)
            ),
        })
        rows.append({
            **outcome,
            "candidate_predictions": candidates,
            "certificate_kind": str(certificate.get("kind", "")),
            "predicted_transition_value": transition,
        })
    names = list(rows[0]["candidate_predictions"]) if rows else []
    correlations = {}
    for name in names:
        all_rows = rows
        choice_rows = [row for row in rows if row.get("real_operation_choice")]
        correlations[name] = {
            "overall_spearman": spearman(
                [row["candidate_predictions"][name] for row in all_rows],
                [row["delayed_realized_proof_return"] for row in all_rows],
            ),
            "choice_conditioned_spearman": spearman(
                [row["candidate_predictions"][name] for row in choice_rows],
                [row["delayed_realized_proof_return"] for row in choice_rows],
            ),
            "choice_count": len(choice_rows),
        }
    return {
        "schema_version": "dynamic-v2.4.3.5-offline-delayed-candidate-audit-v1",
        "source_run": str(run),
        "gold_used": False,
        "correlations": correlations,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run, DynamicV2ResearchConfig.from_yaml(args.config))
    write_json(args.output, report)
    print(json.dumps(report["correlations"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
