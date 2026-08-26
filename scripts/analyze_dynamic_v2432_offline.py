#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tdca_research.dynamic.graph import GraphOperation, OperationType
from tdca_research.dynamic_v2.graph import DynamicReasoningHypergraphV2
from tdca_research.dynamic_v2.transitions import certified_transition_value


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def analyze(run_dir: Path) -> dict[str, Any]:
    last_meta: dict[str, dict[str, Any]] = {}
    for row in _jsonl(run_dir / "reasoning_traces.jsonl"):
        if row.get("event") == "meta_decision":
            last_meta[str(row["qid"])] = row
    graphs = {
        str(row["qid"]): DynamicReasoningHypergraphV2.from_dict(row["graph"])
        for row in _jsonl(run_dir / "dynamic_v2_graphs.jsonl")
    }
    audited = []
    for qid, decision in sorted(last_meta.items()):
        if decision.get("reason") != "affordable_proof_opportunity_below_net_value_threshold":
            continue
        graph = graphs[qid]
        candidates = decision.get("allocation_candidates", [])
        certified = []
        for row in candidates:
            family = str(row.get("operation_family", ""))
            region = [str(value) for value in row.get("target_region", [])]
            target_id = region[0] if region else ""
            branch_id = region[1] if len(region) > 1 else ""
            sources = region[2:]
            if family == "commit:default":
                kind = OperationType.COMMIT
                payload = {"candidate_id": sources[0] if sources else ""}
            elif family == "branch:assignments":
                kind = OperationType.BRANCH
                payload = {"mode": "assignments", "candidate_ids": sources}
            else:
                continue
            operation = GraphOperation(
                str(row["operation_id"]), kind, target_id, sources, branch_id,
                payload, "historical_counterfactual", "v2432_offline_replay",
                {"llm_calls": 0.0, "tokens": 0.0},
            )
            certificate = certified_transition_value(graph, operation)
            if certificate.get("mandatory"):
                certified.append({
                    "allocation_id": row["allocation_id"],
                    "operation_family": family,
                    "old_predicted_evc": row["predicted_evc"],
                    "transition_certificate": certificate,
                })
        audited.append({
            "qid": qid,
            "old_reason": decision["reason"],
            "old_best_predicted_evc": decision["best_predicted_evc"],
            "certified_transitions": certified,
            "unlocked": bool(certified),
        })
    unlocked = [row for row in audited if row["unlocked"]]
    return {
        "analysis_version": "dynamic-v2.4.3.2-certified-transition-replay-v1",
        "gold_used": False,
        "source_run": str(run_dir),
        "threshold_stop_count": len(audited),
        "certified_transition_unlocked_count": len(unlocked),
        "still_stopped_count": len(audited) - len(unlocked),
        "all_unlocked_are_zero_provider": all(
            transition["transition_certificate"]["provider_calls"] == 0
            for row in unlocked for transition in row["certified_transitions"]
        ),
        "unlocked_by_family": {
            family: sum(
                transition["operation_family"] == family
                for row in unlocked for transition in row["certified_transitions"]
            )
            for family in sorted({
                transition["operation_family"]
                for row in unlocked for transition in row["certified_transitions"]
            })
        },
        "rows": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
