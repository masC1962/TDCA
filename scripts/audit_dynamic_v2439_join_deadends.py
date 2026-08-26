#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.engine import (
    _join_attempt_key, _join_can_answer_subgoal, _nary_relevant,
)
from tdca_research.dynamic_v2.graph import DynamicReasoningHypergraphV2
from tdca_research.dynamic_v2.join import MultiHopJoinEngine
from tdca_research.utils import write_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(run: Path, config: DynamicV2ResearchConfig) -> dict[str, Any]:
    trace_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(run / "reasoning_traces.jsonl"):
        trace_by_qid[str(row["qid"])].append(row)
    engine = MultiHopJoinEngine(None, None, config)  # type: ignore[arg-type]
    rows = []
    for item in _jsonl(run / "dynamic_v2_graphs.jsonl"):
        qid = str(item["qid"])
        graph = DynamicReasoningHypergraphV2.from_dict(item["graph"])
        filtered = {
            str(candidate["join_attempt_key"]): {
                "step": int(event["step"]),
                "premise_ids": list(candidate["premise_ids"]),
                "reason_codes": list(candidate["reason_codes"]),
            }
            for event in trace_by_qid[qid]
            if event.get("event") == "join_preallocation_filtered"
            for candidate in event.get("filtered", [])
        }
        rejected = {
            (str(event.get("join_signature", "")), tuple(event.get("premise_ids", []))): {
                "diagnostics": dict(event.get("diagnostics") or {}),
            }
            for event in trace_by_qid[qid]
            if event.get("event") == "join_rejected"
        }
        for obligation in graph.proof_obligations.values():
            if obligation.status != "OPEN" or obligation.obligation_type != "missing_join_premise":
                continue
            subgoal = graph.node(obligation.target_subgoal)
            branch = graph.branches[obligation.branch_id]
            dependency_ids = {
                branch.assignments[value]
                for value in subgoal.dependencies if value in branch.assignments
            }
            for candidate in engine.discover(
                graph, branch.branch_id, subgoal.node_id,
            ):
                if not engine.check_feasible(graph, candidate).feasible:
                    continue
                if not _nary_relevant(graph, candidate, dependency_ids):
                    continue
                if not _join_can_answer_subgoal(
                    graph, candidate, subgoal, branch,
                    numeric_aliases=config.numeric_output_type_normalization,
                ):
                    continue
                key = _join_attempt_key(graph, candidate)
                predicted_provider_calls = engine.predicted_provider_calls(
                    graph, candidate,
                )
                rows.append({
                    "qid": qid,
                    "target_subgoal": subgoal.node_id,
                    "branch_id": branch.branch_id,
                    "join_attempt_key": key,
                    "signature": candidate.signature,
                    "premise_ids": list(candidate.premise_ids),
                    "premise_versions": {
                        value: graph.belief_states[value].version
                        for value in candidate.premise_ids
                    },
                    "predicted_provider_calls": predicted_provider_calls,
                    "certified_provider_free": predicted_provider_calls == 0,
                    "previously_filtered_same_key": key in filtered,
                    "previous_filter": filtered.get(key),
                    "previously_rejected_same_signature": (
                        candidate.signature, candidate.premise_ids
                    ) in rejected,
                    "previous_rejection": rejected.get((
                        candidate.signature, candidate.premise_ids,
                    )),
                })
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3.9-join-deadend-audit-v1",
        "source_run": str(run),
        "gold_used": False,
        "final_viable_join_count": len(rows),
        "same_key_previously_filtered_count": sum(
            row["previously_filtered_same_key"] for row in rows
        ),
        "same_signature_previously_rejected_count": sum(
            row["previously_rejected_same_signature"] for row in rows
        ),
        "certified_provider_free_join_count": sum(
            row["certified_provider_free"] for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, DynamicV2ResearchConfig.from_yaml(args.config))
    write_json(args.output, report)
    print(json.dumps({
        "final_viable_join_count": report["final_viable_join_count"],
        "same_key_previously_filtered_count": report[
            "same_key_previously_filtered_count"
        ],
        "same_signature_previously_rejected_count": report[
            "same_signature_previously_rejected_count"
        ],
        "certified_provider_free_join_count": report[
            "certified_provider_free_join_count"
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
