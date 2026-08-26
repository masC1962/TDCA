#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tdca_research.utils import write_json


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze(run: Path) -> dict:
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in _jsonl(run / "reasoning_traces.jsonl"):
        by_qid[str(row.get("qid", ""))].append(row)

    accepted_readouts = []
    blocked_materializations = []
    for qid, events in sorted(by_qid.items()):
        for index, event in enumerate(events):
            if event.get("event") != "terminal_belief_readout":
                continue
            answer_ids = [
                str(value) for value in event.get("accepted_answer_node_ids", [])
            ]
            if not answer_ids:
                continue
            accepted_readouts.append({
                "qid": qid,
                "step": int(event.get("step", -1)),
                "answer_node_ids": answer_ids,
            })
            following = next((
                row for row in events[index + 1:]
                if row.get("event") == "meta_decision"
            ), None)
            if following is None:
                continue
            commits = [
                row for row in following.get("allocation_candidates", [])
                if row.get("operation_family") == "commit:answer"
            ]
            if commits and following.get("outcome") == "ABSTAIN":
                blocked_materializations.append({
                    "qid": qid,
                    "step": int(following.get("step", -1)),
                    "reason": str(following.get("reason", "")),
                    "answer_node_ids": answer_ids,
                    "allocation_ids": [str(row.get("allocation_id", "")) for row in commits],
                    "provider_calls": [
                        int(row.get("requested_budget", {}).get("llm_calls", -1))
                        for row in commits
                    ],
                    "predicted_evc": [float(row.get("predicted_evc", 0.0)) for row in commits],
                })

    return {
        "schema_version": "dynamic-v2.4.3.4-offline-terminal-replay-v1",
        "source_run": str(run),
        "gold_used": False,
        "accepted_terminal_readout_count": len(accepted_readouts),
        "accepted_terminal_readouts": accepted_readouts,
        "blocked_terminal_materialization_count": len(blocked_materializations),
        "blocked_terminal_materializations": blocked_materializations,
        "all_blocked_are_zero_provider": bool(blocked_materializations) and all(
            calls == 0
            for row in blocked_materializations
            for calls in row["provider_calls"]
        ),
        "policy_delta": {
            "certified_terminal_materialization": True,
            "recompute_absolute_channels_from_sealed_graph": True,
            "recompute_relative_channels_from_complete_competition_snapshot": True,
            "threshold_changed": False,
            "terminal_gate_changed": False,
            "training": False,
            "per_question_patch": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run)
    write_json(args.output, report)
    print(json.dumps({
        "accepted_terminal_readout_count": report["accepted_terminal_readout_count"],
        "blocked_terminal_materialization_count": report["blocked_terminal_materialization_count"],
        "all_blocked_are_zero_provider": report["all_blocked_are_zero_provider"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
