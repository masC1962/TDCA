#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tdca_research.dynamic.graph import ClaimNode, SubgoalNode
from tdca_research.dynamic_v2.graph import DynamicReasoningHypergraphV2
from tdca_research.dynamic_v2.verifier import _structural_projection
from tdca_research.utils import write_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(run: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in _jsonl(run / "dynamic_v2_graphs.jsonl"):
        graph = DynamicReasoningHypergraphV2.from_dict(item["graph"])
        for claim in graph.nodes.values():
            if not isinstance(claim, ClaimNode):
                continue
            subgoal = graph.nodes.get(claim.target_subgoal)
            if not isinstance(subgoal, SubgoalNode):
                continue
            old = _structural_projection(
                graph, subgoal.node_id, claim, subgoal.answer_type,
                numeric_aliases=False,
            )
            new = _structural_projection(
                graph, subgoal.node_id, claim, subgoal.answer_type,
                numeric_aliases=True,
            )
            if old == new:
                continue
            semantics = graph.claim_semantics[claim.node_id]
            rows.append({
                "qid": str(item["qid"]),
                "claim_id": claim.node_id,
                "subgoal_id": subgoal.node_id,
                "branch_id": claim.branch_id,
                "subject": claim.subject,
                "relation": claim.relation,
                "value": claim.value,
                "dependency_claim_ids": list(claim.dependency_claim_ids),
                "source_spans": list(
                    claim.provenance.metadata.get("source_spans", [])
                ),
                "expected_type": subgoal.answer_type,
                "subject_type": semantics.subject_type,
                "value_type": semantics.value_type,
                "old_projection": old,
                "new_projection": new,
                "absolute_support": claim.score.absolute_support,
                "grounding": claim.score.raw.grounding,
                "type_match": claim.score.raw.type_match,
                "evidence_gap": claim.score.evidence_gap,
            })
    qids = sorted({row["qid"] for row in rows})
    supported = [
        row for row in rows
        if row["absolute_support"] >= 0.55
        and row["grounding"] >= 0.55
        and row["type_match"] >= 0.55
    ]
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3.9-numeric-projection-audit-v1",
        "source_run": str(run),
        "gold_used": False,
        "new_projection_count": len(rows),
        "affected_qid_count": len(qids),
        "affected_qids": qids,
        "independently_supported_new_projection_count": len(supported),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run)
    write_json(args.output, report)
    print(json.dumps({
        key: report[key] for key in (
            "new_projection_count", "affected_qid_count",
            "affected_qids", "independently_supported_new_projection_count",
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
