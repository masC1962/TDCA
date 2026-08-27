#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(run: Path) -> dict[str, Any]:
    metrics = {
        str(row["qid"]): row
        for row in _jsonl(run / "dynamic_v2_per_example_metrics.jsonl")
    }
    predictions = {
        str(row["qid"]): row for row in _jsonl(run / "predictions.jsonl")
    }
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    interval_ids_by_qid: dict[str, set[str]] = {}
    for item in _jsonl(run / "dynamic_v2_graphs.jsonl"):
        qid = str(item["qid"])
        graph = item["graph"]
        nodes = graph.get("nodes", {})
        semantics = graph.get("claim_semantics", {})
        interval_ids = [
            node_id for node_id, value in semantics.items()
            if value.get("extraction_mode") == "grounded_numeric_interval_consolidation"
        ]
        if not interval_ids:
            continue
        interval_ids_by_qid[qid] = set(interval_ids)
        for claim_id in interval_ids:
            claim = nodes[claim_id]
            metadata = claim.get("provenance", {}).get("metadata", {})
            qualifiers = metadata.get("typed_qualifiers", {}).get(
                "numeric_interval_consolidation", {}
            )
            spans = [str(value) for value in metadata.get("source_spans", [])]
            surface = str(qualifiers.get("surface", ""))
            direct_children = [
                node_id for node_id, node in nodes.items()
                if claim_id in node.get("dependency_claim_ids", [])
            ]
            descendants = set(direct_children)
            changed = True
            while changed:
                changed = False
                for node_id, node in nodes.items():
                    if node_id in descendants:
                        continue
                    if descendants.intersection(node.get("dependency_claim_ids", [])):
                        descendants.add(node_id)
                        changed = True
            exact = bool(surface) and any(surface in span for span in spans)
            endpoint_ids = [str(value) for value in qualifiers.get("endpoint_claim_ids", [])]
            if not exact:
                violations.append({"qid": qid, "claim_id": claim_id, "reason": "surface_not_exact"})
            if len(endpoint_ids) != 2 or len(set(endpoint_ids)) != 2:
                violations.append({"qid": qid, "claim_id": claim_id, "reason": "endpoint_audit_invalid"})
            rows.append({
                "qid": qid,
                "claim_id": claim_id,
                "subject": claim.get("subject"),
                "relation": claim.get("relation"),
                "value": claim.get("value"),
                "value_type": semantics[claim_id].get("value_type"),
                "absolute_support": claim.get("score", {}).get("absolute_support"),
                "dependency_consistency": claim.get("score", {}).get("raw", {}).get(
                    "dependency_consistency"
                ),
                "source_spans": spans,
                "endpoint_claim_ids": endpoint_ids,
                "evidence_exact": exact,
                "direct_downstream_claim_ids": sorted(direct_children),
                "downstream_claim_ids": sorted(descendants),
                "downstream_used": bool(descendants),
                "candidate_presence": metrics[qid].get("candidate_presence"),
                "full_chain_completion": metrics[qid].get("graph_proof_completion"),
                "termination": metrics[qid].get("termination_outcome"),
                "answer": predictions[qid].get("answer"),
            })
    trace_hits: list[dict[str, Any]] = []
    for event in _jsonl(run / "reasoning_traces.jsonl"):
        qid = str(event.get("qid", ""))
        interval_ids = interval_ids_by_qid.get(qid, set())
        if not interval_ids:
            continue
        serialized = json.dumps(event, ensure_ascii=False)
        matched = sorted(value for value in interval_ids if value in serialized)
        if not matched:
            continue
        allocation = event.get("allocation") or {}
        candidates = event.get("allocation_candidates") or []
        filtered = event.get("filtered") or []
        trace_hits.append({
            "qid": qid,
            "event": event.get("event"),
            "step": event.get("step", event.get("step_id")),
            "operation": event.get("operation"),
            "matched_interval_claim_ids": matched,
            "selected_target_region": allocation.get("target_region", []),
            "candidate_target_regions": [
                row.get("target_region", []) for row in candidates
                if any(value in row.get("target_region", []) for value in matched)
            ],
            "filtered_join_rows": [
                row for row in filtered
                if any(value in row.get("premise_ids", []) for value in matched)
            ],
        })
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3.11-interval-flow-audit-v1",
        "source_run": str(run),
        "gold_used": False,
        "interval_claim_count": len(rows),
        "evidence_exact_interval_count": sum(bool(row["evidence_exact"]) for row in rows),
        "downstream_used_interval_count": sum(bool(row["downstream_used"]) for row in rows),
        "violation_count": len(violations),
        "violations": violations,
        "rows": rows,
        "trace_hits": trace_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.run)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
