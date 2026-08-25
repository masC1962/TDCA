from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from tdca_research.dynamic_v2.metrics import _best_graph_proof
from tdca_research.utils import write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def analyze_run(label: str, run_dir: Path) -> dict[str, Any]:
    graph_path = run_dir / "dynamic_v2_graphs.jsonl"
    if not graph_path.is_file():
        raise FileNotFoundError(f"missing graph artifact: {graph_path}")
    proof_rows = []
    for row in _read_jsonl(graph_path):
        payload = row.get("graph", row.get("graph_snapshot", row))
        proof_rows.append({
            "qid": str(row.get("qid", "")),
            **_best_graph_proof(payload, allow_historical_unseal=True),
        })
    metrics = _read_json(run_dir / "metrics.json")
    dynamic_metrics = _read_json(run_dir / "dynamic_v2_metrics.json")
    return {
        "label": label,
        "run_dir": str(run_dir),
        "count": len(proof_rows),
        "graph_proof_completion_rate": mean(
            float(row["graph_proof_completion"]) for row in proof_rows
        ) if proof_rows else 0.0,
        "proof_connected_rate": mean(
            float(row["proof_connected"]) for row in proof_rows
        ) if proof_rows else 0.0,
        "mean_dependency_coverage": mean(
            float(row["dependency_coverage"]) for row in proof_rows
        ) if proof_rows else 0.0,
        "mean_evidence_leaf_coverage": mean(
            float(row["evidence_leaf_coverage"]) for row in proof_rows
        ) if proof_rows else 0.0,
        "mean_distinct_evidence_leaf_ratio": mean(
            float(row["distinct_evidence_leaf_ratio"]) for row in proof_rows
        ) if proof_rows else 0.0,
        "legacy_execution_plan_completion_rate": metrics.get(
            "execution_plan_completion_rate",
            metrics.get("full_chain_completion_rate", 0.0),
        ),
        "exact_match": metrics.get("exact_match", 0.0),
        "f1": metrics.get("f1", 0.0),
        "candidate_presence_rate": dynamic_metrics.get("candidate_presence_rate", 0.0),
        "per_example": proof_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gold-free v2.4 graph-proof recomputation for frozen runs",
    )
    parser.add_argument(
        "--run", action="append", required=True,
        help="LABEL=RUN_DIR; may be supplied more than once",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for value in args.run:
        if "=" not in value:
            raise ValueError("--run must use LABEL=RUN_DIR")
        label, path = value.split("=", 1)
        rows.append(analyze_run(label, Path(path)))
    write_json(args.output, {
        "schema_version": "dynamic-hypergraph-v2.4-offline-proof-audit-v1",
        "gold_used_for_proof_metrics": False,
        "historical_controller_seal_revalidated": False,
        "historical_unseal_reason": (
            "source-schema drift; artifact checksums remain the immutability authority"
        ),
        "runs": rows,
    })


if __name__ == "__main__":
    main()
