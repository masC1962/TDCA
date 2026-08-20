#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdca_research.data import load_examples


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset-path", default="musique-main/musique-main/data/musique_ans_v1.0_dev.jsonl")
    args = parser.parse_args()
    root = Path(args.run_dir)
    examples = {example.qid: example for example in load_examples(args.dataset_path, args.dataset)}
    predictions = _jsonl(root / "predictions.jsonl")
    metrics = {row["qid"]: row for row in _jsonl(root / "per_example_metrics.jsonl")}
    dynamic = {row["qid"]: row for row in _jsonl(root / "dynamic_per_example_metrics.jsonl")}
    graphs = {row["qid"]: row["graph"] for row in _jsonl(root / "dynamic_graphs.jsonl")}
    print("qid\thop\tem\tstatus\tprediction\tgold\tops\tbranches\tcandidates\tanswer_derivations")
    for prediction in predictions:
        qid = prediction["qid"]
        example = examples[qid]
        row = dynamic.get(qid, {})
        graph = graphs.get(qid, {})
        derivations = []
        for node in graph.get("nodes", {}).values():
            if node.get("kind") != "answer":
                continue
            edge = graph.get("hyperedges", {}).get(node.get("derivation_edge"), {})
            targets = [graph.get("nodes", {}).get(value, {}).get("target_subgoal") for value in node.get("supporting_claims", [])]
            derivations.append(f"{edge.get('inference_type')}:{','.join(map(str, targets))}")
        print("\t".join(map(str, [
            qid, example.hop_count, metrics[qid]["exact_match"], prediction["status"],
            prediction.get("answer"), " | ".join(example.answers), row.get("operation_count"),
            row.get("branch_count"), row.get("candidate_count"), " | ".join(derivations),
        ])))


if __name__ == "__main__":
    main()
