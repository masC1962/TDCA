#!/usr/bin/env python3
"""Predict or score the frozen public natural-revision suite.

`predict` has no label-manifest argument by design. `score` opens labels only
after a complete prediction artifact already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.revision_eval import (
    load_unlabeled_inputs,
    predict_items,
    score_prediction_rows,
)
from tdca_research.runtime import build_llm


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def predict(args: argparse.Namespace) -> None:
    items, manifest = load_unlabeled_inputs(args.inputs, args.split, unlabeled_cache_path=args.unlabeled_cache)
    if args.max_items is not None:
        if args.split == "evaluation":
            raise ValueError("the frozen evaluation split cannot be partially opened")
        items = items[: args.max_items]
    config = DynamicV2ResearchConfig(
        llm_base_url=args.base_url,
        llm_model=args.model,
        api_cache_dir=args.cache_dir,
        prompt_version="dynamic-hypergraph-v2-revision-v1",
        max_candidates_per_subgoal=4,
        max_graph_nodes=32,
        max_graph_operations=16,
        max_total_tokens=4000,
        final_reserve_tokens=200,
    )
    llm = build_llm(config)
    rows = predict_items(items, llm, config)
    manifest_bytes = Path(args.inputs).read_bytes()
    output = {
        "schema_version": 1,
        "split": args.split,
        "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "upstream_commit": manifest["provenance"]["upstream_commit"],
        "selection_seed": manifest["provenance"]["selection_seed"],
        "model": llm.model_name,
        "prompt_version": config.prompt_version,
        "gold_labels_loaded": False,
        "predictions": rows,
    }
    write_json(args.output, output)
    print(json.dumps({
        "output": str(args.output),
        "count": len(rows),
        "provider_calls": sum(row["usage"]["provider_calls"] for row in rows),
        "provider_reported_tokens": sum(row["usage"]["provider_reported_tokens"] for row in rows),
    }, indent=2))


def score(args: argparse.Namespace) -> None:
    artifact = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if artifact.get("input_manifest_sha256") != labels.get("input_manifest_sha256"):
        raise ValueError("prediction input hash does not match sealed labels")
    metrics, rows = score_prediction_rows(artifact["predictions"], labels, artifact["split"])
    output = {
        "schema_version": 1,
        "split": artifact["split"],
        "prediction_artifact": str(args.predictions),
        "label_manifest_opened_after_prediction": True,
        "metrics": metrics,
        "per_item": rows,
    }
    write_json(args.output, output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--inputs", type=Path, default=Path("configs/revision/vitaminc_revision_inputs_seed20260820.json"))
    prediction.add_argument("--unlabeled-cache", type=Path)
    prediction.add_argument("--split", choices=["development", "evaluation"], required=True)
    prediction.add_argument("--max-items", type=int)
    prediction.add_argument("--base-url", default="")
    prediction.add_argument("--model", default="")
    prediction.add_argument("--cache-dir", default=".cache/tdca_research/revision_v1")
    prediction.add_argument("--output", type=Path, required=True)
    prediction.set_defaults(func=predict)
    scoring = subparsers.add_parser("score")
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--labels", type=Path, default=Path("configs/revision/vitaminc_revision_labels_seed20260820.json"))
    scoring.add_argument("--output", type=Path, required=True)
    scoring.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
