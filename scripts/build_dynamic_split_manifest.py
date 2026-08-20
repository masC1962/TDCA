#!/usr/bin/env python3
"""Freeze Dynamic-Hypergraph-only splits from IDs unused by prior TDCA splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tdca_research.data import load_examples
from tdca_research.utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    examples = load_examples(args.dataset_path, args.dataset)
    all_ids = [example.qid for example in examples]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("dataset qids must be unique")
    excluded: set[str] = set()
    exclusions = []
    for raw_path in args.exclude_manifest:
        path = Path(raw_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ids = {
            str(qid) for values in manifest.get("splits", {}).values()
            for qid in values
        }
        excluded.update(ids)
        exclusions.append({"path": raw_path, "sha256": sha256_file(path), "id_count": len(ids)})
    available = [qid for qid in all_ids if qid not in excluded]
    required = 20 + 50 + 200
    if len(available) < required:
        raise ValueError(f"need {required} unassigned IDs, found {len(available)}")
    random.Random(args.seed).shuffle(available)
    splits = {
        "smoke": available[:20],
        "development": available[20:70],
        "heldout": available[70:270],
    }
    selected = [qid for values in splits.values() for qid in values]
    if len(selected) != len(set(selected)) or set(selected) & excluded:
        raise AssertionError("dynamic splits overlap each other or an excluded manifest")
    manifest = {
        "schema_version": "dynamic-hypergraph-split-v1",
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "dataset_sha256": sha256_file(args.dataset_path),
        "seed": args.seed,
        "total_available": len(all_ids),
        "excluded_id_count": len(excluded),
        "eligible_unassigned_id_count": len(available),
        "exclusions": exclusions,
        "splits": splits,
        "policy": "disjoint 20/50/200 sampled only from IDs absent from every excluded manifest",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
