#!/usr/bin/env python3
"""Report whether local datasets are suitable for staged research evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tdca_research.data import (
    DatasetIntegrityError,
    build_split_manifest,
    load_examples,
    validate_dataset_integrity,
)


DEFAULT_SOURCES = (
    ("musique", "musique-main/musique-main/data/musique_ans_v1.0_dev.jsonl"),
    ("hotpotqa", "data/hotpot_dev_distractor_v1.jsonl"),
    ("hotpotqa", "data/hotpotqa_subset_50.jsonl"),
    ("2wikimultihopqa", "data/2wikimultihopqa_subset_50.jsonl"),
    ("2wikimultihopqa", "data/external/2wikimultihopqa_612bc503/dev.jsonl"),
    ("musique", "data/musique_subset_50.jsonl"),
)


def audit(dataset: str, path: str, seed: int) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return {"dataset": dataset, "path": path, "present": False}
    examples = load_examples(source, dataset)
    manifest = build_split_manifest(examples, seed)
    try:
        integrity: dict[str, object] = validate_dataset_integrity(examples, "distractor")
        distractor_suitable = True
        integrity_error = None
    except DatasetIntegrityError as exc:
        integrity = {}
        distractor_suitable = False
        integrity_error = str(exc)
    return {
        "dataset": dataset,
        "path": path,
        "present": True,
        "examples": len(examples),
        "hop_counts": dict(sorted(Counter(str(example.hop_count) for example in examples).items())),
        "distractor_suitable": distractor_suitable,
        "integrity": integrity,
        "integrity_error": integrity_error,
        "disjoint_split_sizes": {name: len(ids) for name, ids in manifest["splits"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {"seed": args.seed, "sources": [audit(dataset, path, args.seed) for dataset, path in DEFAULT_SOURCES]}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
