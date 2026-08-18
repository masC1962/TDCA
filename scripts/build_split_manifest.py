#!/usr/bin/env python3
"""Build a deterministic, disjoint staged-evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdca_research.data import build_split_manifest, load_examples
from tdca_research.utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    examples = load_examples(args.dataset_path, args.dataset)
    qids = [example.qid for example in examples]
    if len(qids) != len(set(qids)):
        raise ValueError("dataset qids must be unique before a split manifest can be frozen")
    manifest = build_split_manifest(examples, args.seed)
    manifest.update({
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "dataset_sha256": sha256_file(args.dataset_path),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
