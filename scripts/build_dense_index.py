#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from dataset_adapters.loader import load_examples
from retriever import DenseTextRetriever, TextRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dense/TF-IDF index for a hotpot-like dataset")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", default="generic")
    parser.add_argument("--encoder_path", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    examples = load_examples(args.dataset_path, dataset_name=args.dataset_name, limit=args.limit)
    seen = set()
    records = []
    for ex in examples:
        for doc in ex.docs:
            key = (doc.title, doc.text)
            if key in seen:
                continue
            seen.add(key)
            records.append(TextRecord(item_id=doc.doc_id + "::" + str(len(records)), text=doc.text, metadata={"title": doc.title, **doc.metadata}))
    retriever = DenseTextRetriever(records=records, encoder_path=args.encoder_path)
    retriever.save_index(args.output)
    print(f"Built {retriever.backend} index with {len(records)} docs -> {args.output}")


if __name__ == "__main__":
    main()
