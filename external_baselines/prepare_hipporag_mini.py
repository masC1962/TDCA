from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a frozen TDCA split for a HippoRAG mini-global run")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    wanted = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["splits"][args.split][: args.limit]
    wanted_set = set(wanted)
    by_id = {}
    with Path(args.dataset).open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("id")) in wanted_set:
                by_id[str(row["id"])] = row
    missing = [qid for qid in wanted if qid not in by_id]
    if missing:
        raise ValueError(f"split IDs missing from source dataset: {missing}")
    samples = [by_id[qid] for qid in wanted]

    corpus_by_key = {}
    for sample in samples:
        for paragraph in sample.get("paragraphs", []):
            title = str(paragraph.get("title", ""))
            text = str(paragraph.get("paragraph_text") or paragraph.get("text") or "")
            corpus_by_key.setdefault((title, text), {"title": title, "text": text})
    corpus = list(corpus_by_key.values())

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.name}.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    (output / f"{args.name}.jsonl").write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    (output / f"{args.name}_corpus.json").write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    (output / f"{args.name}_manifest.json").write_text(json.dumps({
        "setting": "mini_global_union",
        "source_split": args.split,
        "source_manifest": args.manifest,
        "sample_ids": wanted,
        "sample_count": len(samples),
        "corpus_count": len(corpus),
        "warning": "Not distractor-equivalent: corpus is the union of candidate passages for the selected questions.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": len(samples), "corpus_count": len(corpus), "name": args.name}))


if __name__ == "__main__":
    main()
