from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import ResearchConfig
from .data import load_examples, select_split
from .retrieval import build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure question-level retrieval before spending LLM calls")
    parser.add_argument("--config", required=True)
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0, help="diagnostic cutoff; zero uses the frozen config")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = ResearchConfig.from_yaml(args.config)
    top_k = args.top_k or config.top_k
    if top_k <= 0:
        raise ValueError("top-k must be positive")
    examples = load_examples(config.dataset_path, config.dataset)
    manifest = json.loads(Path(config.split_manifest_path).read_text(encoding="utf-8"))
    selected = select_split(examples, config.split, manifest, config.split_seed)
    if args.limit > 0:
        selected = selected[: args.limit]

    report: dict[str, object] = {
        "dataset": config.dataset,
        "setting": config.setting,
        "split": config.split,
        "top_k": top_k,
        "sample_ids": [example.qid for example in selected],
        "retrievers": {},
    }
    for kind in args.retrievers:
        started = time.perf_counter()
        rows = []
        for example in selected:
            retriever = build_retriever(kind, example.passages, config.dense_model, config.dense_fallback)
            hits = retriever.search(example.question, top_k)
            retrieved = {hit.passage.passage_id for hit in hits}
            gold = set(example.gold_document_ids)
            overlap = retrieved & gold
            rows.append({
                "qid": example.qid,
                "hop_count": example.hop_count,
                "gold_count": len(gold),
                "gold_recalled": len(overlap),
                "support_recall": len(overlap) / len(gold) if gold else None,
                "all_gold_recalled": bool(gold) and gold <= retrieved,
                "retrieved_ids": [hit.passage.passage_id for hit in hits],
            })
        recalls = [row["support_recall"] for row in rows if row["support_recall"] is not None]
        by_hop = {}
        for hop in sorted({str(row["hop_count"]) for row in rows}):
            subset = [row for row in rows if str(row["hop_count"]) == hop]
            hop_recalls = [row["support_recall"] for row in subset if row["support_recall"] is not None]
            by_hop[hop] = {
                "count": len(subset),
                "support_recall": sum(hop_recalls) / len(hop_recalls) if hop_recalls else None,
                "all_gold_recalled": sum(row["all_gold_recalled"] for row in subset) / len(subset),
            }
        report["retrievers"][kind] = {
            "support_recall": sum(recalls) / len(recalls) if recalls else None,
            "all_gold_recalled": sum(row["all_gold_recalled"] for row in rows) / len(rows) if rows else None,
            "wall_seconds": time.perf_counter() - started,
            "by_hop": by_hop,
            "rows": rows,
        }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
