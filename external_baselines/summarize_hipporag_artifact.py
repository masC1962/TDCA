from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print compact HippoRAG artifact metrics")
    parser.add_argument("input")
    args = parser.parse_args()
    artifact = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = artifact.get("rows", [])
    retrieval_keys = ("Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20")
    all_gold = {
        key: sum(float(row.get("retrieval_metrics", {}).get(key, 0.0)) >= 1.0 - 1e-9 for row in rows) / max(1, len(rows))
        for key in retrieval_keys
    }
    print(json.dumps({
        "sample_count": artifact.get("sample_count"),
        "wall_seconds": artifact.get("wall_seconds"),
        "qa_metrics": artifact.get("qa_metrics"),
        "retrieval_metrics": {key: artifact.get("retrieval_metrics", {}).get(key) for key in retrieval_keys},
        "all_gold_recalled": all_gold,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
