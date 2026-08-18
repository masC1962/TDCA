from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_examples
from .evaluation import evaluate_predictions, grouped_metrics
from .models import prediction_from_dict
from .utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-qids", help="Optional JSON manifest; require exactly the named split qid set",
    )
    parser.add_argument("--split", help="Split name inside --expected-qids")
    args = parser.parse_args()
    examples = load_examples(args.dataset_path, args.dataset)
    predictions = []
    seen_qids: set[str] = set()
    for line in Path(args.predictions).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        qid = str(row["qid"])
        if qid in seen_qids:
            raise ValueError(f"duplicate prediction qid: {qid}")
        seen_qids.add(qid)
        predictions.append(prediction_from_dict(row))
    if args.expected_qids:
        manifest = json.loads(Path(args.expected_qids).read_text(encoding="utf-8"))
        if not args.split or args.split not in manifest.get("splits", {}):
            raise ValueError("--split must name a split in --expected-qids")
        expected = {str(value) for value in manifest["splits"][args.split]}
        if seen_qids != expected:
            raise ValueError("prediction qids do not exactly match the expected split")
    metrics, rows = evaluate_predictions(examples, predictions)
    by_hop = grouped_metrics(rows, "hop_count")
    by_type = grouped_metrics(rows, "question_type")
    write_json(args.output, {"metrics": metrics, "metrics_by_hop": by_hop, "metrics_by_type": by_type, "rows": rows})


if __name__ == "__main__":
    main()
