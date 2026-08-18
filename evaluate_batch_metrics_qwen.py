#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

from answer_metrics import METRIC_KEYS, aggregate_metric_rows, compute_answer_metrics, exact_match


def load_summary_csv(path: Path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            summary_name = next(name for name in zf.namelist() if name.endswith("summary.csv"))
            text = zf.read(summary_name).decode("utf-8")
            return list(csv.DictReader(text.splitlines())), path.stem
    summary = path / "summary.csv" if path.is_dir() else path
    with summary.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f)), summary.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="batch output dir, summary.csv, or batch zip")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    rows, run_name = load_summary_csv(Path(args.input_path))
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / f"{run_name}_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for row in rows:
        pred = row.get("pred", "")
        gold = row.get("gold", "")
        enriched = dict(row)
        enriched["metric_exact_match"] = exact_match(pred, gold)
        metrics = compute_answer_metrics(pred, gold, row)
        for key, value in metrics.items():
            enriched[f"metric_{key}"] = value
        metric_rows.append(enriched)

    aggregate_source = []
    for row in metric_rows:
        aggregate_source.append({key: row.get(f"metric_{key}", 0.0) for key in METRIC_KEYS})
    aggregate = {
        "count": len(metric_rows),
        "metric_exact_match": round(sum(float(r.get("metric_exact_match", 0.0) or 0.0) for r in metric_rows) / len(metric_rows), 6) if metric_rows else 0.0,
    }
    for key, value in aggregate_metric_rows(aggregate_source).items():
        aggregate[f"metric_{key}"] = value
    aggregate["metric_bert_f1"] = None
    aggregate["metric_sbert_similarity"] = None

    csv_path = out_dir / "metrics_summary.csv"
    json_path = out_dir / "metrics_aggregate.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()) if metric_rows else [])
        if metric_rows:
            writer.writeheader()
            writer.writerows(metric_rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
