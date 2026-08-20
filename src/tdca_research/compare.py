from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .utils import write_json


def compare(base_path: str | Path, new_path: str | Path, seed: int = 520, samples: int = 10000) -> dict:
    def load(path):
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        identifiers = [str(row["qid"]) for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"paired comparison input contains duplicate qids: {path}")
        return {str(row["qid"]): row for row in rows}
    base, new = load(base_path), load(new_path)
    if set(base) != set(new):
        raise ValueError("paired comparison inputs must contain exactly the same qids")
    ids = sorted(base)
    pairs = []
    f1_pairs = []
    for qid in ids:
        if "exact_match" in base[qid] and "exact_match" in new[qid]:
            pairs.append((float(base[qid]["exact_match"]), float(new[qid]["exact_match"])))
        if "f1" in base[qid] and "f1" in new[qid]:
            f1_pairs.append((float(base[qid]["f1"]), float(new[qid]["f1"])))
    interval = None
    if pairs:
        rng = random.Random(seed)
        differences = []
        for _ in range(samples):
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            differences.append(sum(new_value - base_value for base_value, new_value in sample) / len(sample))
        differences.sort()
        lower_index = max(0, min(samples - 1, int(0.025 * samples) - 1))
        upper_index = max(0, min(samples - 1, int(0.975 * samples) - 1))
        interval = {
            "mean_difference": sum(new_value - base_value for base_value, new_value in pairs) / len(pairs),
            "lower_95": differences[lower_index],
            "upper_95": differences[upper_index],
            "seed": seed,
            "samples": samples,
        }
    def bootstrap_interval(values: list[tuple[float, float]]) -> dict | None:
        if not values:
            return None
        rng = random.Random(seed)
        differences = []
        for _ in range(samples):
            sample = [values[rng.randrange(len(values))] for _ in values]
            differences.append(sum(new_value - base_value for base_value, new_value in sample) / len(sample))
        differences.sort()
        lower_index = max(0, min(samples - 1, int(0.025 * samples) - 1))
        upper_index = max(0, min(samples - 1, int(0.975 * samples) - 1))
        return {
            "mean_difference": sum(new_value - base_value for base_value, new_value in values) / len(values),
            "lower_95": differences[lower_index],
            "upper_95": differences[upper_index],
            "seed": seed,
            "samples": samples,
        }
    f1_interval = bootstrap_interval(f1_pairs)
    transitions = {}
    for qid in ids:
        key = f"{base[qid].get('status')}->{new[qid].get('status')}"
        transitions[key] = transitions.get(key, 0) + 1
    costs = {}
    for label, rows in (("base", base), ("new", new)):
        values = []
        for row in rows.values():
            usage = row.get("usage", {})
            total = usage.get("total_tokens", row.get("total_tokens"))
            if total is not None and "exact_match" in row:
                values.append((float(total), float(row["exact_match"])))
        costs[label] = {
            "mean_total_tokens": sum(value[0] for value in values) / len(values) if values else None,
            "exact_match": sum(value[1] for value in values) / len(values) if values else None,
        }
    return {
        "aligned_count": len(ids),
        "status_changes": transitions,
        "paired_bootstrap_exact_match": interval,
        "paired_bootstrap_f1": f1_interval,
        "quality_cost_summary": costs,
        "note": "Bootstrap is available when aligned rows contain per-example exact_match.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aligned prediction files with paired bootstrap")
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    write_json(args.output, compare(args.base, args.new, seed=args.seed, samples=args.samples))


if __name__ == "__main__":
    main()
