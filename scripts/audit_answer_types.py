#!/usr/bin/env python3
"""Count categorical gold answers without emitting question or answer content."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    counts = Counter()
    for line in Path(args.dataset).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answer = str(row.get("answer", "")).strip().lower()
        counts[answer if answer in {"yes", "no", "noanswer"} else "other"] += 1
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
