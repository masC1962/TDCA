from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize JSONL experiment failures without printing dataset text.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--error-chars", type=int, default=240)
    args = parser.parse_args()

    rows = []
    with args.path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            error = str(row.get("error") or row.get("reason") or "")
            rows.append(
                {
                    "qid": row.get("qid") or row.get("id"),
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                    "error": error[: args.error_chars],
                }
            )

    categories = Counter()
    for row in rows:
        error = row["error"].lower()
        if "timeout" in error:
            category = "timeout"
        elif "json" in error or "parse" in error:
            category = "json_or_parse"
        elif "429" in error or "rate" in error:
            category = "rate_limit"
        elif "connection" in error or "connect" in error:
            category = "connection"
        else:
            category = "other"
        categories[category] += 1

    print(json.dumps({"count": len(rows), "categories": categories, "failures": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
