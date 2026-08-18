#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()
    data = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    print(json.dumps({
        "completed_count": data.get("completed_count", len(rows)),
        "sample_count": data.get("sample_count"),
        "response_count": len(rows),
        "empty_response_count": sum(not str(row.get("response", "")).strip() for row in rows),
        "warning": data.get("warning"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
