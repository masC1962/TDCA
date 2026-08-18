from __future__ import annotations

import argparse
import json
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cached HippoRAG responses without credentials")
    parser.add_argument("cache")
    parser.add_argument("--include-messages", action="store_true", help="explicitly emit cached prompt content")
    args = parser.parse_args()
    connection = sqlite3.connect(args.cache)
    rows = connection.execute("SELECT message, metadata FROM cache ORDER BY rowid").fetchall()
    connection.close()
    if not args.include_messages:
        metadata_rows = [json.loads(metadata) for _, metadata in rows]
        print(json.dumps({
            "cached_provider_calls": len(rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in metadata_rows),
            "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in metadata_rows),
            "total_tokens": sum(int(row.get("prompt_tokens", 0)) + int(row.get("completion_tokens", 0)) for row in metadata_rows),
        }))
        return
    print(json.dumps({"warning": "cached message content follows; do not publish if it contains held-out data"}))
    for index, (message, metadata) in enumerate(rows):
        print(json.dumps({
            "index": index,
            "message": message,
            "metadata": json.loads(metadata),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
