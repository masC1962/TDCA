#!/usr/bin/env python3
"""Convert the pinned 2WikiMultiHopQA dev parquet mirror to JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


PINNED_REPOSITORY = "xanhho/2WikiMultihopQA"
PINNED_COMMIT = "612bc5039a457880d9e7d84c3b0a4cf154b70e4f"
PINNED_DEV_SHA256 = "c0d8b60b9026b728fb07ad74c5252a0f188f6942e8ba5c02df4dfa369502ea8d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    observed_hash = sha256(source)
    if observed_hash != PINNED_DEV_SHA256:
        raise ValueError(f"unexpected source hash: {observed_hash}")

    table = pq.read_table(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in table.to_pylist():
            # This mirror serializes nested Arrow values as JSON strings.
            # Decode by field shape, independent of dataset entities/content.
            for key in ("context", "supporting_facts", "evidences"):
                value = row.get(key)
                if isinstance(value, str):
                    row[key] = json.loads(value)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    provenance = {
        "source_repository": PINNED_REPOSITORY,
        "source_commit": PINNED_COMMIT,
        "source_file": "dev.parquet",
        "source_sha256": observed_hash,
        "source_bytes": source.stat().st_size,
        "rows": table.num_rows,
        "columns": table.column_names,
        "output_file": str(output),
        "output_sha256": sha256(output),
    }
    Path(f"{output}.provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
