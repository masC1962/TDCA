#!/usr/bin/env python3
"""Verify every file hash recorded by a post-hoc analysis manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdca_research.utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    path = Path(args.manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("manifest must contain a non-empty artifacts hash mapping")
    if args.expected_count is not None and len(artifacts) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} artifacts, found {len(artifacts)}"
        )
    missing = [name for name in artifacts if not Path(name).is_file()]
    mismatched = [
        name for name, expected in artifacts.items()
        if Path(name).is_file() and sha256_file(name) != expected
    ]
    if missing or mismatched:
        raise ValueError(
            f"analysis manifest verification failed: missing={missing}, "
            f"checksum_mismatches={mismatched}"
        )
    print(json.dumps({
        "manifest": str(path),
        "verified": True,
        "artifact_count": len(artifacts),
    }, indent=2))


if __name__ == "__main__":
    main()
