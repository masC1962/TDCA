#!/usr/bin/env python3
"""Create a checksummed manifest for an immutable post-hoc analysis bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tdca_research.utils import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    paths = [Path(value) for value in args.artifacts]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"analysis bundle is incomplete: {missing}")
    write_json(args.output, {
        "label": args.label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {str(path): sha256_file(path) for path in paths},
        "post_hoc_only": True,
    })
    print(args.output)


if __name__ == "__main__":
    main()
