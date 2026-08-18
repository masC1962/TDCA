#!/usr/bin/env python3
"""Hash the final HippoRAG validation comparison bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tdca_research.utils import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hipporag-artifact", required=True)
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = Path(args.hipporag_artifact)
    comparison = Path(args.comparison_dir)
    paths = [
        artifact,
        comparison / "main_independent_eval.json",
        comparison / "hipporag_independent_eval.json",
        comparison / "paired_comparison.json",
        comparison / "summary.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"analysis bundle is incomplete: {missing}")
    write_json(args.output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {str(path): sha256_file(path) for path in paths},
        "post_hoc_only": True,
    })
    print(args.output)


if __name__ == "__main__":
    main()
