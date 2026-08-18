#!/usr/bin/env python3
"""Hash post-hoc analysis artifacts without mutating the source run manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tdca_research.utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source_run)
    artifacts = [Path(value) for value in args.artifact]
    missing = [str(path) for path in artifacts if not path.exists()]
    if missing:
        raise ValueError(f"analysis artifacts missing: {missing}")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source),
        "source_run_manifest_sha256": sha256_file(source / "run_manifest.json"),
        "source_prediction_sha256": sha256_file(source / "predictions.jsonl"),
        "artifacts": {str(path): sha256_file(path) for path in artifacts},
        "post_hoc_only": True,
    }
    Path(args.output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
