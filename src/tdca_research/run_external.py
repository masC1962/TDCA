from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

from .baselines.external import load_manifest
from .utils import safe_error, write_json


def _head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.STDOUT,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and invoke a pinned official external baseline")
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    entries = {entry.name: entry for entry in load_manifest()}
    if args.name not in entries:
        raise SystemExit(f"baseline {args.name!r} is absent from external_baselines/manifest.yaml")
    entry = entries[args.name]
    adapter = yaml.safe_load(Path(args.adapter).read_text(encoding="utf-8")) or {}
    repository = Path(str(adapter.get("repository_dir", "")))
    report = {
        "name": args.name, "implementation": entry.implementation,
        "source_url": entry.source_url, "expected_commit": entry.commit,
        "repository_dir": str(repository), "verified": False, "executed": False,
    }
    try:
        if not repository.is_dir():
            raise RuntimeError(f"official repository is not installed at {repository}")
        actual = _head(repository)
        report["actual_commit"] = actual
        if entry.commit and actual != entry.commit:
            raise RuntimeError(f"commit mismatch: expected {entry.commit}, got {actual}")
        report["verified"] = True
        command = adapter.get("command")
        if args.verify_only:
            write_json(args.output, report)
            return
        if not isinstance(command, list) or not command:
            raise RuntimeError("adapter command is intentionally unset until the upstream Linux dependency audit")
        argv = [str(value).format(input=args.input, output=args.output) for value in command]
        # Credentials remain inherited environment state and are never serialized.
        completed = subprocess.run(argv, cwd=repository, env=os.environ.copy(), text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(f"upstream command failed ({completed.returncode}): {safe_error(completed.stderr)}")
        report["executed"] = True
        report["command_argv_without_environment"] = argv
        write_json(str(args.output) + ".reproduction.json", report)
    except Exception as exc:
        report["failure_reason"] = safe_error(exc)
        write_json(str(args.output) + ".failure.json", report)
        raise SystemExit(report["failure_reason"])


if __name__ == "__main__":
    main()
