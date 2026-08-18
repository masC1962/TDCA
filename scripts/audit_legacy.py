from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import TDCAConfig


def main() -> None:
    scheduler = Path("tdca_scheduler.py")
    tree = ast.parse(scheduler.read_text(encoding="utf-8-sig"))
    manifest_path = Path("legacy/tdca_v0/source_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib
    hashes = {}
    for value in manifest["sources"]:
        path = Path(value)
        hashes[value] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["source_sha256"] = hashes
    manifest["audit"] = {
        "scheduler_lines": len(scheduler.read_text(encoding="utf-8-sig").splitlines()),
        "scheduler_functions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)),
        "config_fields": len(dataclasses.fields(TDCAConfig)),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
