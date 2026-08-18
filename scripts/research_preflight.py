from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command(command: list[str]) -> dict:
    executable = shutil.which(command[0])
    if not executable:
        return {"available": False, "reason": "not found"}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"available": result.returncode == 0, "returncode": result.returncode, "output": (result.stdout or result.stderr)[:2000]}
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:500]}


def main() -> None:
    dependencies = ["openai", "pydantic", "yaml", "pytest", "torch", "transformers", "sentence_transformers", "faiss", "datasets", "networkx", "igraph", "sklearn"]
    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "is_linux": sys.platform.startswith("linux"),
        "api": {
            "base_url_present": bool(os.getenv("LLM_BASE_URL")),
            "model": os.getenv("LLM_MODEL", ""),
            "credential_present": bool(os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")),
        },
        "commands": {
            "nvidia_smi": command(["nvidia-smi"]),
            "docker": command(["docker", "version"]),
            "git": command(["git", "--version"]),
        },
        "dependencies": {name: bool(importlib.util.find_spec(name)) for name in dependencies},
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "resource_plan": {
            "hipporag2_dataset": {
                "url": "https://huggingface.co/datasets/osunlp/HippoRAG_2",
                "target": "external_data/hipporag2",
                "size": "386074835 bytes (0.360 GiB), 20 files; queried 2026-08-13",
                "core_files": {
                    "musique.json": {"bytes": 12543629, "sha256_lfs": "98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46"},
                    "musique_corpus.json": {"bytes": 6239261, "git_blob_oid": "01ebb7c8a513e64350768acb5679d8c4ebfde241"},
                    "hotpotqa.json": {"bytes": 8183058, "git_blob_oid": "dcb199ffd41a96b5880c08c2948d07ecf4e09e44"},
                    "hotpotqa_corpus.json": {"bytes": 6414109, "git_blob_oid": "69b0e793004d8f5ff1760f4ed363ab6f23aaa7d8"},
                    "2wikimultihopqa.json": {"bytes": 6505789, "git_blob_oid": "c87b01db53166b2b85b82d8773c6ed685bab2c16"},
                    "2wikimultihopqa_corpus.json": {"bytes": 3083943, "git_blob_oid": "d2e236375a97ee41d27287b9efb3ef9036d1e072"},
                },
                "verification": "compare bytes and LFS sha256 where provided; record local sha256 for every downloaded file",
            },
            "hipporag2_repo": {
                "url": "https://github.com/OSU-NLP-Group/HippoRAG.git",
                "target": "external_repos/HippoRAG",
                "commit": "c617143f01477243992a63b2e2151cc003dd3b21",
            },
        },
    }
    target = Path("research_outputs/preflight.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
