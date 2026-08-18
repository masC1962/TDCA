from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .retrieval import DenseRetriever
from .runtime import _code_version, _global_passages
from .utils import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize and validate a reusable global dense index")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--fallback", choices=["error", "explicit_tfidf"], default="error")
    args = parser.parse_args()
    started = time.perf_counter()
    passages = _global_passages(args.corpus)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    retriever = DenseRetriever(passages, args.encoder, fallback=args.fallback)
    retriever.save(output)
    write_json(output / "index_manifest.json", {
        "format_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus,
        "corpus_sha256": sha256_file(args.corpus),
        "passage_count": len(passages),
        "encoder": args.encoder,
        "dimensions": list(retriever.matrix.shape),
        "index_backend": retriever.backend,
        "code_version": _code_version(),
        "index_payload_size_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "build_seconds": time.perf_counter() - started,
    })
if __name__ == "__main__":
    main()
