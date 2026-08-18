# Legacy TDCA v0

This directory freezes the legacy interface as a comparison target while the
research implementation lives under `src/tdca_research`. The legacy source is
intentionally left at the repository root so existing output paths and scripts
remain valid. Its audited source hashes are recorded in `source_manifest.json`.

Run a dry test from the repository root:

```bash
python legacy/tdca_v0/run.py --mock_llm
```

The frozen batch runner is exposed through a thin, non-algorithmic wrapper:

```bash
python legacy/tdca_v0/run_batch.py --dataset data/musique_subset_50.jsonl --limit 50
```

The legacy implementation is not imported by the new research package.
