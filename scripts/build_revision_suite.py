#!/usr/bin/env python3
"""Build a reproducible, case-disjoint VitaminC belief-revision suite.

The generated input manifest deliberately contains no gold labels. Labels are
written to a separate file that prediction code must not open. Raw upstream
data stays under data/external/ and is ignored by git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SEED = 20260820
UPSTREAM_COMMIT = "be6febb761b0b2807687e61e0b5282e459df2fa0"
UPSTREAM_FILE = "test.jsonl"
UPSTREAM_URL = (
    "https://hf-mirror.com/datasets/tals/vitaminc/resolve/"
    f"{UPSTREAM_COMMIT}/{UPSTREAM_FILE}"
)
EXPECTED_UPSTREAM_SHA256 = "7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad"
LICENSE = "CC-BY-SA-3.0"
LABEL_TO_ACTION = {
    "REFUTES": "should_revise",
    "SUPPORTS": "should_not_revise",
    "NOT ENOUGH INFO": "should_not_revise",
}
TARGETS = {
    "development": {"REFUTES": 10, "SUPPORTS": 5, "NOT ENOUGH INFO": 5},
    "evaluation": {"REFUTES": 30, "SUPPORTS": 15, "NOT ENOUGH INFO": 15},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: str) -> str:
    payload = "\x1f".join((str(SEED), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def download_if_needed(path: Path) -> None:
    if path.exists() and sha256_file(path) == EXPECTED_UPSTREAM_SHA256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(UPSTREAM_URL, headers={"User-Agent": "TDCA-research/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    observed = sha256_file(temporary)
    if observed != EXPECTED_UPSTREAM_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"VitaminC checksum mismatch: expected {EXPECTED_UPSTREAM_SHA256}, got {observed}"
        )
    temporary.replace(path)


def iter_real_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("revision_type") != "real" or row.get("label") not in LABEL_TO_ACTION:
                continue
            if not row.get("unique_id") or not row.get("case_id"):
                continue
            yield row


def contradiction_category(row: dict[str, Any]) -> str:
    combined = f"{row.get('claim', '')} {row.get('evidence', '')}".lower()
    if re.search(r"\b(?:18|19|20)\d{2}\b|\b(?:before|after|during|until|since)\b", combined):
        return "temporal"
    if re.search(r"\b\d+(?:[.,]\d+)?\b", combined):
        return "numeric"
    if re.search(r"\b(?:not|never|no|neither|without|instead)\b", combined):
        return "explicit_negation"
    return "entity_or_relation"


def content_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {"claim": row["claim"], "evidence": row["evidence"]},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[str(row["label"])].append(row)
    for label, pool in pools.items():
        pool.sort(key=lambda row: stable_key(label, str(row["case_id"]), str(row["unique_id"])))

    selected: dict[str, list[dict[str, Any]]] = {name: [] for name in TARGETS}
    used_cases: set[str] = set()
    for split, targets in TARGETS.items():
        remaining = dict(targets)
        cursors = defaultdict(int)
        while any(value > 0 for value in remaining.values()):
            progressed = False
            for label in ("REFUTES", "SUPPORTS", "NOT ENOUGH INFO"):
                if remaining[label] <= 0:
                    continue
                pool = pools[label]
                while cursors[label] < len(pool):
                    row = pool[cursors[label]]
                    cursors[label] += 1
                    case_id = str(row["case_id"])
                    if case_id in used_cases:
                        continue
                    used_cases.add(case_id)
                    selected[split].append(row)
                    remaining[label] -= 1
                    progressed = True
                    break
            if not progressed:
                raise RuntimeError(f"Insufficient case-disjoint rows for {split}: {remaining}")
    return selected


def build_manifests(selected: dict[str, list[dict[str, Any]]], upstream_path: Path) -> tuple[dict, dict, list[dict]]:
    provenance = {
        "dataset": "VitaminC",
        "paper": "https://aclanthology.org/2021.naacl-main.52/",
        "repository": "https://github.com/TalSchuster/VitaminC",
        "distribution": "https://huggingface.co/datasets/tals/vitaminc",
        "license": LICENSE,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": UPSTREAM_FILE,
        "upstream_sha256": sha256_file(upstream_path),
        "selection_seed": SEED,
        "selection_policy": "real revisions only; stable hash order; globally case-disjoint",
    }
    inputs: dict[str, Any] = {"schema_version": 1, "provenance": provenance, "splits": {}}
    labels: dict[str, Any] = {
        "schema_version": 1,
        "warning": "Open only after predictions are durably written.",
        "input_manifest_sha256": "FILLED_AFTER_SERIALIZATION",
        "splits": {},
    }
    unlabeled_rows: list[dict[str, Any]] = []
    for split, rows in selected.items():
        ordered = sorted(rows, key=lambda row: stable_key(split, str(row["unique_id"])))
        inputs["splits"][split] = [
            {
                "item_id": str(row["unique_id"]),
                "case_id": str(row["case_id"]),
                "page": str(row.get("page", "")),
                "content_sha256": content_hash(row),
            }
            for row in ordered
        ]
        labels["splits"][split] = [
            {
                "item_id": str(row["unique_id"]),
                "gold_relation": str(row["label"]),
                "expected_action": LABEL_TO_ACTION[str(row["label"])],
                "category": (
                    contradiction_category(row)
                    if row["label"] == "REFUTES"
                    else "support" if row["label"] == "SUPPORTS" else "insufficient_evidence"
                ),
            }
            for row in ordered
        ]
        unlabeled_rows.extend({
            "item_id": str(row["unique_id"]),
            "case_id": str(row["case_id"]),
            "claim": str(row["claim"]),
            "evidence": str(row["evidence"]),
            "page": str(row.get("page", "")),
            "content_sha256": content_hash(row),
        } for row in ordered)
    return inputs, labels, unlabeled_rows


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/external/vitaminc/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("configs/revision"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if not args.no_download:
        download_if_needed(args.raw)
    if not args.raw.exists():
        raise FileNotFoundError(args.raw)
    observed = sha256_file(args.raw)
    if observed != EXPECTED_UPSTREAM_SHA256:
        raise RuntimeError(f"Unexpected upstream checksum: {observed}")

    selected = select_rows(iter_real_rows(args.raw))
    inputs, labels, unlabeled_rows = build_manifests(selected, args.raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_path = args.raw.parent / f"revision_unlabeled_seed{SEED}.jsonl"
    unlabeled_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in unlabeled_rows
    )
    unlabeled_path.write_bytes(unlabeled_bytes)
    inputs["provenance"]["unlabeled_cache"] = str(unlabeled_path)
    inputs["provenance"]["unlabeled_cache_sha256"] = hashlib.sha256(unlabeled_bytes).hexdigest()
    input_path = args.output_dir / f"vitaminc_revision_inputs_seed{SEED}.json"
    label_path = args.output_dir / f"vitaminc_revision_labels_seed{SEED}.json"
    input_bytes = canonical_bytes(inputs)
    labels["input_manifest_sha256"] = hashlib.sha256(input_bytes).hexdigest()
    input_path.write_bytes(input_bytes)
    label_path.write_bytes(canonical_bytes(labels))
    print(json.dumps({
        "inputs": str(input_path),
        "labels": str(label_path),
        "unlabeled_cache": str(unlabeled_path),
        "development": len(inputs["splits"]["development"]),
        "evaluation": len(inputs["splits"]["evaluation"]),
        "case_overlap": len(
            {row["case_id"] for row in inputs["splits"]["development"]}
            & {row["case_id"] for row in inputs["splits"]["evaluation"]}
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
