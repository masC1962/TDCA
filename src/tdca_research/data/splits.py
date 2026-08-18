from __future__ import annotations

import random
from typing import Any

from ..models import QAExample

SPLIT_SIZES = {"smoke": 20, "tuning": 50, "validation": 200, "final": 1000}


def build_split_manifest(examples: list[QAExample], seed: int = 520) -> dict[str, Any]:
    identifiers = [example.qid for example in examples]
    rng = random.Random(seed)
    rng.shuffle(identifiers)
    cursor = 0
    splits: dict[str, list[str]] = {}
    for name in ("smoke", "tuning", "validation", "final"):
        size = min(SPLIT_SIZES[name], max(0, len(identifiers) - cursor))
        splits[name] = identifiers[cursor : cursor + size]
        cursor += size
    return {"seed": seed, "total_available": len(examples), "splits": splits}


def build_nested_development_manifest(examples: list[QAExample], seed: int = 520) -> dict[str, Any]:
    """Nested diagnostic subsets for corpora too small for disjoint 20/50/200/1000.

    These are explicitly marked non-disjoint and must never be used for held-out
    validation claims. The default manifest remains disjoint.
    """
    identifiers = [example.qid for example in examples]
    rng = random.Random(seed)
    rng.shuffle(identifiers)
    return {
        "seed": seed,
        "total_available": len(examples),
        "non_disjoint_diagnostic_only": True,
        "splits": {name: identifiers[: min(size, len(identifiers))] for name, size in SPLIT_SIZES.items()},
    }


def select_split(examples: list[QAExample], name: str, manifest: dict[str, Any] | None = None, seed: int = 520) -> list[QAExample]:
    manifest = manifest or build_split_manifest(examples, seed)
    if name not in manifest["splits"]:
        raise ValueError(f"unknown split {name}")
    wanted = set(manifest["splits"][name])
    return [example for example in examples if example.qid in wanted]
