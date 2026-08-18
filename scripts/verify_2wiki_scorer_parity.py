#!/usr/bin/env python3
"""Compare unified answer metrics with the pinned official 2Wiki scorer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from tdca_research.evaluation import exact_match, token_f1


# Non-ASCII cases use escapes so transport/editor encoding cannot corrupt the fixture.
CASES = (
    ("The Eiffel Tower", "Eiffel Tower"),
    ("Paris, France", "Paris France"),
    ("yes indeed", "yes"),
    ("no", "yes"),
    ("no answer", "noanswer"),
    ("Ma\u0142gorzata Braunek!", "Ma\u0142gorzata Braunek"),
    ("The caf\u00e9, Paris", "caf\u00e9 Paris"),
    ("", "answer"),
    ("", ""),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official",
        default="external_repos/2wikimultihop/2wikimultihop_evaluate.py",
    )
    args = parser.parse_args()
    source = Path(args.official)
    spec = importlib.util.spec_from_file_location("official_2wiki_scorer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official scorer: {source}")
    module = importlib.util.module_from_spec(spec)
    # The official script imports ujson, but its answer functions only require the
    # standard JSON API. Avoid changing the validated project environment solely
    # for this source-level parity check.
    original_ujson = sys.modules.get("ujson")
    sys.modules["ujson"] = json
    try:
        spec.loader.exec_module(module)
    finally:
        if original_ujson is None:
            sys.modules.pop("ujson", None)
        else:
            sys.modules["ujson"] = original_ujson

    rows = []
    for prediction, gold in CASES:
        official_f1 = float(module.f1_score(prediction, gold)[0])
        official_em = float(module.exact_match_score(prediction, gold))
        local_f1 = token_f1(prediction, [gold])
        local_em = exact_match(prediction, [gold])
        row = {
            "prediction": prediction,
            "gold": gold,
            "official_em": official_em,
            "local_em": local_em,
            "official_f1": official_f1,
            "local_f1": local_f1,
        }
        rows.append(row)
        if official_em != local_em or abs(official_f1 - local_f1) > 1e-12:
            raise AssertionError(json.dumps(row, ensure_ascii=False))
    print(json.dumps({"cases": len(rows), "parity": True, "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
