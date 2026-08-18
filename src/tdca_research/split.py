from __future__ import annotations

import argparse

from .data import build_nested_development_manifest, build_split_manifest, load_examples
from .utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--nested_diagnostic", action="store_true")
    args = parser.parse_args()
    examples = load_examples(args.dataset_path, args.dataset)
    manifest = build_nested_development_manifest(examples, args.seed) if args.nested_diagnostic else build_split_manifest(examples, args.seed)
    write_json(args.output, manifest)


if __name__ == "__main__":
    main()

