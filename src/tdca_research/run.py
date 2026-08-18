from __future__ import annotations

import argparse

from .config import ResearchConfig
from .runtime import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured working-memory multi-hop QA experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method")
    parser.add_argument("--dataset")
    parser.add_argument("--dataset_path")
    parser.add_argument("--setting")
    parser.add_argument("--split")
    parser.add_argument("--split_manifest_path")
    parser.add_argument("--retriever")
    parser.add_argument("--scheduler")
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--memory_mode", choices=["none", "text", "typed"])
    parser.add_argument("--verifier", choices=["none", "self", "independent"])
    parser.add_argument("--finalization", choices=["direct", "structured"])
    parser.add_argument("--no_dependency_dag", action="store_true")
    parser.add_argument("--no_explicit_variable_binding", action="store_true")
    parser.add_argument("--oracle_evidence", action="store_true")
    parser.add_argument("--oracle_decomposition", action="store_true")
    parser.add_argument("--resume_dir", help="Resume an interrupted run from its exact durable checkpoint")
    args = parser.parse_args()
    config = ResearchConfig.from_yaml(args.config).merged(
        method=args.method, dataset=args.dataset, dataset_path=args.dataset_path, setting=args.setting,
        split=args.split, split_manifest_path=args.split_manifest_path, retriever=args.retriever, scheduler=args.scheduler, top_k=args.top_k,
        memory_mode=args.memory_mode, verifier=args.verifier, finalization=args.finalization,
        use_dependency_dag=False if args.no_dependency_dag else None,
        explicit_variable_binding=False if args.no_explicit_variable_binding else None,
        oracle_evidence=True if args.oracle_evidence else None,
        oracle_decomposition=True if args.oracle_decomposition else None,
    )
    print(run(config, resume_dir=args.resume_dir))


if __name__ == "__main__":
    main()
