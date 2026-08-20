from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from .config import ResearchConfig
from .dynamic.config import DynamicResearchConfig
from .runtime import run


def _load_config(path: str, method_override: str | None) -> ResearchConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    method = method_override or str(raw.get("method", "structured_tdca"))
    if method == "dynamic_hypergraph_tdca":
        allowed = {field.name for field in fields(DynamicResearchConfig)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown dynamic config fields: {unknown}")
        return DynamicResearchConfig(**(raw | {"method": method})).apply_ablation()
    return ResearchConfig.from_yaml(path)


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
    parser.add_argument("--max_llm_calls", type=int)
    parser.add_argument("--max_total_tokens", type=int)
    parser.add_argument("--max_retrieval_calls", type=int)
    parser.add_argument("--max_graph_operations", type=int)
    parser.add_argument("--memory_mode", choices=["none", "text", "typed"])
    parser.add_argument("--verifier", choices=["none", "self", "independent"])
    parser.add_argument("--finalization", choices=["direct", "structured"])
    parser.add_argument("--dynamic_ablation", choices=[f"A{i}" for i in range(1, 7)])
    parser.add_argument("--no_dependency_dag", action="store_true")
    parser.add_argument("--no_explicit_variable_binding", action="store_true")
    parser.add_argument("--oracle_evidence", action="store_true")
    parser.add_argument("--oracle_decomposition", action="store_true")
    parser.add_argument("--resume_dir", help="Resume an interrupted run from its exact durable checkpoint")
    args = parser.parse_args()
    loaded = _load_config(args.config, args.method)
    dynamic_overrides = {}
    if isinstance(loaded, DynamicResearchConfig):
        dynamic_overrides = {
            "dynamic_ablation": args.dynamic_ablation,
            "max_retrieval_calls": args.max_retrieval_calls,
            "max_graph_operations": args.max_graph_operations,
        }
    elif args.dynamic_ablation or args.max_retrieval_calls or args.max_graph_operations:
        parser.error("dynamic-only flags require --method dynamic_hypergraph_tdca")
    config = loaded.merged(
        method=args.method, dataset=args.dataset, dataset_path=args.dataset_path, setting=args.setting,
        split=args.split, split_manifest_path=args.split_manifest_path, retriever=args.retriever, scheduler=args.scheduler, top_k=args.top_k,
        memory_mode=args.memory_mode, verifier=args.verifier, finalization=args.finalization,
        use_dependency_dag=False if args.no_dependency_dag else None,
        explicit_variable_binding=False if args.no_explicit_variable_binding else None,
        oracle_evidence=True if args.oracle_evidence else None,
        oracle_decomposition=True if args.oracle_decomposition else None,
        max_llm_calls=args.max_llm_calls,
        max_total_tokens=args.max_total_tokens,
        **dynamic_overrides,
    )
    if isinstance(config, DynamicResearchConfig):
        config = config.apply_ablation()
    print(run(config, resume_dir=args.resume_dir))


if __name__ == "__main__":
    main()
