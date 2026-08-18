from __future__ import annotations

import argparse
import os
from config import TDCAConfig
from core_models import HeteroGraph
from knowledge_memory import EvidenceStore, MemoryBank
from llm_evaluator import MockLLM, OpenAICompatibleLLM, ValueEvaluator
from tdca_scheduler import TDCAScheduler
from utils import ensure_dir, set_seed, timestamp


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TDCA prototype runner")
    parser.add_argument("--config", type=str, default="", help="Optional JSON config path")
    parser.add_argument("--query", type=str, default="What is the birth city of the director of the movie Inception?")
    parser.add_argument("--mock_llm", action="store_true", help="Run with mock model for pipeline testing")
    parser.add_argument("--model_path", type=str, default="", help="Deprecated. Local model execution is disabled; use API config instead.")
    parser.add_argument("--evidence_path", type=str, default="", help="Override evidence corpus path")
    parser.add_argument("--memory_path", type=str, default="", help="Override memory bank path")
    parser.add_argument("--scheduler_mode", type=str, default="", help="tdca | greedy | uniform | no_diffusion")
    parser.add_argument("--scoring_mode", type=str, default="", help="hybrid | llm")
    parser.add_argument("--llm_backend", type=str, default="", help="mock | openai | openrouter")
    parser.add_argument("--openai_base_url", type=str, default="", help="OpenAI-compatible base URL, e.g. https://yh.m7ai.com/v1")
    parser.add_argument("--served_model_name", type=str, default="", help="Model name exposed by OpenAI-compatible server, e.g. gpt-5.4")
    parser.add_argument("--openai_api_key", type=str, default="", help="API key for OpenAI-compatible endpoint")
    parser.add_argument("--reasoning_effort", type=str, default="", help="Reasoning effort for GPT-5/o-series models; use none to disable hidden reasoning.")
    parser.add_argument("--openrouter_site_url", type=str, default="", help="Optional OpenRouter HTTP-Referer header")
    parser.add_argument("--openrouter_app_name", type=str, default="", help="Optional OpenRouter X-OpenRouter-Title header")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_total_generated_tokens", type=int, default=-1)
    parser.add_argument("--local_device", type=str, default="", help="auto | cuda | cpu")
    return parser


def load_config(args: argparse.Namespace) -> TDCAConfig:
    config = TDCAConfig.from_json(args.config) if args.config else TDCAConfig()

    if args.model_path:
        config.model_path = args.model_path
    if args.evidence_path:
        config.evidence_path = args.evidence_path
    if args.memory_path:
        config.memory_path = args.memory_path
    if args.scheduler_mode:
        config.scheduler_mode = args.scheduler_mode
    if args.scoring_mode:
        config.scoring_mode = args.scoring_mode
    if args.llm_backend:
        config.llm_backend = args.llm_backend
    if args.openai_base_url:
        config.openai_base_url = args.openai_base_url
    if args.served_model_name:
        config.served_model_name = args.served_model_name
    if args.openai_api_key:
        config.openai_api_key = args.openai_api_key
    if args.reasoning_effort:
        config.reasoning_effort = args.reasoning_effort
    if args.openrouter_site_url:
        config.openrouter_site_url = args.openrouter_site_url
    if args.openrouter_app_name:
        config.openrouter_app_name = args.openrouter_app_name
    if args.max_steps > 0:
        config.max_steps = args.max_steps
    if args.max_total_generated_tokens > 0:
        config.max_total_generated_tokens = args.max_total_generated_tokens
    if args.local_device:
        config.local_device = args.local_device

    if args.mock_llm:
        config.llm_backend = "mock"

    if config.llm_backend in {"openai", "openrouter"}:
        config.model_path = ""

    config.sync_algorithm_aliases()
    return config


def build_llm(config: TDCAConfig):
    if config.llm_backend == "mock":
        print("[TDCA] Using MockLLM for dry-run.")
        return MockLLM()

    if config.llm_backend in {"openai", "openrouter"}:
        backend_name = "OpenRouter" if config.llm_backend == "openrouter" else "OpenAI-compatible"
        print(f"[TDCA] Using {backend_name} backend @ {config.openai_base_url}")
        print(f"[TDCA] served_model_name = {config.served_model_name}")
        print(f"[TDCA] reasoning_effort = {config.reasoning_effort}")

        headers = {}
        provider_preferences = None
        if config.llm_backend == "openrouter":
            if config.openrouter_site_url:
                headers["HTTP-Referer"] = config.openrouter_site_url
            if config.openrouter_app_name:
                headers["X-OpenRouter-Title"] = config.openrouter_app_name
            provider_preferences = {
                "allow_fallbacks": bool(config.openrouter_allow_fallbacks),
            }
            if config.openrouter_data_collection:
                provider_preferences["data_collection"] = config.openrouter_data_collection

        return OpenAICompatibleLLM(
            model_name=config.served_model_name,
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            default_headers=headers,
            provider_preferences=provider_preferences,
            reasoning_effort=config.reasoning_effort,
        )

    raise ValueError(
        "Local model execution has been disabled for this project. "
        "Use --llm_backend openrouter or --llm_backend openai, or --mock_llm for smoke tests."
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args)

    if config.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", config.openai_api_key)
        os.environ.setdefault("OPENROUTER_API_KEY", config.openai_api_key)

    set_seed(config.seed)
    project_root = config.resolve_project_root()

    evidence_path = config.resolve_path(config.evidence_path)
    memory_path = config.resolve_path(config.memory_path)
    output_root = ensure_dir(config.resolve_path(config.output_root))
    run_dir = ensure_dir(output_root / timestamp())

    print(f"[TDCA] project_root = {project_root}")
    print(f"[TDCA] llm_backend  = {config.llm_backend}")
    print(f"[TDCA] algorithm    = {config.selected_algorithm}")
    if config.model_path:
        print(f"[TDCA] model_path   = {config.model_path}")
    print(f"[TDCA] evidence     = {evidence_path}")
    print(f"[TDCA] memory       = {memory_path}")
    print(f"[TDCA] outputs      = {run_dir}")

    if config.selected_algorithm != "tdca":
        raise ValueError(
            "main.py is the single-query TDCA entrypoint and only supports algorithm=tdca. "
            "Use tdca_batch_hotpotqa.py for batch baselines such as closed_book/sparse_rag/dense_rag/ircot."
        )

    llm = build_llm(config)

    graph = HeteroGraph()
    evidence_store = EvidenceStore(evidence_path)
    memory_bank = MemoryBank(memory_path)
    evaluator = ValueEvaluator(llm=llm, value_weights=config.value_weights)
    scheduler = TDCAScheduler(
        llm=llm,
        graph=graph,
        evaluator=evaluator,
        evidence_store=evidence_store,
        memory_bank=memory_bank,
        config=config,
    )

    result = scheduler.solve(question=args.query, output_dir=str(run_dir))

    print("\n========== TDCA FINAL RESULT ==========")
    print(result["final_answer"])
    print("\n========== TDCA STATS ==========")
    for key, value in result["stats"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
