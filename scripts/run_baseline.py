#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import os
from baseline_batch_runner import run_baseline_batch
from config import TDCAConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified baseline runner for TDCA repo")
    parser.add_argument("--algorithm", choices=["closed_book", "sparse_rag", "dense_rag", "ircot"], default="")
    parser.add_argument("--baseline", choices=["closed_book", "sparse_rag", "dense_rag", "ircot"], default="")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", default="hotpotqa")
    parser.add_argument("--output_dir", required=True, help="Output root/name. A timestamp suffix is appended by default to avoid overwriting runs.")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--retriever_type", default="dense")
    parser.add_argument("--encoder_path", default="")
    parser.add_argument("--index_path", default="")
    parser.add_argument("--model_path", default="", help="Deprecated. Local model execution is disabled; use OpenRouter/OpenAI-compatible API settings instead.")
    parser.add_argument("--llm_backend", default=os.getenv("LLM_BACKEND", ""))
    parser.add_argument("--openai_base_url", default=os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "")))
    parser.add_argument("--openai_api_key", default=os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY", "")))))
    parser.add_argument("--served_model_name", default=os.getenv("LLM_MODEL", os.getenv("SERVED_MODEL_NAME", os.getenv("DASHSCOPE_MODEL", ""))))
    parser.add_argument("--reasoning_effort", default="none")
    parser.add_argument("--ircot_max_steps", type=int, default=4)
    parser.add_argument("--ircot_step_max_new_tokens", type=int, default=512, help="Per-step token budget for IRCoT intermediate reasoning/query generation.")
    parser.add_argument("--max_new_tokens_answer", type=int, default=1200, help="Completion token budget for baseline answer generation; default matches TDCA 1200-token runs.")
    parser.add_argument("--run_tag", default="", help="Optional extra tag appended to the timestamped output directory.")
    parser.add_argument("--no_timestamp_output", action="store_true", help="Write exactly to --output_dir instead of appending a timestamp suffix.")
    return parser


def make_config(args) -> TDCAConfig:
    config = TDCAConfig()
    if args.model_path:
        config.model_path = args.model_path
    if args.llm_backend:
        config.llm_backend = args.llm_backend
    if args.openai_base_url:
        config.openai_base_url = args.openai_base_url
    if args.openai_api_key:
        config.openai_api_key = args.openai_api_key
    if args.served_model_name:
        config.served_model_name = args.served_model_name
    if args.reasoning_effort:
        config.reasoning_effort = args.reasoning_effort
    if config.llm_backend in {"openai", "openrouter"}:
        config.model_path = ""
    config.algorithm = args.algorithm or args.baseline or config.algorithm
    config.baseline = args.baseline or config.algorithm or config.baseline
    config.dataset_name = args.dataset_name
    config.retriever_type = args.retriever_type
    config.dense_encoder_path = args.encoder_path
    config.retriever_index_path = args.index_path
    config.top_k = args.top_k
    config.ircot_max_steps = args.ircot_max_steps
    config.ircot_step_max_new_tokens = args.ircot_step_max_new_tokens
    config.max_new_tokens_answer = args.max_new_tokens_answer
    config.baseline_output_dir = args.output_dir
    config.timestamped_output = not args.no_timestamp_output
    config.run_tag = args.run_tag
    config.sync_algorithm_aliases()
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not (args.algorithm or args.baseline):
        parser.error("one of --algorithm or --baseline is required")
    config = make_config(args)
    run_baseline_batch(
        config=config,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        limit=args.limit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
