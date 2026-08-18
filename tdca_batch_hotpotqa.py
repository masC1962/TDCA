#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from answer_metrics import METRIC_KEYS, aggregate_metric_rows, compute_answer_metrics, exact_match as metric_exact_match
from batch_dataset_utils import extract_evidence_rows, extract_gold_titles, get_gold, get_gold_answers, get_question, iter_jsonl
from config import TDCAConfig
from core_models import HeteroGraph
from knowledge_memory import EvidenceStore, MemoryBank
from llm_evaluator import ValueEvaluator
from main import build_llm
from tdca_scheduler import TDCAScheduler
from utils import (
    ensure_dir,
    load_jsonl,
    save_jsonl,
    set_seed,
    timestamp,
    write_json,
)

DEFAULT_TEMPLATE_MEMORIES: List[Dict[str, Any]] = [
    {
        "id": "mem_1",
        "text": "Template: for compositional multi-hop questions, first solve the intermediate entity and then ask for the final attribute.",
        "score": 0.96,
        "tag": "compositional_template",
        "memory_kind": "template",
    },
    {
        "id": "mem_2",
        "text": "Template: for comparison questions, split the problem into one sub-question per entity before comparing them.",
        "score": 0.90,
        "tag": "comparison_template",
        "memory_kind": "template",
    },
    {
        "id": "mem_3",
        "text": "Template: when evidence already grounds the answer, switch from branching to short verification and synthesis.",
        "score": 0.88,
        "tag": "verification_then_stop",
        "memory_kind": "template",
    },
]


def load_config(args: argparse.Namespace) -> TDCAConfig:
    config = TDCAConfig.from_json(args.config) if args.config else TDCAConfig()
    if args.project_root:
        config.project_root = args.project_root
    if args.model_path:
        config.model_path = args.model_path
    if args.evidence_path:
        config.evidence_path = args.evidence_path
    if args.memory_path:
        config.memory_path = args.memory_path
    if args.output_root:
        config.output_root = args.output_root
    if getattr(args, "baseline_output_dir", ""):
        config.baseline_output_dir = args.baseline_output_dir
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
    if getattr(args, "openai_api_key", ""):
        config.openai_api_key = args.openai_api_key
    if getattr(args, "reasoning_effort", ""):
        config.reasoning_effort = args.reasoning_effort
    if args.max_steps > 0:
        config.max_steps = args.max_steps
    if args.local_device:
        config.local_device = args.local_device
    if args.max_total_generated_tokens > 0:
        config.max_total_generated_tokens = args.max_total_generated_tokens
    if getattr(args, "answer_synthesis_reserve_tokens", -1) > 0:
        config.answer_synthesis_reserve_tokens = args.answer_synthesis_reserve_tokens
    if getattr(args, "intermediate_generation_budget_fraction", -1.0) > 0:
        config.intermediate_generation_budget_fraction = args.intermediate_generation_budget_fraction
    if getattr(args, "open_goal_intermediate_budget_fraction", -1.0) > 0:
        config.open_goal_intermediate_budget_fraction = args.open_goal_intermediate_budget_fraction
    if getattr(args, "algorithm", ""):
        config.algorithm = args.algorithm
    if getattr(args, "baseline", ""):
        config.baseline = args.baseline
    if getattr(args, "dataset_name", ""):
        config.dataset_name = args.dataset_name
    if getattr(args, "retriever_type", ""):
        config.retriever_type = args.retriever_type
    if getattr(args, "encoder_path", ""):
        config.dense_encoder_path = args.encoder_path
    if getattr(args, "index_path", ""):
        config.retriever_index_path = args.index_path
    if getattr(args, "top_k", 0):
        config.top_k = args.top_k
    if getattr(args, "ircot_max_steps", 0):
        config.ircot_max_steps = args.ircot_max_steps
    if getattr(args, "ircot_step_max_new_tokens", 0):
        config.ircot_step_max_new_tokens = args.ircot_step_max_new_tokens
    if getattr(args, "max_new_tokens_answer", 0):
        config.max_new_tokens_answer = args.max_new_tokens_answer
    if bool(getattr(args, "enable_final_chain_buffer", False)):
        config.enable_final_chain_buffer = True
    if bool(getattr(args, "enable_score_based_final_admission", False)):
        config.enable_score_based_final_admission = True
    if getattr(args, "final_chain_score_threshold", -1.0) >= 0:
        config.final_chain_score_threshold = args.final_chain_score_threshold
    if bool(getattr(args, "enable_terminal_chain_closure", False)):
        config.enable_terminal_chain_closure = True
    if bool(getattr(args, "enable_tcc_final_audit", False)):
        config.enable_tcc_final_audit = True
    if getattr(args, "tcc_final_audit_mode", ""):
        config.tcc_final_audit_mode = args.tcc_final_audit_mode
    if getattr(args, "tcc_rerank_policy", ""):
        config.tcc_rerank_policy = args.tcc_rerank_policy
    if bool(getattr(args, "enable_tcc_verified_promotion", False)):
        config.enable_tcc_verified_promotion = True
    if getattr(args, "tcc_promotion_policy", ""):
        config.tcc_promotion_policy = args.tcc_promotion_policy
    if getattr(args, "tcc_promotion_min_hop", -1) >= 0:
        config.tcc_promotion_min_hop = args.tcc_promotion_min_hop
    if bool(getattr(args, "allow_strict_2hop_promotion", False)):
        config.allow_strict_2hop_promotion = True
    if bool(getattr(args, "enable_terminal_memory_consolidation", False)):
        config.enable_terminal_memory_consolidation = True
    if bool(getattr(args, "enable_iterative_memory_construction", False)):
        config.enable_iterative_memory_construction = True
    if getattr(args, "imc_max_rounds", -1) >= 0:
        config.imc_max_rounds = args.imc_max_rounds
    if getattr(args, "imc_max_repair_goals", -1) >= 0:
        config.imc_max_repair_goals = args.imc_max_repair_goals
    if getattr(args, "tmc_candidate_limit", -1) >= 0:
        config.tmc_candidate_limit = args.tmc_candidate_limit
    for arg_name, config_name in [
        ("tcc_score_threshold", "tcc_score_threshold"),
        ("tcc_min_path_completeness", "tcc_min_path_completeness"),
        ("tcc_min_dependency_closure", "tcc_min_dependency_closure"),
        ("tcc_min_last_hop_entailment", "tcc_min_last_hop_entailment"),
        ("tcc_min_terminality", "tcc_min_terminality"),
        ("tcc_min_root_consistency", "tcc_min_root_consistency"),
        ("tcc_min_dependency_closure_shorthop", "tcc_min_dependency_closure_shorthop"),
        ("tcc_min_root_consistency_shorthop", "tcc_min_root_consistency_shorthop"),
        ("tcc_min_last_hop_entailment_shorthop", "tcc_min_last_hop_entailment_shorthop"),
        ("tcc_min_terminality_shorthop", "tcc_min_terminality_shorthop"),
        ("tcc_min_dependency_closure_longhop", "tcc_min_dependency_closure_longhop"),
        ("tcc_min_root_consistency_longhop", "tcc_min_root_consistency_longhop"),
        ("tcc_min_last_hop_entailment_longhop", "tcc_min_last_hop_entailment_longhop"),
        ("tcc_min_terminality_longhop", "tcc_min_terminality_longhop"),
        ("tcc_promotion_score_threshold", "tcc_promotion_score_threshold"),
        ("tcc_promotion_min_terminality", "tcc_promotion_min_terminality"),
        ("tcc_promotion_min_root_consistency", "tcc_promotion_min_root_consistency"),
        ("tcc_promotion_min_dependency_closure", "tcc_promotion_min_dependency_closure"),
        ("tcc_promotion_min_last_hop_entailment", "tcc_promotion_min_last_hop_entailment"),
    ]:
        value = getattr(args, arg_name, -1.0)
        if value >= 0:
            setattr(config, config_name, value)
    if bool(getattr(args, "enable_anytime_fallback", False)):
        config.enable_anytime_fallback = True
    if getattr(args, "anytime_fallback_threshold", -1.0) >= 0:
        config.anytime_fallback_threshold = args.anytime_fallback_threshold
    for arg_name, config_name in [
        ("final_min_root_alignment", "final_min_root_alignment"),
        ("final_min_dependency_satisfaction", "final_min_dependency_satisfaction"),
        ("final_min_last_hop_support", "final_min_last_hop_support"),
        ("final_min_dependency_satisfaction_longhop", "final_min_dependency_satisfaction_longhop"),
        ("final_min_last_hop_support_longhop", "final_min_last_hop_support_longhop"),
    ]:
        value = getattr(args, arg_name, -1.0)
        if value >= 0:
            setattr(config, config_name, value)
    config.timestamped_output = not bool(getattr(args, "no_timestamp_output", False))
    config.run_tag = getattr(args, "run_tag", "") or config.run_tag
    config.sync_algorithm_aliases()
    return config
def build_runtime_evidence_file(item: Dict[str, Any], run_dir: Path, fallback_evidence_path: Path) -> Path:
    runtime_rows = extract_evidence_rows(item)
    evidence_file = run_dir / "runtime_evidence.jsonl"
    if runtime_rows:
        save_jsonl(evidence_file, runtime_rows)
        return evidence_file
    return fallback_evidence_path


def build_runtime_memory_file(base_memory_path: Path, run_dir: Path) -> Path:
    rows = load_jsonl(base_memory_path)
    template_rows = [
        row
        for row in rows
        if str(row.get("memory_kind", "")).strip().lower() == "template"
        and str(row.get("source", "")).strip().lower() != "tdca_run"
    ]
    if not template_rows:
        template_rows = DEFAULT_TEMPLATE_MEMORIES
    normalized_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(template_rows, start=1):
        normalized = dict(row)
        normalized["id"] = f"mem_{idx}"
        normalized_rows.append(normalized)
    memory_file = run_dir / "runtime_memories.jsonl"
    save_jsonl(memory_file, normalized_rows)
    return memory_file


def extract_answer_text(result: Dict[str, Any], question: str) -> str:
    final_answer = result.get("final_answer", "")
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer.strip()
    return ""


def normalize_short(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def graph_evidence_titles(graph: HeteroGraph) -> List[str]:
    titles: List[str] = []
    for node in graph.kg_nodes():
        title = str((node.metadata or {}).get("title") or "").strip()
        if title:
            titles.append(title)
    return list(dict.fromkeys(titles))


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-process batch runner for TDCA")
    parser.add_argument("--dataset", type=str, default="data/hotpotqa_subset_50.jsonl")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--project_root", type=str, default=os.getenv("TDCA_PROJECT_ROOT", "/workspace/TDCA"))
    parser.add_argument("--model_path", type=str, default=os.getenv("TDCA_MODEL_PATH", ""))
    parser.add_argument("--evidence_path", type=str, default="")
    parser.add_argument("--memory_path", type=str, default="")
    parser.add_argument("--output_root", type=str, default="batch_outputs")
    parser.add_argument("--baseline_output_dir", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--algorithm", type=str, default="", choices=["tdca", "closed_book", "sparse_rag", "dense_rag", "ircot"])
    parser.add_argument("--baseline", type=str, default="tdca", choices=["tdca", "closed_book", "sparse_rag", "dense_rag", "ircot"])
    parser.add_argument("--llm_backend", type=str, default=os.getenv("LLM_BACKEND", ""))
    parser.add_argument("--openai_base_url", type=str, default=os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "")))
    parser.add_argument("--served_model_name", type=str, default=os.getenv("LLM_MODEL", os.getenv("SERVED_MODEL_NAME", os.getenv("DASHSCOPE_MODEL", ""))))
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY", "")))))
    parser.add_argument("--reasoning_effort", type=str, default=os.getenv("REASONING_EFFORT", "none"))
    parser.add_argument("--scheduler_mode", type=str, default=os.getenv("SCHEDULER_MODE", "tdca"))
    parser.add_argument("--scoring_mode", type=str, default=os.getenv("SCORING_MODE", "hybrid"))
    parser.add_argument("--local_device", type=str, default=os.getenv("LOCAL_DEVICE", "auto"))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--retriever_type", type=str, default="dense")
    parser.add_argument("--encoder_path", type=str, default="")
    parser.add_argument("--index_path", type=str, default="")
    parser.add_argument("--ircot_max_steps", type=int, default=4)
    parser.add_argument("--ircot_step_max_new_tokens", type=int, default=512)
    parser.add_argument("--max_new_tokens_answer", type=int, default=1200)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--no_timestamp_output", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_total_generated_tokens", type=int, default=-1)
    parser.add_argument("--answer_synthesis_reserve_tokens", type=int, default=-1)
    parser.add_argument("--intermediate_generation_budget_fraction", type=float, default=-1.0)
    parser.add_argument("--open_goal_intermediate_budget_fraction", type=float, default=-1.0)
    parser.add_argument("--enable_final_chain_buffer", action="store_true", default=os.getenv("ENABLE_FINAL_CHAIN_BUFFER", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--enable_score_based_final_admission", action="store_true", default=os.getenv("ENABLE_SCORE_BASED_FINAL_ADMISSION", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--final_chain_score_threshold", type=float, default=float(os.getenv("FINAL_CHAIN_SCORE_THRESHOLD") or "-1"))
    parser.add_argument("--enable_terminal_chain_closure", action="store_true", default=os.getenv("ENABLE_TERMINAL_CHAIN_CLOSURE", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--enable_tcc_final_audit", action="store_true", default=os.getenv("ENABLE_TCC_FINAL_AUDIT", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--tcc_final_audit_mode", type=str, default=os.getenv("TCC_FINAL_AUDIT_MODE", ""))
    parser.add_argument("--tcc_rerank_policy", type=str, default=os.getenv("TCC_RERANK_POLICY", ""))
    parser.add_argument("--enable_tcc_verified_promotion", action="store_true", default=os.getenv("ENABLE_TCC_VERIFIED_PROMOTION", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--tcc_promotion_policy", type=str, default=os.getenv("TCC_PROMOTION_POLICY", ""))
    parser.add_argument("--tcc_promotion_min_hop", type=int, default=int(os.getenv("TCC_PROMOTION_MIN_HOP") or "-1"))
    parser.add_argument("--allow_strict_2hop_promotion", action="store_true", default=os.getenv("ALLOW_STRICT_2HOP_PROMOTION", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--tcc_promotion_score_threshold", type=float, default=float(os.getenv("TCC_PROMOTION_SCORE_THRESHOLD") or "-1"))
    parser.add_argument("--tcc_promotion_min_terminality", type=float, default=float(os.getenv("TCC_PROMOTION_MIN_TERMINALITY") or "-1"))
    parser.add_argument("--tcc_promotion_min_root_consistency", type=float, default=float(os.getenv("TCC_PROMOTION_MIN_ROOT_CONSISTENCY") or "-1"))
    parser.add_argument("--tcc_promotion_min_dependency_closure", type=float, default=float(os.getenv("TCC_PROMOTION_MIN_DEPENDENCY_CLOSURE") or "-1"))
    parser.add_argument("--tcc_promotion_min_last_hop_entailment", type=float, default=float(os.getenv("TCC_PROMOTION_MIN_LAST_HOP_ENTAILMENT") or "-1"))
    parser.add_argument("--enable_terminal_memory_consolidation", action="store_true", default=os.getenv("ENABLE_TERMINAL_MEMORY_CONSOLIDATION", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--enable_iterative_memory_construction", action="store_true", default=os.getenv("ENABLE_ITERATIVE_MEMORY_CONSTRUCTION", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--imc_max_rounds", type=int, default=int(os.getenv("IMC_MAX_ROUNDS") or "-1"))
    parser.add_argument("--imc_max_repair_goals", type=int, default=int(os.getenv("IMC_MAX_REPAIR_GOALS") or "-1"))
    parser.add_argument("--tmc_candidate_limit", type=int, default=int(os.getenv("TMC_CANDIDATE_LIMIT") or "-1"))
    parser.add_argument("--tcc_score_threshold", type=float, default=float(os.getenv("TCC_SCORE_THRESHOLD") or "-1"))
    parser.add_argument("--tcc_min_path_completeness", type=float, default=float(os.getenv("TCC_MIN_PATH_COMPLETENESS") or "-1"))
    parser.add_argument("--tcc_min_dependency_closure", type=float, default=float(os.getenv("TCC_MIN_DEPENDENCY_CLOSURE") or "-1"))
    parser.add_argument("--tcc_min_last_hop_entailment", type=float, default=float(os.getenv("TCC_MIN_LAST_HOP_ENTAILMENT") or "-1"))
    parser.add_argument("--tcc_min_terminality", type=float, default=float(os.getenv("TCC_MIN_TERMINALITY") or "-1"))
    parser.add_argument("--tcc_min_root_consistency", type=float, default=float(os.getenv("TCC_MIN_ROOT_CONSISTENCY") or "-1"))
    parser.add_argument("--tcc_min_dependency_closure_shorthop", type=float, default=float(os.getenv("TCC_MIN_DEPENDENCY_CLOSURE_SHORTHOP") or "-1"))
    parser.add_argument("--tcc_min_root_consistency_shorthop", type=float, default=float(os.getenv("TCC_MIN_ROOT_CONSISTENCY_SHORTHOP") or "-1"))
    parser.add_argument("--tcc_min_last_hop_entailment_shorthop", type=float, default=float(os.getenv("TCC_MIN_LAST_HOP_ENTAILMENT_SHORTHOP") or "-1"))
    parser.add_argument("--tcc_min_terminality_shorthop", type=float, default=float(os.getenv("TCC_MIN_TERMINALITY_SHORTHOP") or "-1"))
    parser.add_argument("--tcc_min_dependency_closure_longhop", type=float, default=float(os.getenv("TCC_MIN_DEPENDENCY_CLOSURE_LONGHOP") or "-1"))
    parser.add_argument("--tcc_min_root_consistency_longhop", type=float, default=float(os.getenv("TCC_MIN_ROOT_CONSISTENCY_LONGHOP") or "-1"))
    parser.add_argument("--tcc_min_last_hop_entailment_longhop", type=float, default=float(os.getenv("TCC_MIN_LAST_HOP_ENTAILMENT_LONGHOP") or "-1"))
    parser.add_argument("--tcc_min_terminality_longhop", type=float, default=float(os.getenv("TCC_MIN_TERMINALITY_LONGHOP") or "-1"))
    parser.add_argument("--enable_anytime_fallback", action="store_true", default=os.getenv("ENABLE_ANYTIME_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--anytime_fallback_threshold", type=float, default=float(os.getenv("ANYTIME_FALLBACK_THRESHOLD") or "-1"))
    parser.add_argument("--final_min_root_alignment", type=float, default=float(os.getenv("FINAL_MIN_ROOT_ALIGNMENT") or "-1"))
    parser.add_argument("--final_min_dependency_satisfaction", type=float, default=float(os.getenv("FINAL_MIN_DEPENDENCY_SATISFACTION") or "-1"))
    parser.add_argument("--final_min_last_hop_support", type=float, default=float(os.getenv("FINAL_MIN_LAST_HOP_SUPPORT") or "-1"))
    parser.add_argument("--final_min_dependency_satisfaction_longhop", type=float, default=float(os.getenv("FINAL_MIN_DEPENDENCY_SATISFACTION_LONGHOP") or "-1"))
    parser.add_argument("--final_min_last_hop_support_longhop", type=float, default=float(os.getenv("FINAL_MIN_LAST_HOP_SUPPORT_LONGHOP") or "-1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args)
    config.project_root = args.project_root
    set_seed(args.seed)

    project_root = config.resolve_project_root()
    dataset_path = config.resolve_path(args.dataset)

    if config.selected_algorithm != "tdca":
        from baseline_batch_runner import run_baseline_batch

        baseline_output_dir = config.baseline_output_dir or ""
        if not baseline_output_dir and args.output_root and args.output_root != "batch_outputs":
            baseline_output_dir = args.output_root
        run_baseline_batch(
            config=config,
            dataset_path=str(dataset_path),
            dataset_name=config.dataset_name,
            limit=args.limit,
            output_dir=baseline_output_dir or None,
        )
        return

    fallback_evidence_path = config.resolve_path(config.evidence_path)
    base_memory_path = config.resolve_path(config.memory_path)
    batch_root = ensure_dir(config.resolve_path(config.output_root) / f"{dataset_path.stem}_{timestamp()}")
    ensure_dir(batch_root / "runs")

    print(f"[TDCA-BATCH] project_root = {project_root}")
    print(f"[TDCA-BATCH] dataset      = {dataset_path}")
    print(f"[TDCA-BATCH] fallback evidence = {fallback_evidence_path}")
    print(f"[TDCA-BATCH] base memory  = {base_memory_path}")
    print(f"[TDCA-BATCH] outputs      = {batch_root}")
    print(f"[TDCA-BATCH] backend      = {config.llm_backend}")
    print(f"[TDCA-BATCH] algorithm    = {config.selected_algorithm}")
    print(f"[TDCA-BATCH] start_index  = {args.start_index}")

    llm = build_llm(config)

    rows: List[Dict[str, Any]] = []
    all_items = list(iter_jsonl(dataset_path))
    start_index = max(0, args.start_index)
    items = all_items[start_index : start_index + args.limit]

    for idx, item in enumerate(items, start=start_index):
        question = get_question(item)
        gold = get_gold(item)
        gold_answers = get_gold_answers(item)
        gold_titles = extract_gold_titles(item)
        sample_id = str(item.get("id", idx))
        run_dir = ensure_dir(batch_root / "runs" / f"{idx:03d}_{sample_id}")

        evidence_path = build_runtime_evidence_file(item, run_dir, fallback_evidence_path)
        memory_path = build_runtime_memory_file(base_memory_path, run_dir)

        graph = HeteroGraph()
        evidence_store = EvidenceStore(
            evidence_path,
            retriever_type=config.retriever_type,
            encoder_path=config.dense_encoder_path,
            index_path=config.retriever_index_path,
        )
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

        llm.call_count = 0
        llm.total_generated_tokens = 0

        print(f"\n[{idx+1}/{len(items)}] {question}")
        if evidence_path == fallback_evidence_path:
            print("[TDCA-BATCH] WARNING: no per-sample context detected; falling back to shared evidence corpus.")
        result = scheduler.solve(question=question, output_dir=str(run_dir))
        pred = extract_answer_text(result, question)
        exact_match = metric_exact_match(pred, gold_answers) if gold_answers else None
        retrieved_titles = graph_evidence_titles(graph)

        row = {
            "index": idx,
            "sample_id": sample_id,
            "question": question,
            "gold": gold,
            "pred": pred,
            "exact_match": exact_match,
            "retrieved_titles": json.dumps(retrieved_titles, ensure_ascii=False),
            "gold_titles": json.dumps(gold_titles, ensure_ascii=False),
            "steps": result.get("stats", {}).get("steps"),
            "llm_calls": result.get("stats", {}).get("llm_calls"),
            "generated_tokens": result.get("stats", {}).get("generated_tokens"),
            "stop_reason": result.get("stats", {}).get("stop_reason", ""),
            "anytime_answer": result.get("anytime_answer", ""),
            "anytime_answer_score": result.get("anytime_answer_score", 0.0),
            "anytime_answer_source": result.get("anytime_answer_source", ""),
            "best_node_type": (result.get("best_node") or {}).get("node_type", ""),
            "best_node_id": (result.get("best_node") or {}).get("node_id", ""),
            "run_dir": str(run_dir),
        }
        diagnostics = result.get("final_diagnostics", {})
        if isinstance(diagnostics, dict):
            row.update(diagnostics)
        row.update(compute_answer_metrics(pred, gold_answers, row))
        rows.append(row)

    summary_jsonl = batch_root / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_csv = batch_root / "summary.csv"
    fieldnames = [
        "index",
        "sample_id",
        "question",
        "gold",
        "pred",
        "exact_match",
        *METRIC_KEYS,
        "retrieved_titles",
        "gold_titles",
        "steps",
        "llm_calls",
        "generated_tokens",
        "stop_reason",
        "anytime_answer",
        "anytime_answer_score",
        "anytime_answer_source",
        "best_node_type",
        "best_node_id",
        "run_dir",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    em_values = [r["exact_match"] for r in rows if r["exact_match"] is not None]
    aggregate = {
        "count": len(rows),
        "answered_with_gold": len(em_values),
        "exact_match": (sum(em_values) / len(em_values)) if em_values else None,
        **aggregate_metric_rows(rows),
        "avg_steps": (sum((r["steps"] or 0) for r in rows) / len(rows)) if rows else None,
        "avg_llm_calls": (sum((r["llm_calls"] or 0) for r in rows) / len(rows)) if rows else None,
        "avg_generated_tokens": (sum((r["generated_tokens"] or 0) for r in rows) / len(rows)) if rows else None,
    }
    write_json(batch_root / "aggregate.json", aggregate)

    print("\n[TDCA-BATCH] aggregate:")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"[TDCA-BATCH] summary.csv  = {summary_csv}")
    print(f"[TDCA-BATCH] summary.jsonl= {summary_jsonl}")


if __name__ == "__main__":
    main()
