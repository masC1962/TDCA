from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def _default_llm_backend() -> str:
    configured = _env_first("TDCA_LLM_BACKEND", "LLM_BACKEND", default="")
    if configured:
        return configured
    if _env_first("TDCA_LLM_BASE_URL", "LLM_BASE_URL", "DASHSCOPE_API_KEY", default=""):
        return "openai"
    return "openrouter"


@dataclass
class TDCAConfig:
    project_root: str = field(default_factory=lambda: os.getenv("TDCA_PROJECT_ROOT", "/workspace/TDCA"))
    model_path: str = field(default_factory=lambda: _env_first("TDCA_MODEL_PATH", "MODEL_PATH", default=""))
    evidence_path: str = "data/demo_corpus.jsonl"
    memory_path: str = "data/demo_memories.jsonl"
    output_root: str = "outputs"

    scheduler_mode: str = "tdca"
    scoring_mode: str = "hybrid"
    seed: int = 42

    llm_backend: str = field(default_factory=_default_llm_backend)

    # OpenAI-compatible endpoint settings. Generic LLM_* env vars work for
    # providers such as DashScope:
    #   LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    #   LLM_MODEL=qwen-plus
    #   DASHSCOPE_API_KEY=<key>
    openai_base_url: str = field(default_factory=lambda: _env_first(
        "TDCA_OPENAI_BASE_URL",
        "TDCA_LLM_BASE_URL",
        "TDCA_OPENROUTER_BASE_URL",
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "DASHSCOPE_BASE_URL",
        "OPENROUTER_BASE_URL",
        default="https://yh.m7ai.com/v1",
    ))
    openai_api_key: str = field(default_factory=lambda: _env_first(
        "TDCA_OPENAI_API_KEY",
        "TDCA_LLM_API_KEY",
        "TDCA_DASHSCOPE_API_KEY",
        "TDCA_OPENROUTER_API_KEY",
        "LLM_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        default="",
    ))
    served_model_name: str = field(default_factory=lambda: _env_first(
        "TDCA_SERVED_MODEL_NAME",
        "TDCA_LLM_MODEL",
        "TDCA_DASHSCOPE_MODEL",
        "TDCA_OPENROUTER_MODEL",
        "LLM_MODEL",
        "DASHSCOPE_MODEL",
        "SERVED_MODEL_NAME",
        "OPENROUTER_MODEL",
        "OPENAI_MODEL",
        default="gpt-5.4",
    ))
    reasoning_effort: str = field(default_factory=lambda: _env_first(
        "TDCA_REASONING_EFFORT",
        "REASONING_EFFORT",
        default="none",
    ))

    # Optional OpenRouter attribution/routing controls.
    openrouter_site_url: str = field(default_factory=lambda: _env_first(
        "TDCA_OPENROUTER_SITE_URL",
        "OPENROUTER_SITE_URL",
        default="",
    ))
    openrouter_app_name: str = field(default_factory=lambda: _env_first(
        "TDCA_OPENROUTER_APP_NAME",
        "OPENROUTER_APP_NAME",
        default="TDCA",
    ))
    openrouter_allow_fallbacks: bool = field(default_factory=lambda: _env_first(
        "TDCA_OPENROUTER_ALLOW_FALLBACKS",
        "OPENROUTER_ALLOW_FALLBACKS",
        default="true",
    ).strip().lower() not in {"0", "false", "no", "off"})
    openrouter_data_collection: Optional[str] = field(default_factory=lambda: _env_first(
        "TDCA_OPENROUTER_DATA_COLLECTION",
        "OPENROUTER_DATA_COLLECTION",
        default="deny",
    ) or None)

    max_steps: int = 18
    max_llm_calls: int = 40
    max_total_generated_tokens: int = 3000
    answer_synthesis_reserve_tokens: int = 96
    intermediate_generation_budget_fraction: float = 0.86
    open_goal_intermediate_budget_fraction: float = 0.94
    max_new_tokens_expand: int = 320
    max_new_tokens_score: int = 128
    max_new_tokens_answer: int = 96

    branching_factor: int = 3
    max_state_depth: int = 4
    retrieve_top_k_evidence: int = 4
    retrieve_top_k_memory: int = 2

    init_temperature_sigma: float = 0.18
    anneal_decay: float = 0.90
    diffuse_every: int = 1
    anneal_every: int = 4
    prune_every: int = 3
    prune_threshold: float = 0.10
    consume_gamma: float = 0.45
    lambda_diffusion: float = 0.45
    semantic_diffusion_enabled: bool = field(default_factory=lambda: _env_first(
        "TDCA_SEMANTIC_DIFFUSION_ENABLED",
        "SEMANTIC_DIFFUSION_ENABLED",
        default="true",
    ).strip().lower() not in {"0", "false", "no", "off"})
    semantic_diffusion_floor: float = 0.30
    semantic_diffusion_weight: float = 0.70

    support_boost: float = 0.20
    memory_boost: float = 0.10
    answer_bonus: float = 0.12
    support_reheat: float = 0.12
    memory_reheat: float = 0.08
    goal_residual_reheat: float = 0.34
    goal_terminal_reheat: float = 0.16
    goal_bridge_reheat: float = 0.10
    goal_composition_reheat: float = 0.28
    goal_operand_cooling: float = 0.72
    goal_answered_slot_cooling: float = 0.68
    goal_sibling_reheat: float = 0.18
    goal_frontier_slots_per_step: int = 2
    goal_frontier_min_open_slots: int = 2
    goal_slot_retry_max_visits: int = 2
    duplicate_merge_gain: float = 0.08
    min_stop_confidence: float = 0.78
    min_answer_value_to_stop: float = 0.62
    anytime_confidence_floor: float = 0.52
    anytime_min_steps: int = 1
    memory_write_min_value: float = 0.72
    memory_promote_min_support: float = 0.55
    verification_priority_decay: float = 0.55
    generic_derived_memory_decay: float = 0.92

    duplicate_state_threshold: float = 0.94
    state_keep_top_k: int = 6
    state_delete_min_depth: int = 1
    require_two_hop_grounding_for_nested: bool = True
    final_prune_on_exit: bool = True
    final_answer_judge_enabled: bool = field(default_factory=lambda: _env_first(
        "TDCA_FINAL_ANSWER_JUDGE_ENABLED",
        "FINAL_ANSWER_JUDGE_ENABLED",
        default="true",
    ).strip().lower() not in {"0", "false", "no", "off"})
    final_answer_judge_min_candidates: int = 2
    final_answer_judge_max_candidates: int = 5
    final_answer_judge_max_tokens: int = 96
    answer_rerank_enabled: bool = field(default_factory=lambda: _env_first(
        "TDCA_ANSWER_RERANK_ENABLED",
        "ANSWER_RERANK_ENABLED",
        default="true",
    ).strip().lower() not in {"0", "false", "no", "off"})
    answer_rerank_override_final: bool = field(default_factory=lambda: _env_first(
        "TDCA_ANSWER_RERANK_OVERRIDE_FINAL",
        "ANSWER_RERANK_OVERRIDE_FINAL",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    answer_rerank_base_weight: float = 0.42
    answer_rerank_evidence_weight: float = 0.18
    answer_rerank_type_weight: float = 0.12
    answer_rerank_root_weight: float = 0.10
    answer_rerank_coverage_weight: float = 0.10
    answer_rerank_semantic_weight: float = 0.08
    path_terminal_min_score: float = 0.48
    lightmem_final_chain_admission_enabled: bool = field(default_factory=lambda: _env_first(
        "TDCA_LIGHTMEM_FINAL_CHAIN_ADMISSION_ENABLED",
        "LIGHTMEM_FINAL_CHAIN_ADMISSION_ENABLED",
        default="true",
    ).strip().lower() not in {"0", "false", "no", "off"})
    lightmem_final_chain_min_support: float = 0.86
    lightmem_final_chain_min_target_overlap: float = 0.30
    enable_final_chain_buffer: bool = field(default_factory=lambda: _env_first(
        "TDCA_ENABLE_FINAL_CHAIN_BUFFER",
        "ENABLE_FINAL_CHAIN_BUFFER",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    enable_score_based_final_admission: bool = field(default_factory=lambda: _env_first(
        "TDCA_ENABLE_SCORE_BASED_FINAL_ADMISSION",
        "ENABLE_SCORE_BASED_FINAL_ADMISSION",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    final_chain_score_threshold: float = field(default_factory=lambda: float(_env_first(
        "TDCA_FINAL_CHAIN_SCORE_THRESHOLD",
        "FINAL_CHAIN_SCORE_THRESHOLD",
        default="0.72",
    )))
    enable_terminal_chain_closure: bool = field(default_factory=lambda: _env_first(
        "TDCA_ENABLE_TERMINAL_CHAIN_CLOSURE",
        "ENABLE_TERMINAL_CHAIN_CLOSURE",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    enable_tcc_final_audit: bool = field(default_factory=lambda: _env_first(
        "TDCA_ENABLE_TCC_FINAL_AUDIT",
        "ENABLE_TCC_FINAL_AUDIT",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    tcc_final_audit_mode: str = field(default_factory=lambda: _env_first(
        "TDCA_TCC_FINAL_AUDIT_MODE",
        "TCC_FINAL_AUDIT_MODE",
        default="audit_only",
    ))
    tcc_rerank_policy: str = field(default_factory=lambda: _env_first(
        "TDCA_TCC_RERANK_POLICY",
        "TCC_RERANK_POLICY",
        default="longhop_or_weak",
    ))
    tcc_score_threshold: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_SCORE_THRESHOLD",
        "TCC_SCORE_THRESHOLD",
        default="0.70",
    )))
    tcc_min_path_completeness: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_PATH_COMPLETENESS",
        "TCC_MIN_PATH_COMPLETENESS",
        default="0.45",
    )))
    tcc_min_dependency_closure: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_DEPENDENCY_CLOSURE",
        "TCC_MIN_DEPENDENCY_CLOSURE",
        default="0.45",
    )))
    tcc_min_last_hop_entailment: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_LAST_HOP_ENTAILMENT",
        "TCC_MIN_LAST_HOP_ENTAILMENT",
        default="0.50",
    )))
    tcc_min_terminality: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_TERMINALITY",
        "TCC_MIN_TERMINALITY",
        default="0.60",
    )))
    tcc_min_root_consistency: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_ROOT_CONSISTENCY",
        "TCC_MIN_ROOT_CONSISTENCY",
        default="0.55",
    )))
    tcc_min_dependency_closure_shorthop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_DEPENDENCY_CLOSURE_SHORTHOP",
        "TCC_MIN_DEPENDENCY_CLOSURE_SHORTHOP",
        default="0.30",
    )))
    tcc_min_root_consistency_shorthop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_ROOT_CONSISTENCY_SHORTHOP",
        "TCC_MIN_ROOT_CONSISTENCY_SHORTHOP",
        default="0.45",
    )))
    tcc_min_last_hop_entailment_shorthop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_LAST_HOP_ENTAILMENT_SHORTHOP",
        "TCC_MIN_LAST_HOP_ENTAILMENT_SHORTHOP",
        default="0.45",
    )))
    tcc_min_terminality_shorthop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_TERMINALITY_SHORTHOP",
        "TCC_MIN_TERMINALITY_SHORTHOP",
        default="0.55",
    )))
    tcc_min_dependency_closure_longhop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_DEPENDENCY_CLOSURE_LONGHOP",
        "TCC_MIN_DEPENDENCY_CLOSURE_LONGHOP",
        default="0.45",
    )))
    tcc_min_root_consistency_longhop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_ROOT_CONSISTENCY_LONGHOP",
        "TCC_MIN_ROOT_CONSISTENCY_LONGHOP",
        default="0.55",
    )))
    tcc_min_last_hop_entailment_longhop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_LAST_HOP_ENTAILMENT_LONGHOP",
        "TCC_MIN_LAST_HOP_ENTAILMENT_LONGHOP",
        default="0.50",
    )))
    tcc_min_terminality_longhop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_MIN_TERMINALITY_LONGHOP",
        "TCC_MIN_TERMINALITY_LONGHOP",
        default="0.60",
    )))
    enable_tcc_verified_promotion: bool = field(default_factory=lambda: _env_first(
        "TDCA_ENABLE_TCC_VERIFIED_PROMOTION",
        "ENABLE_TCC_VERIFIED_PROMOTION",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    tcc_promotion_policy: str = field(default_factory=lambda: _env_first(
        "TDCA_TCC_PROMOTION_POLICY",
        "TCC_PROMOTION_POLICY",
        default="empty_only_strict",
    ))
    tcc_promotion_min_hop: int = field(default_factory=lambda: int(_env_first(
        "TDCA_TCC_PROMOTION_MIN_HOP",
        "TCC_PROMOTION_MIN_HOP",
        default="3",
    )))
    allow_strict_2hop_promotion: bool = field(default_factory=lambda: _env_first(
        "TDCA_ALLOW_STRICT_2HOP_PROMOTION",
        "ALLOW_STRICT_2HOP_PROMOTION",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    tcc_promotion_score_threshold: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_PROMOTION_SCORE_THRESHOLD",
        "TCC_PROMOTION_SCORE_THRESHOLD",
        default="0.70",
    )))
    tcc_promotion_min_terminality: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_PROMOTION_MIN_TERMINALITY",
        "TCC_PROMOTION_MIN_TERMINALITY",
        default="0.60",
    )))
    tcc_promotion_min_root_consistency: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_PROMOTION_MIN_ROOT_CONSISTENCY",
        "TCC_PROMOTION_MIN_ROOT_CONSISTENCY",
        default="0.55",
    )))
    tcc_promotion_min_dependency_closure: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_PROMOTION_MIN_DEPENDENCY_CLOSURE",
        "TCC_PROMOTION_MIN_DEPENDENCY_CLOSURE",
        default="0.40",
    )))
    tcc_promotion_min_last_hop_entailment: float = field(default_factory=lambda: float(_env_first(
        "TDCA_TCC_PROMOTION_MIN_LAST_HOP_ENTAILMENT",
        "TCC_PROMOTION_MIN_LAST_HOP_ENTAILMENT",
        default="0.45",
    )))
    enable_terminal_memory_consolidation: bool = field(default_factory=lambda: _env_first(
        "TDCA_ENABLE_TERMINAL_MEMORY_CONSOLIDATION",
        "ENABLE_TERMINAL_MEMORY_CONSOLIDATION",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    enable_iterative_memory_construction: bool = field(default_factory=lambda: _env_first(
        "TDCA_ENABLE_ITERATIVE_MEMORY_CONSTRUCTION",
        "ENABLE_ITERATIVE_MEMORY_CONSTRUCTION",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    imc_max_rounds: int = field(default_factory=lambda: int(_env_first(
        "TDCA_IMC_MAX_ROUNDS",
        "IMC_MAX_ROUNDS",
        default="2",
    )))
    imc_max_repair_goals: int = field(default_factory=lambda: int(_env_first(
        "TDCA_IMC_MAX_REPAIR_GOALS",
        "IMC_MAX_REPAIR_GOALS",
        default="2",
    )))
    tmc_candidate_limit: int = field(default_factory=lambda: int(_env_first(
        "TDCA_TMC_CANDIDATE_LIMIT",
        "TMC_CANDIDATE_LIMIT",
        default="5",
    )))
    enable_anytime_fallback: bool = field(default_factory=lambda: _env_first(
        "TDCA_ENABLE_ANYTIME_FALLBACK",
        "ENABLE_ANYTIME_FALLBACK",
        default="false",
    ).strip().lower() not in {"0", "false", "no", "off"})
    anytime_fallback_threshold: float = field(default_factory=lambda: float(_env_first(
        "TDCA_ANYTIME_FALLBACK_THRESHOLD",
        "ANYTIME_FALLBACK_THRESHOLD",
        default="0.82",
    )))
    final_min_root_alignment: float = field(default_factory=lambda: float(_env_first(
        "TDCA_FINAL_MIN_ROOT_ALIGNMENT",
        "FINAL_MIN_ROOT_ALIGNMENT",
        default="0.55",
    )))
    final_min_dependency_satisfaction: float = field(default_factory=lambda: float(_env_first(
        "TDCA_FINAL_MIN_DEPENDENCY_SATISFACTION",
        "FINAL_MIN_DEPENDENCY_SATISFACTION",
        default="0.40",
    )))
    final_min_last_hop_support: float = field(default_factory=lambda: float(_env_first(
        "TDCA_FINAL_MIN_LAST_HOP_SUPPORT",
        "FINAL_MIN_LAST_HOP_SUPPORT",
        default="0.50",
    )))
    final_min_dependency_satisfaction_longhop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_FINAL_MIN_DEPENDENCY_SATISFACTION_LONGHOP",
        "FINAL_MIN_DEPENDENCY_SATISFACTION_LONGHOP",
        default="0.55",
    )))
    final_min_last_hop_support_longhop: float = field(default_factory=lambda: float(_env_first(
        "TDCA_FINAL_MIN_LAST_HOP_SUPPORT_LONGHOP",
        "FINAL_MIN_LAST_HOP_SUPPORT_LONGHOP",
        default="0.60",
    )))
    edge_weights: Dict[str, float] = field(default_factory=lambda: {
        "state_transition": 0.60,
        "refines": 0.55,
        "verifies": 0.50,
        "supports": 0.30,
        "recalls": 0.20,
        "derives": 0.40,
    })

    value_weights: Dict[str, float] = field(default_factory=lambda: {
        "task_progress": 0.38,
        "evidence_support": 0.28,
        "memory_usefulness": 0.12,
        "answerability": 0.17,
        "uncertainty_penalty": 0.05,
    })

    generation_temperature: float = 0.65
    score_temperature: float = 0.10
    use_bfloat16_if_available: bool = True
    local_device: str = "auto"
    gpu_min_free_gb_for_local: float = 8.0



    # Baseline / dataset runner settings
    algorithm: str = field(default_factory=lambda: _env_first(
        "TDCA_ALGORITHM",
        "ALGORITHM",
        default="tdca",
    ))
    baseline: str = ""
    dataset_name: str = "hotpotqa"
    retriever_type: str = "sparse"
    retriever_model_name: str = ""
    retriever_index_path: str = ""
    dense_encoder_path: str = ""
    reranker_type: str = "none"
    top_k: int = 5
    language: str = "en"
    ircot_max_steps: int = 4
    ircot_step_max_new_tokens: int = 512
    ircot_query_mode: str = "reasoning"
    baseline_output_dir: str = ""
    timestamped_output: bool = True
    run_tag: str = ""

    def sync_algorithm_aliases(self) -> str:
        chosen = (self.algorithm or self.baseline or "tdca").strip() or "tdca"
        self.algorithm = chosen
        self.baseline = chosen
        return chosen

    @property
    def selected_algorithm(self) -> str:
        return self.sync_algorithm_aliases()

    def resolve_project_root(self) -> Path:
        configured = Path(self.project_root)
        if configured.exists():
            return configured
        return Path(__file__).resolve().parent

    def resolve_path(self, maybe_relative: str) -> Path:
        path = Path(maybe_relative)
        if path.is_absolute():
            return path
        return self.resolve_project_root() / maybe_relative

    @classmethod
    def from_json(cls, config_path: str | os.PathLike[str]) -> "TDCAConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = cls(**data)
        config.sync_algorithm_aliases()
        return config

    def to_dict(self) -> dict:
        self.sync_algorithm_aliases()
        return asdict(self)

    def save_json(self, output_path: str | os.PathLike[str]) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
