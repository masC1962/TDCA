from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ResearchConfig:
    method: str = "structured_tdca"
    dataset: str = "musique"
    dataset_path: str = "musique-main/musique-main/data/musique_ans_v1.0_dev.jsonl"
    global_corpus_path: str = ""
    setting: str = "distractor"
    split: str = "smoke"
    split_seed: int = 520
    split_manifest_path: str = ""
    output_root: str = "research_outputs"
    retriever: str = "bm25"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_fallback: str = "error"
    dense_index_path: str = ""
    top_k: int = 10
    max_steps: int = 8
    max_llm_calls: int = 16
    max_total_tokens: int = 16000
    final_reserve_tokens: int = 1200
    evidence_char_budget: int = 6000
    evidence_compaction: str = "none"
    scheduler: str = "expected_utility"
    beam_width: int = 3
    memory_mode: str = "typed"
    use_dependency_dag: bool = True
    explicit_variable_binding: bool = True
    diffusion_alpha: float = 0.25
    diffusion_decay: float = 0.90
    verifier: str = "independent"
    finalization: str = "structured"
    min_claim_confidence: float = 0.55
    min_answer_confidence: float = 0.58
    planner_max_tokens: int = 700
    claim_max_tokens: int = 500
    verifier_max_tokens: int = 350
    final_max_tokens: int = 300
    llm_backend: str = "openai"
    llm_base_url: str = ""
    llm_model: str = ""
    baseline_source: str = "native_controlled"
    baseline_commit: str = ""
    temperature: float = 0.0
    api_cache_dir: str = ".cache/tdca_research/llm"
    prompt_version: str = "wmgs-v4-dependency-grounded"
    request_timeout_seconds: float = 120.0
    max_api_attempts: int = 3
    campaign_id: str = ""
    campaign_ledger_path: str = ""
    campaign_provider_call_cap: int = 0
    campaign_provider_token_cap: int = 0
    isolate_api_cache_by_experiment_arm: bool = False
    oracle_evidence: bool = False
    oracle_decomposition: bool = False
    persistent_episodic_memory: bool = False

    def __post_init__(self) -> None:
        if not self.llm_base_url:
            self.llm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not self.llm_model:
            self.llm_model = os.getenv("LLM_MODEL", "qwen-plus")
        self.validate()

    def validate(self) -> None:
        self._validate_common({
            "structured_tdca", "closed_book", "bm25_rag", "dense_rag", "hybrid_rag", "ircot",
        })

    def _validate_common(self, supported_methods: set[str]) -> None:
        """Validate shared experiment fields without changing static semantics.

        Dynamic methods use a separate config subclass and pass their own explicit
        method set.  Keeping this helper here avoids weakening the static method
        whitelist or adding dynamic-only fields to historical resolved configs.
        """
        if self.method not in supported_methods:
            raise ValueError(
                f"method {self.method!r} is not a native controlled adapter; external official baselines "
                "must use python -m tdca_research.run_external"
            )
        if self.setting not in {"distractor", "global"}:
            raise ValueError("setting must be distractor or global")
        if self.setting == "global" and not self.global_corpus_path:
            raise ValueError("global setting requires global_corpus_path")
        valid_retrievers = {"bm25", "sparse", "dense", "hybrid", "entity", "entity_aware"}
        if self.retriever not in valid_retrievers:
            raise ValueError(f"retriever must be one of {sorted(valid_retrievers)}")
        valid_schedulers = {"greedy", "best_first", "beam", "diffusion", "tdca", "expected_utility", "structured_tdca"}
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {sorted(valid_schedulers)}")
        if self.dense_fallback not in {"error", "explicit_tfidf"}:
            raise ValueError("dense_fallback must be error or explicit_tfidf")
        if self.memory_mode not in {"none", "text", "typed"}:
            raise ValueError("memory_mode must be none, text or typed")
        if self.verifier not in {"none", "self", "independent"}:
            raise ValueError("verifier must be none, self or independent")
        if self.finalization not in {"direct", "structured"}:
            raise ValueError("finalization must be direct or structured")
        if self.top_k <= 0 or self.max_steps <= 0 or self.max_llm_calls <= 0:
            raise ValueError("budgets and top_k must be positive")
        campaign_values = (
            bool(self.campaign_id), bool(self.campaign_ledger_path),
            self.campaign_provider_call_cap > 0, self.campaign_provider_token_cap > 0,
        )
        if any(campaign_values) and not all(campaign_values):
            raise ValueError(
                "campaign_id, campaign_ledger_path, and both positive provider caps "
                "must be configured together"
            )
        if not 0 <= self.diffusion_alpha <= 1 or not 0 < self.diffusion_decay <= 1:
            raise ValueError("invalid diffusion parameters")
        if self.final_reserve_tokens >= self.max_total_tokens:
            raise ValueError("final reserve must be smaller than total token budget")
        if self.evidence_char_budget < 500:
            raise ValueError("evidence_char_budget is too small")
        if self.evidence_compaction not in {"none", "query_sentence"}:
            raise ValueError("evidence_compaction must be none or query_sentence")
        if self.request_timeout_seconds <= 0 or self.max_api_attempts <= 0:
            raise ValueError("request timeout and API attempts must be positive")
        if self.persistent_episodic_memory:
            raise ValueError("persistent episodic memory is intentionally disabled in this project")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ResearchConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown config fields: {unknown}")
        return cls(**data)

    def merged(self, **overrides: Any) -> "ResearchConfig":
        data = asdict(self)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return ResearchConfig(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
