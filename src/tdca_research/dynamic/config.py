from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from ..config import ResearchConfig


@dataclass
class DynamicResearchConfig(ResearchConfig):
    """Config for the isolated Dynamic Hypergraph method.

    Defaults are development starting points, not frozen scientific claims.  Every
    policy threshold is serialized into the resolved config and may only be tuned on
    the dedicated DH development split.
    """

    method: str = "dynamic_hypergraph_tdca"
    dynamic_ablation: str = "A6"

    max_retrieval_calls: int = 8
    max_graph_operations: int = 48
    max_policy_iterations: int = 192
    max_candidates_per_subgoal: int = 3
    max_active_branches: int = 3
    max_graph_nodes: int = 64
    max_hyperedges: int = 96
    max_graph_revisions: int = 4
    max_revision_per_candidate: int = 2
    max_graph_depth: int = 6
    max_retrieval_rounds_per_subgoal: int = 2

    candidate_temperature: float = 0.50
    branch_margin_threshold: float = 0.15
    branch_entropy_threshold: float = 0.55
    retain_support_threshold: float = 0.35
    commit_support_threshold: float = 0.70
    commit_margin_threshold: float = 0.20
    commit_entropy_threshold: float = 0.50
    contradiction_threshold: float = 0.70
    reopen_score_delta: float = 0.15
    revision_cooldown_steps: int = 1
    prune_value_threshold: float = 0.20

    score_weight_grounding: float = 0.20
    score_weight_entailment: float = 0.20
    score_weight_type_match: float = 0.20
    score_weight_dependency: float = 0.20
    score_weight_retrieval: float = 0.20
    score_weight_contradiction: float = 0.20

    utility_weight_uncertainty: float = 1.0
    utility_weight_unlock: float = 1.0
    utility_weight_answer_impact: float = 1.0
    utility_weight_novelty: float = 0.5
    utility_weight_recovery: float = 0.75
    utility_weight_cost: float = 1.0
    utility_weight_growth_risk: float = 0.5

    initial_plan_max_tokens: int = 500
    candidate_set_max_tokens: int = 700
    soft_verifier_max_tokens: int = 900
    soft_verifier_model_weight: float = 0.25
    graph_editor_max_tokens: int = 650
    terminal_derivation_max_tokens: int = 500

    enable_adaptive_planning: bool = True
    enable_candidate_preservation: bool = True
    enable_hyperedges: bool = True
    enable_soft_verification: bool = True
    enable_revision: bool = True
    enable_operation_scheduler: bool = True

    def validate(self) -> None:
        self._validate_common({"dynamic_hypergraph_tdca"})
        if self.oracle_evidence or self.oracle_decomposition:
            raise ValueError("Dynamic Hypergraph v1 does not mix oracle fields into normal inference")
        if self.dynamic_ablation not in {f"A{i}" for i in range(1, 7)}:
            raise ValueError("dynamic_ablation must be A1..A6; A0 uses structured_tdca")
        positive_ints = {
            "max_retrieval_calls": self.max_retrieval_calls,
            "max_graph_operations": self.max_graph_operations,
            "max_policy_iterations": self.max_policy_iterations,
            "max_candidates_per_subgoal": self.max_candidates_per_subgoal,
            "max_active_branches": self.max_active_branches,
            "max_graph_nodes": self.max_graph_nodes,
            "max_hyperedges": self.max_hyperedges,
            "max_graph_revisions": self.max_graph_revisions,
            "max_revision_per_candidate": self.max_revision_per_candidate,
            "max_graph_depth": self.max_graph_depth,
            "max_retrieval_rounds_per_subgoal": self.max_retrieval_rounds_per_subgoal,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"dynamic graph limits must be positive: {invalid}")
        unit_fields = {
            name: value for name, value in asdict(self).items()
            if name.endswith("_threshold") or name == "candidate_temperature"
        }
        if self.candidate_temperature <= 0:
            raise ValueError("candidate_temperature must be positive")
        for name, value in unit_fields.items():
            if name == "candidate_temperature":
                continue
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        score_weights = [
            self.score_weight_grounding, self.score_weight_entailment,
            self.score_weight_type_match, self.score_weight_dependency,
            self.score_weight_retrieval,
        ]
        if any(value < 0 for value in score_weights) or sum(score_weights) <= 0:
            raise ValueError("candidate score weights must be non-negative with positive total")
        if self.score_weight_contradiction < 0:
            raise ValueError("contradiction penalty must be non-negative")
        if not 0 <= self.soft_verifier_model_weight <= 1:
            raise ValueError("soft_verifier_model_weight must be in [0,1]")

    def apply_ablation(self) -> "DynamicResearchConfig":
        """Return cumulative A1..A6 feature flags without duplicating pipelines."""
        level = int(self.dynamic_ablation[1:])
        return self.merged(
            enable_adaptive_planning=level >= 1,
            enable_candidate_preservation=level >= 2,
            enable_hyperedges=level >= 3,
            enable_soft_verification=level >= 4,
            enable_revision=level >= 5,
            enable_operation_scheduler=level >= 6,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DynamicResearchConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown dynamic config fields: {unknown}")
        return cls(**data).apply_ablation()

    def merged(self, **overrides: Any) -> "DynamicResearchConfig":
        data = asdict(self)
        data.update({key: value for key, value in overrides.items() if value is not None})
        return DynamicResearchConfig(**data)
