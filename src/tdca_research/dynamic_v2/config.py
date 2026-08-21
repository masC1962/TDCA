from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from ..dynamic.config import DynamicResearchConfig


@dataclass
class DynamicV2ResearchConfig(DynamicResearchConfig):
    """Frozen-shape configuration for training-free hypergraph computation.

    Numeric defaults are initial development values.  Every value is serialized
    into run artifacts; selecting final values is allowed only on smoke-20 and
    development-50 before the heldout gate is opened.
    """

    method: str = "dynamic_hypergraph_tdca_v2"
    prompt_version: str = "dynamic-hypergraph-v2"

    max_extracted_claims_per_round: int = 6
    max_join_arity: int = 3
    max_join_proposals_per_step: int = 3
    max_join_attempts_per_question: int = 6
    max_join_depth: int = 4
    typed_extraction_max_tokens: int = 850
    join_validation_max_tokens: int = 650

    diffusion_steps: int = 3
    diffusion_restart: float = 0.40
    diffusion_decay: float = 0.65
    diffusion_min_delta: float = 1e-4
    heat_weight_uncertainty: float = 1.0
    heat_weight_answer_impact: float = 1.0
    heat_weight_evidence_gap: float = 0.75
    heat_weight_contradiction: float = 1.0
    heat_weight_unlock: float = 0.75

    evc_weight_heat: float = 1.0
    evc_weight_uncertainty_reduction: float = 1.0
    evc_weight_answer_impact: float = 1.0
    evc_weight_unlock: float = 0.75
    evc_weight_novelty: float = 0.50
    evc_weight_recovery: float = 0.75
    evc_weight_cost: float = 1.0
    evc_weight_growth_risk: float = 0.50
    meta_stop_evc_threshold: float = 0.08

    allocation_min_token_fraction: float = 0.35
    allocation_mid_token_fraction: float = 0.65
    allocation_high_heat_threshold: float = 0.67
    allocation_mid_heat_threshold: float = 0.34
    max_independent_verifications: int = 2
    max_adaptive_top_k: int = 15

    revision_support_drop_threshold: float = 0.20
    revision_entropy_rise_threshold: float = 0.20
    revision_evidence_gap_rise_threshold: float = 0.20
    natural_revision_precision_threshold: float = 0.60

    def validate(self) -> None:
        self._validate_common({"dynamic_hypergraph_tdca_v2"})
        if self.oracle_evidence or self.oracle_decomposition:
            raise ValueError("Dynamic Hypergraph v2 forbids oracle inference fields")
        positive = {
            "max_extracted_claims_per_round": self.max_extracted_claims_per_round,
            "max_join_arity": self.max_join_arity,
            "max_join_proposals_per_step": self.max_join_proposals_per_step,
            "max_join_attempts_per_question": self.max_join_attempts_per_question,
            "max_join_depth": self.max_join_depth,
            "diffusion_steps": self.diffusion_steps,
            "max_independent_verifications": self.max_independent_verifications,
            "max_adaptive_top_k": self.max_adaptive_top_k,
        }
        invalid = [name for name, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"v2 positive integer fields invalid: {invalid}")
        unit = {
            name: value for name, value in asdict(self).items()
            if name.startswith("diffusion_") and name not in {"diffusion_steps", "diffusion_min_delta"}
            or name.endswith("_threshold")
            or name.endswith("_fraction")
        }
        for name, value in unit.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.diffusion_min_delta < 0:
            raise ValueError("diffusion_min_delta must be non-negative")
        weights = [
            value for name, value in asdict(self).items()
            if name.startswith("heat_weight_") or name.startswith("evc_weight_")
        ]
        if any(float(value) < 0 for value in weights):
            raise ValueError("heat/EVC weights must be non-negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DynamicV2ResearchConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown dynamic v2 config fields: {unknown}")
        return cls(**data)

    def merged(self, **overrides: Any) -> "DynamicV2ResearchConfig":
        data = asdict(self)
        data.update({key: value for key, value in overrides.items() if value is not None})
        return DynamicV2ResearchConfig(**data)
