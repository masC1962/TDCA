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
    prompt_version: str = "dynamic-hypergraph-v2.2.3-immutable-root"
    architecture_version: str = "terminal-belief-gap-aware-proof-hypergraph-v2.2"

    max_extracted_claims_per_round: int = 8
    max_join_arity: int = 4
    max_join_proposals_per_step: int = 3
    max_join_attempts_per_question: int = 6
    max_join_depth: int = 4
    max_join_frontier_candidates: int = 48
    join_min_premise_support: float = 0.55
    typed_extraction_max_tokens: int = 900
    join_validation_max_tokens: int = 650
    extraction_focus_sentences_per_evidence: int = 3
    extraction_focus_min_chars: int = 120
    structured_output_recovery: bool = True
    relation_light_memory: bool = True
    query_graph_compiler: bool = True
    deterministic_goal_path_join: bool = True
    deterministic_enumeration_expansion: bool = False

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
    evc_weight_observed_value: float = 1.00
    evc_weight_failure_cooldown: float = 1.00
    evc_weight_terminal_gap: float = 1.25
    evc_weight_terminal_proximity: float = 0.75
    evc_weight_call_cost: float = 0.75
    evc_weight_token_cost: float = 0.75
    evc_weight_retrieval_cost: float = 0.75
    evc_weight_retrieval_saturation: float = 1.00
    meta_stop_evc_threshold: float = 0.08

    # v2.3 features are opt-in so frozen v2.2 experiment configurations retain
    # their original allocation semantics and remain reproducible.
    multi_resource_evc: bool = False
    hierarchical_within_question_feedback: bool = False
    retrieval_attempt_aware_scheduling: bool = False
    terminal_dependency_closure: bool = False
    focused_empty_extraction_recovery: bool = False
    goal_conditioned_join_frontier: bool = False

    # v2.4 structural efficiency features are independently opt-in.  Keeping
    # these disabled by default preserves frozen v2.2/v2.3 experiment arms.
    join_preallocation_feasibility_filter: bool = False
    region_level_retrieval_stopping: bool = False
    bounded_extraction_recovery: bool = False
    retrieval_query_max_token_overlap: float = 0.80

    # v2.4.1 proof-gap recovery features are opt-in.  Frozen v2.4 configs do
    # not observe these fields and therefore retain byte-for-byte policy intent.
    proof_gap_conditioned_recovery: bool = False
    proof_usable_target_gate: bool = False
    feasibility_reasoned_recovery: bool = False
    no_diff_editor_preallocation_gate: bool = False
    choice_conditioned_evc: bool = False
    evc_weight_proof_gap_reducibility: float = 1.25
    evc_weight_feasibility_unlock: float = 1.00

    # v2.4.2 separates the value realized by the selected mutation from value
    # realized by its later causal descendants.  The feature remains opt-in so
    # frozen v2.4.1 and earlier configurations retain their original policy.
    horizon_aware_evc: bool = False
    delayed_credit_assignment: bool = False
    evc_immediate_horizon_weight: float = 0.40
    evc_delayed_horizon_weight: float = 0.60
    delayed_credit_gamma: float = 0.85
    delayed_credit_weight_proof_completeness: float = 0.35
    delayed_credit_weight_candidate_availability: float = 0.20
    delayed_credit_weight_accepted_evidence: float = 0.20
    delayed_credit_weight_successful_join: float = 0.15
    delayed_credit_weight_supported_terminal_answer: float = 0.10
    delayed_structural_signal_weight: float = 0.25
    delayed_capacity_commit_answer: float = 0.00
    delayed_capacity_commit_claim: float = 0.75
    delayed_capacity_retrieve: float = 0.75
    delayed_capacity_verify: float = 0.65
    delayed_capacity_merge: float = 0.70
    delayed_capacity_extract: float = 0.50
    delayed_capacity_branch: float = 0.55
    delayed_capacity_expand: float = 0.45
    delayed_capacity_revise: float = 0.45
    delayed_capacity_prune: float = 0.10

    # Terminal acceptance is a conjunctive readout over independent belief
    # channels.  These are initial development values, never learned weights.
    terminal_min_absolute_support: float = 0.70
    terminal_min_relative_margin: float = 0.20
    terminal_max_entropy: float = 0.50
    terminal_max_evidence_gap: float = 0.50
    terminal_max_contradiction: float = 0.70
    terminal_min_type_consistency: float = 0.80
    terminal_min_chain_coverage: float = 1.00

    allocator_mode: str = "adaptive_evc"
    outcome_feedback_prior_strength: float = 2.0
    outcome_feedback_cooldown_failures: int = 2
    outcome_feedback_cooldown_steps: int = 2
    actual_utility_weight_uncertainty: float = 1.0
    actual_utility_weight_support: float = 1.0
    actual_utility_weight_evidence_gap: float = 1.0
    actual_utility_weight_entropy: float = 0.5
    actual_utility_weight_unlock: float = 0.75
    actual_utility_weight_novelty: float = 0.5
    actual_utility_weight_chain_progress: float = 1.0
    actual_utility_weight_contradiction_resolution: float = 0.75
    actual_utility_weight_terminal_gap: float = 1.25
    actual_utility_weight_cost: float = 1.0

    allocation_min_token_fraction: float = 0.35
    allocation_mid_token_fraction: float = 0.65
    allocation_high_heat_threshold: float = 0.67
    allocation_mid_heat_threshold: float = 0.34
    max_independent_verifications: int = 2
    max_adaptive_top_k: int = 15
    allocation_fidelity_levels: int = 3
    activation_entity_boost: float = 0.35

    revision_support_drop_threshold: float = 0.20
    revision_entropy_rise_threshold: float = 0.20
    revision_evidence_gap_rise_threshold: float = 0.20
    revision_lexical_support_override_threshold: float = 0.80
    natural_revision_precision_threshold: float = 0.80

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
            "max_join_frontier_candidates": self.max_join_frontier_candidates,
            "diffusion_steps": self.diffusion_steps,
            "max_independent_verifications": self.max_independent_verifications,
            "max_adaptive_top_k": self.max_adaptive_top_k,
            "extraction_focus_sentences_per_evidence": self.extraction_focus_sentences_per_evidence,
            "extraction_focus_min_chars": self.extraction_focus_min_chars,
            "allocation_fidelity_levels": self.allocation_fidelity_levels,
            "outcome_feedback_cooldown_failures": self.outcome_feedback_cooldown_failures,
            "outcome_feedback_cooldown_steps": self.outcome_feedback_cooldown_steps,
        }
        invalid = [name for name, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"v2 positive integer fields invalid: {invalid}")
        unit = {
            name: value for name, value in asdict(self).items()
            if name.startswith("diffusion_") and name not in {"diffusion_steps", "diffusion_min_delta"}
            or name.endswith("_threshold")
            or name.endswith("_fraction")
            or name.startswith("delayed_capacity_")
            or name in {
                "terminal_min_absolute_support", "terminal_min_relative_margin",
                "terminal_max_entropy", "terminal_max_evidence_gap",
                "terminal_max_contradiction", "terminal_min_type_consistency",
                "terminal_min_chain_coverage",
                "evc_immediate_horizon_weight", "evc_delayed_horizon_weight",
                "delayed_credit_gamma",
                "delayed_structural_signal_weight",
            }
        }
        for name, value in unit.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.diffusion_min_delta < 0:
            raise ValueError("diffusion_min_delta must be non-negative")
        if self.allocator_mode not in {"adaptive_evc", "uniform", "fixed_order"}:
            raise ValueError("allocator_mode must be adaptive_evc, uniform, or fixed_order")
        if self.outcome_feedback_prior_strength <= 0:
            raise ValueError("outcome_feedback_prior_strength must be positive")
        if not 0.0 <= self.join_min_premise_support <= 1.0:
            raise ValueError("join_min_premise_support must be in [0,1]")
        if not 0.0 <= self.activation_entity_boost <= 1.0:
            raise ValueError("activation_entity_boost must be in [0,1]")
        if not 0.0 <= self.retrieval_query_max_token_overlap <= 1.0:
            raise ValueError("retrieval_query_max_token_overlap must be in [0,1]")
        horizon_weight = (
            float(self.evc_immediate_horizon_weight)
            + float(self.evc_delayed_horizon_weight)
        )
        if abs(horizon_weight - 1.0) > 1e-9:
            raise ValueError("immediate and delayed EVC horizon weights must sum to 1")
        delayed_weights = [
            float(value) for name, value in asdict(self).items()
            if name.startswith("delayed_credit_weight_")
        ]
        if (
            not delayed_weights
            or any(value < 0.0 for value in delayed_weights)
            or sum(delayed_weights) <= 0.0
        ):
            raise ValueError("delayed credit component weights must have positive mass")
        weights = [
            value for name, value in asdict(self).items()
            if name.startswith("heat_weight_")
            or name.startswith("evc_weight_")
            or name.startswith("actual_utility_weight_")
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
