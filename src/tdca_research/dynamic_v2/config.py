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

    # v2.4.3 replaces choice-relative resource prices and operation-family
    # delayed priors with graph-local, auditable quantities.  The feature is
    # opt-in so every frozen v2.4.2 artifact keeps its original semantics.
    absolute_resource_cost: bool = False
    proof_obligation_tracking: bool = False
    graph_local_delayed_value: bool = False
    certified_meta_stop: bool = False
    absolute_cost_weight_call: float = 0.35
    absolute_cost_weight_token: float = 0.35
    absolute_cost_weight_retrieval: float = 0.20
    absolute_cost_weight_graph_risk: float = 0.10
    absolute_cost_scarcity_max_multiplier: float = 2.0
    graph_local_weight_obligation_closure: float = 0.30
    graph_local_weight_terminal_reachability: float = 0.25
    graph_local_weight_missing_premise_reduction: float = 0.20
    graph_local_weight_candidate_reachability: float = 0.15
    graph_local_weight_evidence_path: float = 0.10

    # v2.4.3.1 separates proof-obligation importance from the probability that
    # the concrete ready operation can close it.  Fidelity is costed with the
    # exact requested sample count and high fidelity is admitted only when its
    # marginal EVC is positive and preserves the critical-obligation reserve.
    operation_conditioned_obligation_closure: bool = False
    exact_fidelity_resource_accounting: bool = False
    marginal_fidelity_evc_gate: bool = False
    critical_obligation_budget_reserve: bool = False

    # v2.4.3.2 separates proof-deficit closure from certified execution-state
    # transitions.  It is opt-in to preserve every frozen v2.4.3.1 artifact.
    certified_transition_option_value: bool = False

    # v2.4.3.3 keeps the reasoning objective distinct from retrieval syntax and
    # predicts one-step value from normalized graph progress, not difficulty.
    preserve_subgoal_question_on_retrieval: bool = False
    one_step_progress_immediate_value: bool = False

    # v2.4.3.4 treats an independently accepted terminal readout as a
    # provider-free state transition.  The transition certificate recomputes
    # graph-local support and answer-set competition before it can bypass the
    # generic net-EVC threshold.
    certified_terminal_materialization: bool = False

    # v2.4.3.5 binds transition promises to the concrete fidelity-truncated
    # operation and recognizes claim provenance inherited by child branches.
    bind_transition_certificate_to_execution: bool = False
    terminal_certificate_accepts_ancestor_claims: bool = False
    feedback_conditioned_delayed_value: bool = False
    compact_objective_recovery_query: bool = False

    # v2.4.3.6 projects proof-usability failures into the same controller-owned
    # obligation ledger used by allocation and delayed credit.  It is opt-in so
    # every frozen v2.4.3.5 trace retains its original state semantics.
    proof_quality_obligation_alignment: bool = False
    anchored_proof_recovery_query: bool = False

    # v2.4.3.7 makes newly retrieved recovery evidence consumable before stale
    # proposed claims in the same region are re-verified.
    proof_recovery_extraction_priority: bool = False

    # v2.4.3.8 reconstructs recovery provenance from the controller-owned
    # allocation target ledger.  Concrete provider operations are deliberately
    # not trusted to preserve or assert this policy label.
    controller_derived_recovery_provenance: bool = False

    # v2.4.3.9 closes a schema-level alias gap for quantitative outputs while
    # preserving all frozen v2.4.3.8 projection decisions by default.
    numeric_output_type_normalization: bool = False

    # v2.4.3.9 retries a previously infeasible JOIN only after a premise field
    # used by the deterministic feasibility gate actually changes.
    semantic_join_attempt_state_key: bool = False

    # v2.4.3.10 proves whether a concrete JOIN can be materialized from sealed
    # independent scores without a provider request, then charges it exactly.
    certified_deterministic_join_allocation: bool = False

    # v2.4.3.11 repairs a representation loss at the extraction boundary.  Two
    # scalar endpoints are consolidated only when the same exact evidence span
    # explicitly states their interval and all relational provenance agrees.
    grounded_numeric_interval_consolidation: bool = False

    # v2.4.3.12 prevents structural deduplication from replacing the semantic
    # JOIN frontier order with an arbitrary stable-hash order.
    stable_join_frontier_priority: bool = False

    # Diagnostic-only trace.  This never changes candidates, allocation, or
    # graph state and is disabled in every result-bearing experiment arm.
    audit_join_frontier_selection: bool = False

    # v2.4.3.13 prefers the proof frontier with fewer unresolved endpoints
    # before using raw endpoint overlap as a tie breaker.
    prefer_minimal_join_open_frontier: bool = False

    # v2.4.3.14 gives a lossless, evidence-exact interval projection priority
    # over scalar fragments while leaving every other JOIN order untouched.
    grounded_interval_projection_priority: bool = False

    # v2.4.3.15 keeps the raw grounding channel independent and evidence-local:
    # an extracted dependent claim must literally anchor both tuple endpoints in
    # its own cited evidence span/title.  JOIN claims are
    # excluded because their support is audited through premise closure instead.
    evidence_endpoint_grounding: bool = False

    # v2.4.3.17 applies the endpoint audit only when an extraction turns a
    # universal/generic statement into an entity-specific dependent tuple.
    generic_evidence_endpoint_grounding: bool = False

    # v2.4.3.18 / HARA separates evidence truth from complete satisfaction of
    # the current query-graph constraint.  Structural dependency coverage may
    # override an unreliable model residual only when the controller can prove
    # the binding from candidate endpoints and declared dependency lineage.
    query_conditioned_semantic_alignment: bool = False
    structural_dependency_binding_coverage: bool = False
    # v2.4.3.20 replaces the unconditional second verifier request with a
    # controller-owned certificate over the compiled query graph.  Evidence
    # scoring remains an independent provider pass; relation, binding, output
    # and qualifier coverage are computed without reading evidence scores.
    controller_query_alignment_certificates: bool = False

    # v2.4.3.21 makes evidence grounding a necessary condition of absolute
    # support.  The original additive fusion remains frozen by default.
    grounding_conjunctive_absolute_support: bool = False

    # v2.4.3.22 prices verifier prompt context as well as its completion cap.
    # This prevents a high-fidelity multi-sample packet from fitting on paper
    # while exhausting the real per-question token budget mid-operation.
    prompt_inclusive_verifier_resource_accounting: bool = False

    # v2.4.3.27 keys extraction recovery to semantic prompt inputs instead of
    # diffusion-only belief versions, and preserves named query constraints.
    semantic_extraction_fingerprint: bool = False
    constraint_aware_query_entities: bool = False
    constraint_aware_direct_projection: bool = False

    # v2.4.3.28 preserves event-time arguments that a binary LLM extraction
    # can otherwise omit, canonicalizes only named constraint suffixes, and
    # forbids a composed path from inventing a different terminal output slot.
    deterministic_temporal_projection: bool = False
    constraint_projection_canonicalization: bool = False
    join_requires_verified_projection_premise: bool = False

    # v2.4.3.29 prevents lexical containment from conflating an entity with a
    # derived place (River != River Delta), certifies an exact typed wh-slot,
    # and hoists the projection-premise requirement before JOIN allocation.
    strict_query_endpoint_identity: bool = False
    controller_typed_output_consistency: bool = False
    terminal_min_relation_target_alignment: float = 0.70
    terminal_min_subject_binding_coverage: float = 0.70
    terminal_min_dependency_binding_coverage: float = 0.70
    terminal_min_qualifier_coverage: float = 0.70
    terminal_min_output_slot_coverage: float = 0.70
    terminal_min_full_subgoal_coverage: float = 0.70

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
        self._validate_common({"dynamic_hypergraph_tdca_v2", "hara"})
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
                "terminal_min_relation_target_alignment",
                "terminal_min_subject_binding_coverage",
                "terminal_min_dependency_binding_coverage",
                "terminal_min_qualifier_coverage",
                "terminal_min_output_slot_coverage",
                "terminal_min_full_subgoal_coverage",
                "evc_immediate_horizon_weight", "evc_delayed_horizon_weight",
                "delayed_credit_gamma",
                "delayed_structural_signal_weight",
                "absolute_cost_weight_call", "absolute_cost_weight_token",
                "absolute_cost_weight_retrieval", "absolute_cost_weight_graph_risk",
                "graph_local_weight_obligation_closure",
                "graph_local_weight_terminal_reachability",
                "graph_local_weight_missing_premise_reduction",
                "graph_local_weight_candidate_reachability",
                "graph_local_weight_evidence_path",
            }
        }
        for name, value in unit.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.diffusion_min_delta < 0:
            raise ValueError("diffusion_min_delta must be non-negative")
        if not 1.0 <= float(self.absolute_cost_scarcity_max_multiplier) <= 4.0:
            raise ValueError("absolute_cost_scarcity_max_multiplier must be in [1,4]")
        absolute_cost_mass = sum((
            self.absolute_cost_weight_call,
            self.absolute_cost_weight_token,
            self.absolute_cost_weight_retrieval,
            self.absolute_cost_weight_graph_risk,
        ))
        if abs(float(absolute_cost_mass) - 1.0) > 1e-9:
            raise ValueError("absolute resource cost weights must sum to 1")
        if self.operation_conditioned_obligation_closure and not (
            self.proof_obligation_tracking and self.graph_local_delayed_value
        ):
            raise ValueError(
                "operation-conditioned closure requires proof obligations and graph-local value"
            )
        if self.exact_fidelity_resource_accounting and not self.absolute_resource_cost:
            raise ValueError("exact fidelity accounting requires absolute resource cost")
        if self.marginal_fidelity_evc_gate and not (
            self.exact_fidelity_resource_accounting and self.horizon_aware_evc
        ):
            raise ValueError("marginal fidelity gate requires exact horizon accounting")
        if self.critical_obligation_budget_reserve and not self.proof_obligation_tracking:
            raise ValueError("critical obligation reserve requires proof obligation tracking")
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
