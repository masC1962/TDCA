from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..dynamic.graph import (
    AnswerNode,
    AnswerStatus,
    CandidateStatus,
    ClaimNode,
    DynamicReasoningHypergraph,
    GraphInvariantError,
    GraphLimits,
    _branch_from_dict,
    _hyperedge_from_dict,
    _node_from_dict,
    _operation_from_dict,
    _primitive,
    _revision_from_dict,
)
from ..utils import normalize_text, stable_hash


class TerminationKind(str, Enum):
    CONTINUE = "CONTINUE"
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


@dataclass
class ClaimSemantics:
    node_id: str
    subject_type: str
    value_type: str
    normalized_subject: str
    normalized_relation: str
    normalized_value: str
    qualifiers: dict[str, str] = field(default_factory=dict)
    extraction_mode: str = "typed_evidence_extraction"
    join_depth: int = 0
    join_signature: str = ""
    canonical_subject_id: str = ""
    canonical_value_id: str = ""
    subject_type_lineage: list[str] = field(default_factory=list)
    value_type_lineage: list[str] = field(default_factory=list)


@dataclass
class ActivatedPassageState:
    passage_id: str
    evidence_node_id: str
    subgoal_id: str
    branch_id: str
    query: str
    rank: int
    score: float
    entity_ids: list[str] = field(default_factory=list)


@dataclass
class ActivatedEntityState:
    entity_id: str
    canonical_name: str
    aliases: list[str]
    passage_ids: list[str]
    query_overlap: float


@dataclass
class CrossLayerEdge:
    source: str
    target: str
    edge_type: str


@dataclass
class BeliefState:
    """Separate belief and computation channels for one graph node."""

    absolute_support: float = 0.0
    relative_weight: float = 0.0
    entropy: float = 1.0
    evidence_gap: float = 1.0
    support_influence: float = 0.0
    contradiction_pressure: float = 0.0
    downstream_answer_impact: float = 0.0
    dependency_unlock_value: float = 0.0
    uncertainty: float = 1.0
    computation_heat: float = 0.0
    valid: bool = True
    version: int = 0
    updated_at_step: int = 0
    update_reason: str = "initial"


@dataclass
class DiffusionSnapshot:
    diffusion_id: str
    step: int
    seed_node_ids: list[str]
    iterations: int
    converged: bool
    max_delta: float
    channels_by_node: dict[str, dict[str, float]]
    typed_messages: list[dict[str, Any]]


@dataclass
class AllocationRecord:
    allocation_id: str
    operation_id: str
    step: int
    target_region: list[str]
    predicted_evc: float
    evc_components_raw: dict[str, float]
    evc_components_normalized: dict[str, float]
    requested_budget: dict[str, int]
    remaining_global_budget: dict[str, int]
    actual_cost: dict[str, float] = field(default_factory=dict)
    allocator_mode: str = "adaptive_evc"
    pre_state_summary: dict[str, float] = field(default_factory=dict)
    feedback_prior: dict[str, float] = field(default_factory=dict)
    post_state_summary: dict[str, float] = field(default_factory=dict)
    state_delta: dict[str, float] = field(default_factory=dict)
    actual_utility_components_raw: dict[str, float] = field(default_factory=dict)
    actual_utility_components_normalized: dict[str, float] = field(default_factory=dict)
    actual_utility: float = 0.0
    feedback_applied: bool = False
    selected: bool = False
    completed: bool = False
    failure_reason: str = ""
    fidelity_level: str = "medium"
    fidelity_fraction: float = 0.65
    predicted_immediate_utility: float = 0.0
    predicted_delayed_proof_return: float = 0.0
    predicted_normalized_cost: float = 0.0
    actual_immediate_utility: float = 0.0
    actual_normalized_cost: float = 0.0
    delayed_realized_proof_return: float = 0.0
    combined_realized_utility: float = 0.0
    credit_finalized: bool = False
    predicted_gross_opportunity: float = 0.0
    target_obligation_ids: list[str] = field(default_factory=list)
    obligation_estimate: dict[str, Any] = field(default_factory=dict)
    predicted_marginal_evc: float = 0.0
    predicted_provider_calls: int = 0
    critical_obligation_reserve: dict[str, int] = field(default_factory=dict)
    reserve_feasible: bool = True
    pre_target_obligation_statuses: dict[str, str] = field(default_factory=dict)
    actual_closed_target_ids: list[str] = field(default_factory=list)
    actual_target_closure_rate: float = 0.0
    actual_obligation_delta: float = 0.0
    transition_certificate: dict[str, Any] = field(default_factory=dict)
    predicted_transition_value: float = 0.0
    transition_realized: bool = False
    actual_transition_value: float = 0.0
    transition_observations: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalAttemptRecord:
    """Controller-owned ledger of every retrieval call, including zero-yield calls.

    Evidence nodes cannot represent an empty or duplicate-only retrieval.  Keeping
    attempts separately prevents the scheduler from mistaking such a call for an
    untried round and spending the remaining retrieval budget on the same query.
    """

    attempt_id: str
    operation_id: str
    step: int
    target_subgoal: str
    branch_id: str
    query: str
    normalized_query: str
    allocated_top_k: int
    hit_count: int
    new_evidence_count: int
    passage_ids: list[str] = field(default_factory=list)


@dataclass
class JoinAttemptRecord:
    """Auditable accepted or rejected conjunctive inference attempt."""

    attempt_id: str
    step: int
    operation_id: str
    target_subgoal: str
    branch_id: str
    premise_ids: list[str]
    premise_versions: dict[str, int]
    variable_bindings: dict[str, list[str]]
    constraints: list[dict[str, Any]]
    join_kind: str
    signature: str
    independent_support: dict[str, float]
    deterministic_validation: dict[str, Any]
    model_validation: dict[str, Any]
    accepted: bool
    proof_leaf_ids: list[str] = field(default_factory=list)
    conclusion_node_id: str = ""
    rejection_reason: str = ""
    creation_cost: dict[str, float] = field(default_factory=dict)
    downstream_unlock: float = 0.0


@dataclass
class OperationOutcomeRecord:
    """One selected computation and its measured within-question value."""

    outcome_id: str
    allocation_id: str
    operation_id: str
    step: int
    operation_family: str
    region_key: str
    pre_state_summary: dict[str, float]
    post_state_summary: dict[str, float]
    state_delta: dict[str, float]
    actual_utility_components_raw: dict[str, float]
    actual_utility_components_normalized: dict[str, float]
    actual_utility: float
    actual_cost: dict[str, float]
    progressed: bool
    failure_reason: str
    statistics_before: dict[str, float]
    statistics_after: dict[str, float]
    actual_immediate_utility: float = 0.0
    actual_normalized_cost: float = 0.0
    delayed_realized_proof_return: float = 0.0
    combined_realized_utility: float = 0.0


@dataclass(frozen=True)
class CreditAssignmentRecord:
    """Append-only, controller-owned causal credit observation.

    Every row is immutable.  Later graph mutations append a new observation for
    the source allocation instead of rewriting historical attribution.
    """

    credit_id: str
    allocation_id: str
    operation_id: str
    source_step: int
    observed_at_step: int
    gamma: float
    seed_node_ids: list[str]
    causal_descendant_ids: list[str]
    causal_distance_by_node: dict[str, int]
    delayed_components_raw: dict[str, float]
    delayed_components_normalized: dict[str, float]
    delayed_realized_proof_return: float
    causal_event_ids: list[str]
    terminal: bool = False
    attribution_version: str = "provenance-delayed-credit-v2.4.2"


@dataclass
class ProofObligationState:
    """Controller-owned statement of a graph-local proof deficit.

    This is not a probability.  It records what remains to be established,
    whether the deficit is still executable, and the graph events supporting
    that diagnosis.
    """

    obligation_id: str
    target_subgoal: str
    branch_id: str
    obligation_type: str
    status: str
    severity: float
    terminal_reachable: bool
    required_node_ids: list[str] = field(default_factory=list)
    satisfied_by_node_ids: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    provenance_event_ids: list[str] = field(default_factory=list)
    created_at_step: int = 0
    updated_at_step: int = 0


@dataclass(frozen=True)
class ProofObligationSnapshot:
    snapshot_id: str
    step: int
    trigger_operation_id: str
    obligations: list[dict[str, Any]]


@dataclass
class OperationFeedbackStats:
    """Conservative, deterministic posterior maintained only inside one graph."""

    observations: int = 0
    successes: int = 0
    no_ops: int = 0
    cumulative_utility: float = 0.0
    cumulative_cost: float = 0.0
    posterior_value: float = 0.5
    posterior_success: float = 0.5
    consecutive_failures: int = 0
    cooldown_until_step: int = 0


@dataclass
class SupersessionRecord:
    supersession_id: str
    step: int
    trigger: str
    trigger_source: str
    target_claim_id: str
    invalidated_node_ids: list[str]
    invalidated_hyperedge_ids: list[str]
    replacement_claim_id: str | None
    evidence_ids: list[str]
    natural: bool
    correctness_label: str = "pending"


@dataclass
class TerminationRecord:
    step: int
    outcome: TerminationKind
    best_predicted_evc: float
    answer_node_id: str | None
    reason: str
    remaining_budget: dict[str, int]
    dead_end_certificate: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminalBeliefState:
    """Auditable multi-channel readout for one terminal answer candidate.

    Acceptance is conjunctive over the channels below.  `terminal_gap` is a
    scheduling signal, not a replacement probability or an acceptance score.
    """

    answer_node_id: str
    candidate_answer: str
    branch_id: str
    absolute_support: float
    relative_weight: float
    entropy: float
    competition_entropy: float
    evidence_gap: float
    relative_margin: float
    contradiction_pressure: float
    answer_type_consistency: float
    chain_coverage: float
    terminal_gap: float
    proof_depth: int
    supporting_claims: list[str]
    supporting_evidence: list[str]
    raw_claim_channels: dict[str, dict[str, float]]
    sufficient_chain: bool
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)
    scoring_version: str = "terminal-belief-readout-v2.2"


@dataclass
class DynamicReasoningHypergraphV2(DynamicReasoningHypergraph):
    """V2 graph with sealed controller-owned mutable state.

    `controller_state_hash` is excluded from the state payload it seals.  Any
    mutation outside the V2 controller changes the payload without updating the
    seal and is rejected by `validate()`.
    """

    claim_semantics: dict[str, ClaimSemantics] = field(default_factory=dict)
    belief_states: dict[str, BeliefState] = field(default_factory=dict)
    diffusion_history: list[DiffusionSnapshot] = field(default_factory=list)
    allocation_history: list[AllocationRecord] = field(default_factory=list)
    retrieval_attempt_history: list[RetrievalAttemptRecord] = field(default_factory=list)
    join_attempt_history: list[JoinAttemptRecord] = field(default_factory=list)
    operation_outcome_history: list[OperationOutcomeRecord] = field(default_factory=list)
    credit_assignment_history: list[CreditAssignmentRecord] = field(default_factory=list)
    proof_obligations: dict[str, ProofObligationState] = field(default_factory=dict)
    proof_obligation_history: list[ProofObligationSnapshot] = field(default_factory=list)
    operation_feedback: dict[str, OperationFeedbackStats] = field(default_factory=dict)
    supersession_history: list[SupersessionRecord] = field(default_factory=list)
    invalidated_hyperedges: list[str] = field(default_factory=list)
    termination_history: list[TerminationRecord] = field(default_factory=list)
    terminal_beliefs: dict[str, TerminalBeliefState] = field(default_factory=dict)
    terminal_readout_version: str = "terminal-belief-readout-v2.2"
    corpus_memory_fingerprint: str = ""
    query_graph: dict[str, Any] = field(default_factory=dict)
    activated_passages: dict[str, ActivatedPassageState] = field(default_factory=dict)
    activated_entities: dict[str, ActivatedEntityState] = field(default_factory=dict)
    cross_layer_edges: list[CrossLayerEdge] = field(default_factory=list)
    proof_obligation_version: str = ""
    controller_state_hash: str = ""

    def state_payload(self) -> dict[str, Any]:
        payload = super().state_payload()
        payload.update({
            "claim_semantics": {
                key: _primitive(value) for key, value in sorted(self.claim_semantics.items())
            },
            "belief_states": {
                key: _primitive(value) for key, value in sorted(self.belief_states.items())
            },
            "diffusion_history": _primitive(self.diffusion_history),
            "allocation_history": _v243_compatible_allocations(
                self.allocation_history, self.proof_obligation_version
            ),
            "retrieval_attempt_history": _primitive(self.retrieval_attempt_history),
            "join_attempt_history": _primitive(self.join_attempt_history),
            "operation_outcome_history": _primitive(self.operation_outcome_history),
            "credit_assignment_history": _primitive(self.credit_assignment_history),
            "operation_feedback": {
                key: _primitive(value) for key, value in sorted(self.operation_feedback.items())
            },
            "supersession_history": _primitive(self.supersession_history),
            "invalidated_hyperedges": sorted(set(self.invalidated_hyperedges)),
            "termination_history": _v243_compatible_terminations(
                self.termination_history, bool(self.proof_obligation_version)
            ),
            "terminal_beliefs": {
                key: _primitive(value) for key, value in sorted(self.terminal_beliefs.items())
            },
            "terminal_readout_version": self.terminal_readout_version,
            "corpus_memory_fingerprint": self.corpus_memory_fingerprint,
            "query_graph": _primitive(self.query_graph),
            "activated_passages": {
                key: _primitive(value) for key, value in sorted(self.activated_passages.items())
            },
            "activated_entities": {
                key: _primitive(value) for key, value in sorted(self.activated_entities.items())
            },
            "cross_layer_edges": _primitive(self.cross_layer_edges),
        })
        if self.proof_obligation_version:
            payload.update({
                "proof_obligation_version": self.proof_obligation_version,
                "proof_obligations": {
                    key: _primitive(value)
                    for key, value in sorted(self.proof_obligations.items())
                },
                "proof_obligation_history": _primitive(self.proof_obligation_history),
            })
        return payload

    def to_dict(self) -> dict[str, Any]:
        return self.state_payload() | {
            "operation_history": _primitive(self.operation_history),
            "controller_state_hash": self.controller_state_hash,
            "graph_schema_version": "dynamic-hypergraph-v2",
        }

    def seal_controller_state(self) -> None:
        self.controller_state_hash = stable_hash(self.state_payload())

    def validate(self, *, allow_unsealed: bool = False) -> None:
        super().validate()
        for claim in self.claims():
            semantics = self.claim_semantics.get(claim.node_id)
            if semantics is None:
                raise GraphInvariantError(f"v2 claim {claim.node_id} lacks typed semantics")
            if semantics.node_id != claim.node_id:
                raise GraphInvariantError(f"claim semantics key/id mismatch: {claim.node_id}")
            if not semantics.normalized_value or not semantics.value_type:
                raise GraphInvariantError(f"claim {claim.node_id} has incomplete typed semantics")
            if semantics.join_depth > 0 and claim.status not in {
                CandidateStatus.INVALID, CandidateStatus.ARCHIVED,
            }:
                edges = [
                    edge for edge in self.hyperedges.values()
                    if edge.target_node == claim.node_id and edge.edge_id not in self.invalidated_hyperedges
                ]
                if not any(len(edge.source_node_set) >= 2 for edge in edges):
                    raise GraphInvariantError(f"joined claim {claim.node_id} lacks multi-premise hyperedge")
        for node_id, state in self.belief_states.items():
            if node_id not in self.nodes:
                raise GraphInvariantError(f"belief state references missing node {node_id}")
            for name in (
                "absolute_support", "relative_weight", "entropy", "evidence_gap",
                "support_influence", "contradiction_pressure", "downstream_answer_impact",
                "dependency_unlock_value", "uncertainty", "computation_heat",
            ):
                value = float(getattr(state, name))
                if not 0.0 <= value <= 1.0:
                    raise GraphInvariantError(f"belief channel {node_id}.{name} outside [0,1]")
        invalid_edges = set(self.invalidated_hyperedges)
        if any(edge_id not in self.hyperedges for edge_id in invalid_edges):
            raise GraphInvariantError("invalidated hyperedge id does not exist")
        for answer in self.answers():
            if answer.status != AnswerStatus.ACCEPTED:
                continue
            if answer.derivation_edge in invalid_edges:
                raise GraphInvariantError("accepted answer depends on invalidated hyperedge")
            for claim_id in answer.supporting_claims:
                claim = self.node(claim_id, ClaimNode)
                if claim.status in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}:
                    raise GraphInvariantError("accepted answer depends on invalidated claim")
            if self.terminal_readout_version:
                terminal = self.terminal_beliefs.get(answer.node_id)
                if terminal is None or not terminal.accepted or terminal.rejection_reasons:
                    raise GraphInvariantError("accepted v2.2 answer lacks passing terminal belief readout")
                if terminal.answer_node_id != answer.node_id:
                    raise GraphInvariantError("terminal belief answer id mismatch")
                if set(terminal.supporting_claims) != set(answer.supporting_claims):
                    raise GraphInvariantError("terminal belief claim support mismatch")
                if set(terminal.supporting_evidence) != set(answer.supporting_evidence):
                    raise GraphInvariantError("terminal belief evidence support mismatch")
        for answer_id, terminal in self.terminal_beliefs.items():
            if answer_id not in self.nodes or not isinstance(self.nodes[answer_id], AnswerNode):
                raise GraphInvariantError("terminal belief references missing answer")
            for name in (
                "absolute_support", "relative_weight", "entropy", "competition_entropy",
                "evidence_gap", "relative_margin", "contradiction_pressure",
                "answer_type_consistency", "chain_coverage", "terminal_gap",
            ):
                if not 0.0 <= float(getattr(terminal, name)) <= 1.0:
                    raise GraphInvariantError(f"terminal belief {answer_id}.{name} outside [0,1]")
            if terminal.proof_depth < 0:
                raise GraphInvariantError("terminal belief proof depth must be non-negative")
            if set(terminal.raw_claim_channels) != set(terminal.supporting_claims):
                raise GraphInvariantError("terminal belief lacks independent raw claim channels")
        allocation_ids = [row.allocation_id for row in self.allocation_history]
        if len(allocation_ids) != len(set(allocation_ids)):
            raise GraphInvariantError("allocation ids must be unique")
        for row in self.allocation_history:
            if row.predicted_evc < 0 or not row.evc_components_raw or not row.requested_budget:
                raise GraphInvariantError(f"allocation {row.allocation_id} lacks complete EVC trace")
            for name in (
                "predicted_immediate_utility", "predicted_delayed_proof_return",
                "predicted_normalized_cost", "actual_immediate_utility",
                "actual_normalized_cost", "delayed_realized_proof_return",
                "predicted_gross_opportunity",
            ):
                if not 0.0 <= float(getattr(row, name)) <= 1.0:
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id}.{name} outside [0,1]"
                    )
            if self.proof_obligation_version and any(
                value not in self.proof_obligations for value in row.target_obligation_ids
            ):
                raise GraphInvariantError(
                    f"allocation {row.allocation_id} targets unknown proof obligation"
                )
            if self.proof_obligation_version in {
                "proof-obligation-state-v2.4.3.1",
                "proof-obligation-state-v2.4.3.2",
            }:
                if not -1.0 <= float(row.predicted_marginal_evc) <= 1.0:
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id} marginal EVC outside [-1,1]"
                    )
                if row.predicted_provider_calls != int(
                    row.requested_budget.get("llm_calls", 0)
                ):
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id} provider-call prediction mismatch"
                    )
                if any(
                    value not in row.target_obligation_ids
                    for value in row.actual_closed_target_ids
                ):
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id} closes an untargeted obligation"
                    )
                if not 0.0 <= float(row.actual_target_closure_rate) <= 1.0:
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id} closure rate outside [0,1]"
                    )
                if not 0.0 <= float(row.actual_obligation_delta) <= 1.0:
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id} obligation delta outside [0,1]"
                    )
            if self.proof_obligation_version == "proof-obligation-state-v2.4.3.2":
                for name in ("predicted_transition_value", "actual_transition_value"):
                    if not 0.0 <= float(getattr(row, name)) <= 1.0:
                        raise GraphInvariantError(
                            f"allocation {row.allocation_id}.{name} outside [0,1]"
                        )
                if row.transition_certificate and (
                    row.transition_certificate.get("certificate_version")
                    != "certified-transition-option-v2.4.3.2"
                ):
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id} has invalid transition certificate"
                    )
            if not -1.0 <= float(row.combined_realized_utility) <= 1.0:
                raise GraphInvariantError(
                    f"allocation {row.allocation_id} combined utility outside [-1,1]"
                )
            if row.completed and not row.actual_cost:
                raise GraphInvariantError(f"completed allocation {row.allocation_id} lacks actual cost")
            if row.feedback_applied:
                if not row.pre_state_summary or not row.post_state_summary or not row.state_delta:
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id} lacks outcome-aware state delta"
                    )
                if not row.actual_utility_components_raw or not row.actual_utility_components_normalized:
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id} lacks normalized actual utility"
                    )
                if not -1.0 <= float(row.actual_utility) <= 1.0:
                    raise GraphInvariantError(
                        f"allocation {row.allocation_id} actual utility outside [-1,1]"
                    )
        retrieval_attempt_ids = [row.attempt_id for row in self.retrieval_attempt_history]
        if len(retrieval_attempt_ids) != len(set(retrieval_attempt_ids)):
            raise GraphInvariantError("retrieval attempt ids must be unique")
        for row in self.retrieval_attempt_history:
            if row.target_subgoal not in self.nodes:
                raise GraphInvariantError(
                    f"retrieval attempt {row.attempt_id} references missing subgoal"
                )
            if not row.query or row.normalized_query != normalize_text(row.query):
                raise GraphInvariantError(
                    f"retrieval attempt {row.attempt_id} has invalid normalized query"
                )
            if min(row.allocated_top_k, row.hit_count, row.new_evidence_count) < 0:
                raise GraphInvariantError(
                    f"retrieval attempt {row.attempt_id} has a negative counter"
                )
            if row.new_evidence_count > row.hit_count:
                raise GraphInvariantError(
                    f"retrieval attempt {row.attempt_id} creates more evidence than hits"
                )
            if len(row.passage_ids) != row.new_evidence_count:
                raise GraphInvariantError(
                    f"retrieval attempt {row.attempt_id} evidence count mismatch"
                )
        attempt_ids = [row.attempt_id for row in self.join_attempt_history]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise GraphInvariantError("JOIN attempt ids must be unique")
        for row in self.join_attempt_history:
            if len(row.premise_ids) < 2 or len(row.premise_ids) != len(set(row.premise_ids)):
                raise GraphInvariantError(f"JOIN attempt {row.attempt_id} has invalid premises")
            if any(node_id not in self.nodes for node_id in row.premise_ids):
                raise GraphInvariantError(f"JOIN attempt {row.attempt_id} references missing premise")
            if set(row.premise_versions) != set(row.premise_ids):
                raise GraphInvariantError(f"JOIN attempt {row.attempt_id} lacks premise versions")
            if any(not 0.0 <= float(value) <= 1.0 for value in row.independent_support.values()):
                raise GraphInvariantError(f"JOIN attempt {row.attempt_id} support outside [0,1]")
            if row.accepted and row.conclusion_node_id not in self.nodes:
                raise GraphInvariantError(f"accepted JOIN attempt {row.attempt_id} lacks conclusion")
            if row.proof_leaf_ids and any(node_id not in self.nodes for node_id in row.proof_leaf_ids):
                raise GraphInvariantError(f"JOIN attempt {row.attempt_id} has missing proof leaf")
        outcome_ids = [row.outcome_id for row in self.operation_outcome_history]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise GraphInvariantError("operation outcome ids must be unique")
        for row in self.operation_outcome_history:
            if row.allocation_id not in set(allocation_ids):
                raise GraphInvariantError(f"outcome {row.outcome_id} lacks allocation ledger row")
            if not -1.0 <= float(row.actual_utility) <= 1.0:
                raise GraphInvariantError(f"outcome {row.outcome_id} utility outside [-1,1]")
            for name in (
                "actual_immediate_utility", "actual_normalized_cost",
                "delayed_realized_proof_return",
            ):
                if not 0.0 <= float(getattr(row, name)) <= 1.0:
                    raise GraphInvariantError(f"outcome {row.outcome_id}.{name} outside [0,1]")
            if not -1.0 <= float(row.combined_realized_utility) <= 1.0:
                raise GraphInvariantError(
                    f"outcome {row.outcome_id} combined utility outside [-1,1]"
                )
        credit_ids = [row.credit_id for row in self.credit_assignment_history]
        if len(credit_ids) != len(set(credit_ids)):
            raise GraphInvariantError("credit assignment ids must be unique")
        observed_steps: dict[str, int] = {}
        for row in self.credit_assignment_history:
            if row.allocation_id not in set(allocation_ids):
                raise GraphInvariantError(
                    f"credit {row.credit_id} lacks allocation ledger row"
                )
            if row.observed_at_step < row.source_step:
                raise GraphInvariantError(f"credit {row.credit_id} precedes its source")
            if row.observed_at_step < observed_steps.get(row.allocation_id, -1):
                raise GraphInvariantError("credit observations must be append-only by step")
            observed_steps[row.allocation_id] = row.observed_at_step
            if not 0.0 <= float(row.gamma) <= 1.0:
                raise GraphInvariantError(f"credit {row.credit_id} gamma outside [0,1]")
            if not 0.0 <= float(row.delayed_realized_proof_return) <= 1.0:
                raise GraphInvariantError(f"credit {row.credit_id} return outside [0,1]")
            if set(row.causal_distance_by_node) != set(row.causal_descendant_ids):
                raise GraphInvariantError(f"credit {row.credit_id} causal distance mismatch")
            if any(int(value) < 0 for value in row.causal_distance_by_node.values()):
                raise GraphInvariantError(f"credit {row.credit_id} has negative distance")
        for obligation_id, row in self.proof_obligations.items():
            if obligation_id != row.obligation_id:
                raise GraphInvariantError("proof obligation key/id mismatch")
            if row.target_subgoal not in self.nodes:
                raise GraphInvariantError(
                    f"proof obligation {obligation_id} references missing subgoal"
                )
            if row.status not in {"OPEN", "CLOSED", "BLOCKED"}:
                raise GraphInvariantError(f"proof obligation {obligation_id} has invalid status")
            if not 0.0 <= float(row.severity) <= 1.0:
                raise GraphInvariantError(f"proof obligation {obligation_id} severity outside [0,1]")
            if any(node_id not in self.nodes for node_id in row.required_node_ids):
                raise GraphInvariantError(f"proof obligation {obligation_id} has missing requirement")
            if any(node_id not in self.nodes for node_id in row.satisfied_by_node_ids):
                raise GraphInvariantError(f"proof obligation {obligation_id} has missing satisfaction")
        for key, stats in self.operation_feedback.items():
            if stats.observations < 0 or stats.successes < 0 or stats.no_ops < 0:
                raise GraphInvariantError(f"negative feedback counter for {key}")
            if stats.successes > stats.observations or stats.no_ops > stats.observations:
                raise GraphInvariantError(f"invalid feedback counter for {key}")
            if not 0.0 <= stats.posterior_value <= 1.0:
                raise GraphInvariantError(f"feedback posterior value outside [0,1] for {key}")
            if not 0.0 <= stats.posterior_success <= 1.0:
                raise GraphInvariantError(f"feedback posterior success outside [0,1] for {key}")
        for evidence_id, row in self.activated_passages.items():
            if evidence_id != row.evidence_node_id or evidence_id not in self.nodes:
                raise GraphInvariantError("activated passage must reference its evidence node")
            if not 0.0 <= float(row.score) or row.rank <= 0:
                raise GraphInvariantError("activated passage has invalid retrieval state")
        for entity_id, row in self.activated_entities.items():
            if entity_id != row.entity_id or not row.canonical_name:
                raise GraphInvariantError("activated entity key/id or name is invalid")
            if not 0.0 <= float(row.query_overlap) <= 1.0:
                raise GraphInvariantError("activated entity overlap outside [0,1]")
        valid_cross_nodes = set(self.nodes) | set(self.activated_entities)
        for edge in self.cross_layer_edges:
            if edge.source not in valid_cross_nodes or edge.target not in valid_cross_nodes:
                raise GraphInvariantError("cross-layer edge references missing state")
        if self.query_graph:
            variables = self.query_graph.get("variables", [])
            constraints = self.query_graph.get("constraints", [])
            if not variables or not constraints:
                raise GraphInvariantError("query graph must contain variables and constraints")
        if self.controller_state_hash and not allow_unsealed:
            actual = stable_hash(self.state_payload())
            if actual != self.controller_state_hash:
                raise GraphInvariantError("graph state changed outside the V2 controller")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DynamicReasoningHypergraphV2":
        graph = cls(question=str(value["question"]), limits=GraphLimits(**value["limits"]))
        graph.nodes = {
            str(node_id): _node_from_dict(node_value)
            for node_id, node_value in value.get("nodes", {}).items()
        }
        graph.hyperedges = {
            str(edge_id): _hyperedge_from_dict(edge_value)
            for edge_id, edge_value in value.get("hyperedges", {}).items()
        }
        graph.branches = {
            str(branch_id): _branch_from_dict(branch_value)
            for branch_id, branch_value in value.get("branches", {}).items()
        }
        from ..dynamic.graph import ExecutionDependencyGraph

        graph.execution_graph = ExecutionDependencyGraph.from_dict(value.get("execution_graph", {}))
        graph.revision_history = [_revision_from_dict(row) for row in value.get("revision_history", [])]
        graph.operation_history = [_operation_from_dict(row) for row in value.get("operation_history", [])]
        graph.step = int(value.get("step", 0))
        graph.retrieval_calls = int(value.get("retrieval_calls", 0))
        graph.claim_semantics = {
            str(key): ClaimSemantics(**row) for key, row in value.get("claim_semantics", {}).items()
        }
        graph.belief_states = {
            str(key): BeliefState(**row) for key, row in value.get("belief_states", {}).items()
        }
        graph.diffusion_history = [DiffusionSnapshot(**row) for row in value.get("diffusion_history", [])]
        graph.allocation_history = [AllocationRecord(**row) for row in value.get("allocation_history", [])]
        graph.retrieval_attempt_history = [
            RetrievalAttemptRecord(**row)
            for row in value.get("retrieval_attempt_history", [])
        ]
        graph.join_attempt_history = [
            JoinAttemptRecord(**row) for row in value.get("join_attempt_history", [])
        ]
        graph.operation_outcome_history = [
            OperationOutcomeRecord(**row) for row in value.get("operation_outcome_history", [])
        ]
        graph.credit_assignment_history = [
            CreditAssignmentRecord(**row)
            for row in value.get("credit_assignment_history", [])
        ]
        graph.proof_obligations = {
            str(key): ProofObligationState(**row)
            for key, row in value.get("proof_obligations", {}).items()
        }
        graph.proof_obligation_history = [
            ProofObligationSnapshot(**row)
            for row in value.get("proof_obligation_history", [])
        ]
        graph.operation_feedback = {
            str(key): OperationFeedbackStats(**row)
            for key, row in value.get("operation_feedback", {}).items()
        }
        graph.supersession_history = [SupersessionRecord(**row) for row in value.get("supersession_history", [])]
        graph.invalidated_hyperedges = [str(item) for item in value.get("invalidated_hyperedges", [])]
        graph.termination_history = [
            TerminationRecord(
                **{**row, "outcome": TerminationKind(row["outcome"])}
            ) for row in value.get("termination_history", [])
        ]
        graph.terminal_beliefs = {
            str(key): TerminalBeliefState(**row)
            for key, row in value.get("terminal_beliefs", {}).items()
        }
        # An empty version keeps frozen v2/v2.1 artifacts readable.  Fresh v2.2
        # graphs use the dataclass default and require the new terminal invariant.
        graph.terminal_readout_version = str(value.get("terminal_readout_version", ""))
        graph.corpus_memory_fingerprint = str(value.get("corpus_memory_fingerprint", ""))
        graph.query_graph = dict(value.get("query_graph", {}))
        graph.activated_passages = {
            str(key): ActivatedPassageState(**row)
            for key, row in value.get("activated_passages", {}).items()
        }
        graph.activated_entities = {
            str(key): ActivatedEntityState(**row)
            for key, row in value.get("activated_entities", {}).items()
        }
        graph.cross_layer_edges = [
            CrossLayerEdge(**row) for row in value.get("cross_layer_edges", [])
        ]
        graph.proof_obligation_version = str(value.get("proof_obligation_version", ""))
        graph.controller_state_hash = str(value.get("controller_state_hash", ""))
        graph.validate()
        return graph


def _v243_compatible_allocations(
    rows: list[AllocationRecord], version: str,
) -> list[dict[str, Any]]:
    payload = [_primitive(row) for row in rows]
    v2431_fields = {
        "obligation_estimate", "predicted_marginal_evc",
        "predicted_provider_calls", "critical_obligation_reserve",
        "reserve_feasible", "pre_target_obligation_statuses",
        "actual_closed_target_ids", "actual_target_closure_rate",
        "actual_obligation_delta",
    }
    v2432_fields = {
        "transition_certificate", "predicted_transition_value",
        "transition_realized", "actual_transition_value",
        "transition_observations",
    }
    for row in payload:
        if version not in {
            "proof-obligation-state-v2.4.3.1",
            "proof-obligation-state-v2.4.3.2",
        }:
            for name in v2431_fields:
                row.pop(name, None)
        if version != "proof-obligation-state-v2.4.3.2":
            for name in v2432_fields:
                row.pop(name, None)
        if not version:
            row.pop("predicted_gross_opportunity", None)
            row.pop("target_obligation_ids", None)
    return payload


def _v243_compatible_terminations(
    rows: list[TerminationRecord], enabled: bool,
) -> list[dict[str, Any]]:
    payload = [_primitive(row) for row in rows]
    if enabled:
        return payload
    for row in payload:
        row.pop("dead_end_certificate", None)
    return payload
