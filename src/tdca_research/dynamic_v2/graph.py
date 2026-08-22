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
from ..utils import stable_hash


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
    join_attempt_history: list[JoinAttemptRecord] = field(default_factory=list)
    operation_outcome_history: list[OperationOutcomeRecord] = field(default_factory=list)
    operation_feedback: dict[str, OperationFeedbackStats] = field(default_factory=dict)
    supersession_history: list[SupersessionRecord] = field(default_factory=list)
    invalidated_hyperedges: list[str] = field(default_factory=list)
    termination_history: list[TerminationRecord] = field(default_factory=list)
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
            "allocation_history": _primitive(self.allocation_history),
            "join_attempt_history": _primitive(self.join_attempt_history),
            "operation_outcome_history": _primitive(self.operation_outcome_history),
            "operation_feedback": {
                key: _primitive(value) for key, value in sorted(self.operation_feedback.items())
            },
            "supersession_history": _primitive(self.supersession_history),
            "invalidated_hyperedges": sorted(set(self.invalidated_hyperedges)),
            "termination_history": _primitive(self.termination_history),
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
        allocation_ids = [row.allocation_id for row in self.allocation_history]
        if len(allocation_ids) != len(set(allocation_ids)):
            raise GraphInvariantError("allocation ids must be unique")
        for row in self.allocation_history:
            if row.predicted_evc < 0 or not row.evc_components_raw or not row.requested_budget:
                raise GraphInvariantError(f"allocation {row.allocation_id} lacks complete EVC trace")
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
        outcome_ids = [row.outcome_id for row in self.operation_outcome_history]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise GraphInvariantError("operation outcome ids must be unique")
        for row in self.operation_outcome_history:
            if row.allocation_id not in set(allocation_ids):
                raise GraphInvariantError(f"outcome {row.outcome_id} lacks allocation ledger row")
            if not -1.0 <= float(row.actual_utility) <= 1.0:
                raise GraphInvariantError(f"outcome {row.outcome_id} utility outside [-1,1]")
        for key, stats in self.operation_feedback.items():
            if stats.observations < 0 or stats.successes < 0 or stats.no_ops < 0:
                raise GraphInvariantError(f"negative feedback counter for {key}")
            if stats.successes > stats.observations or stats.no_ops > stats.observations:
                raise GraphInvariantError(f"invalid feedback counter for {key}")
            if not 0.0 <= stats.posterior_value <= 1.0:
                raise GraphInvariantError(f"feedback posterior value outside [0,1] for {key}")
            if not 0.0 <= stats.posterior_success <= 1.0:
                raise GraphInvariantError(f"feedback posterior success outside [0,1] for {key}")
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
        graph.join_attempt_history = [
            JoinAttemptRecord(**row) for row in value.get("join_attempt_history", [])
        ]
        graph.operation_outcome_history = [
            OperationOutcomeRecord(**row) for row in value.get("operation_outcome_history", [])
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
        graph.controller_state_hash = str(value.get("controller_state_hash", ""))
        graph.validate()
        return graph
