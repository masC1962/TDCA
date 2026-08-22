from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from ..budget import Budget
from ..dynamic.graph import (
    AnswerStatus,
    CandidateStatus,
    ClaimNode,
    EvidenceNode,
    GraphOperation,
    OperationType,
    SubgoalNode,
    SubgoalStatus,
)
from ..utils import stable_hash
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2, OperationFeedbackStats


@dataclass(frozen=True)
class EVCSignals:
    graph_heat: float = 0.0
    uncertainty_reduction: float = 0.0
    answer_impact: float = 0.0
    dependency_unlock: float = 0.0
    evidence_novelty: float = 0.0
    recovery_value: float = 0.0
    observed_value: float = 0.5
    expected_cost: float = 0.0
    graph_growth_risk: float = 0.0
    failure_cooldown: float = 0.0


@dataclass(frozen=True)
class ComputationPacket:
    allocation_id: str
    operation: GraphOperation
    target_region: tuple[str, ...]
    predicted_evc: float
    raw: EVCSignals
    normalized: EVCSignals
    requested_budget: dict[str, int]
    remaining_global_budget: dict[str, int]
    allocator_mode: str = "adaptive_evc"
    operation_family: str = ""
    region_key: str = ""
    pre_state_summary: dict[str, float] = field(default_factory=dict)
    feedback_prior: dict[str, float] = field(default_factory=dict)
    fidelity_level: str = "medium"
    fidelity_fraction: float = 0.65

    def trace(self) -> dict:
        return {
            "allocation_id": self.allocation_id,
            "operation_id": self.operation.operation_id,
            "operation_family": self.operation_family,
            "region_key": self.region_key,
            "target_region": list(self.target_region),
            "predicted_evc": self.predicted_evc,
            "evc_components_raw": self.raw.__dict__,
            "evc_components_normalized": self.normalized.__dict__,
            "requested_budget": dict(self.requested_budget),
            "remaining_global_budget": dict(self.remaining_global_budget),
            "allocator_mode": self.allocator_mode,
            "pre_state_summary": dict(self.pre_state_summary),
            "feedback_prior": dict(self.feedback_prior),
            "fidelity_level": self.fidelity_level,
            "fidelity_fraction": self.fidelity_fraction,
        }


@dataclass
class AdaptiveComputationAllocator:
    config: DynamicV2ResearchConfig
    last_packets: list[ComputationPacket] = field(default_factory=list)
    allocation_serial: int = 0

    def allocate(
        self,
        graph: DynamicReasoningHypergraphV2,
        operations: list[GraphOperation],
        budget: Budget,
    ) -> list[ComputationPacket]:
        if not operations:
            self.last_packets = []
            return []
        mode = self.config.allocator_mode
        expanded: list[tuple[int, GraphOperation, str, float, EVCSignals]] = []
        for index, operation in enumerate(operations):
            base = self._signals(graph, operation)
            levels = self._fidelity_options(operation)
            for name, fraction in levels:
                expanded.append((
                    index, operation, name, fraction,
                    self._fidelity_signals(base, fraction),
                ))
        raw_rows = [row[4] for row in expanded]
        names = tuple(EVCSignals.__dataclass_fields__)
        columns = {
            name: _minmax([float(getattr(row, name)) for row in raw_rows]) for name in names
        }
        remaining = {
            "llm_calls": max(0, budget.max_llm_calls - budget.usage.llm_calls),
            "tokens": max(0, budget.max_total_tokens - budget.usage.total_tokens),
            "retrieval_calls": max(0, graph.limits.max_retrieval_calls - graph.retrieval_calls),
            "graph_operations": max(0, graph.limits.max_graph_operations - len(graph.operation_history)),
        }
        packets: list[tuple[int, ComputationPacket]] = []
        serial_start = self.allocation_serial
        for packet_index, (index, operation, fidelity_name, fidelity_fraction, raw_row) in enumerate(expanded):
            normalized = EVCSignals(**{
                name: columns[name][packet_index] for name in names
            })
            if mode == "adaptive_evc":
                evc = self._adaptive_evc(normalized)
            elif mode == "uniform":
                evc = 0.5
            else:
                evc = max(0.1, 1.0 - 0.1 * _fixed_priority(operation.operation_type))
            region = tuple(dict.fromkeys([operation.target_id, *operation.source_ids]))
            family = operation_family(operation)
            rkey = operation_region_key(operation)
            prior = feedback_prior(graph, family, rkey)
            packets.append((index, ComputationPacket(
                allocation_id=f"allocation_{serial_start + packet_index + 1:06d}",
                operation=operation,
                target_region=region,
                predicted_evc=evc,
                raw=raw_row,
                normalized=normalized,
                requested_budget=self._budget_packet(
                    operation, raw_row.graph_heat, remaining, prior,
                    fidelity_fraction=fidelity_fraction,
                ),
                remaining_global_budget=dict(remaining),
                allocator_mode=mode,
                operation_family=family,
                region_key=rkey,
                pre_state_summary=summarize_operation_region(graph, operation),
                feedback_prior=prior,
                fidelity_level=fidelity_name,
                fidelity_fraction=fidelity_fraction,
            )))
        self.allocation_serial += len(expanded)
        if mode == "adaptive_evc":
            packets.sort(key=lambda row: (
                -row[1].predicted_evc,
                row[1].operation.operation_type.value,
                row[1].operation.operation_id,
            ))
        elif mode == "fixed_order":
            packets.sort(key=lambda row: (
                _fixed_priority(row[1].operation.operation_type),
                row[1].operation.operation_id,
            ))
        else:
            packets.sort(key=lambda row: row[0])
        self.last_packets = [packet for _, packet in packets]
        return self.last_packets

    def _fidelity_options(self, operation: GraphOperation) -> list[tuple[str, float]]:
        mode = str(operation.payload.get("mode", ""))
        if operation.operation_type == OperationType.BRANCH and mode == "extract_typed":
            # The smoke trace showed that low/medium JSON caps truncate atomic
            # claim coverage. A full schema is the minimum viable extraction;
            # adaptation occurs in passage/span activation and candidate count.
            return [("high", 1.0)]
        if operation.operation_type == OperationType.MERGE and mode == "validate_join":
            # JOIN validation has a compact but irreducible audit schema.
            return [("high", 1.0)]
        adjustable = operation.operation_type in {
            OperationType.RETRIEVE, OperationType.BRANCH, OperationType.VERIFY,
            OperationType.MERGE, OperationType.EXPAND,
        }
        if self.config.allocator_mode != "adaptive_evc" or not adjustable:
            return [("medium", self.config.allocation_mid_token_fraction)]
        rows = [
            ("low", self.config.allocation_min_token_fraction),
            ("medium", self.config.allocation_mid_token_fraction),
            ("high", 1.0),
        ]
        return rows[: self.config.allocation_fidelity_levels]

    @staticmethod
    def _fidelity_signals(base: EVCSignals, fraction: float) -> EVCSignals:
        # Score *marginal* progress rather than total progress. Diminishing gains
        # make a cheap pass preferable until graph heat/impact justifies deeper
        # compute; failed cheap passes can then expose a new high-value action.
        fraction = max(0.1, min(1.0, float(fraction)))
        gain = fraction ** 0.5
        efficiency = gain / fraction
        return EVCSignals(
            graph_heat=base.graph_heat * fraction,
            uncertainty_reduction=base.uncertainty_reduction * efficiency,
            answer_impact=base.answer_impact * gain,
            dependency_unlock=base.dependency_unlock * efficiency,
            evidence_novelty=base.evidence_novelty * efficiency,
            recovery_value=base.recovery_value * efficiency,
            observed_value=base.observed_value * efficiency,
            expected_cost=base.expected_cost * fraction,
            graph_growth_risk=base.graph_growth_risk * fraction,
            failure_cooldown=base.failure_cooldown,
        )

    def _adaptive_evc(self, normalized: EVCSignals) -> float:
        return max(0.0, (
            self.config.evc_weight_heat * normalized.graph_heat
            + self.config.evc_weight_uncertainty_reduction * normalized.uncertainty_reduction
            + self.config.evc_weight_answer_impact * normalized.answer_impact
            + self.config.evc_weight_unlock * normalized.dependency_unlock
            + self.config.evc_weight_novelty * normalized.evidence_novelty
            + self.config.evc_weight_recovery * normalized.recovery_value
            + self.config.evc_weight_observed_value * normalized.observed_value
            - self.config.evc_weight_cost * normalized.expected_cost
            - self.config.evc_weight_growth_risk * normalized.graph_growth_risk
            - self.config.evc_weight_failure_cooldown * normalized.failure_cooldown
        ))

    @staticmethod
    def attach(operation: GraphOperation, packet: ComputationPacket) -> GraphOperation:
        value = deepcopy(operation)
        value.payload = dict(value.payload)
        value.payload["_allocation"] = packet.trace()
        return value

    def _signals(
        self, graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
    ) -> EVCSignals:
        assignment_branch = (
            operation.operation_type == OperationType.BRANCH
            and str(operation.payload.get("mode", "")) == "assignments"
        )
        # Diffusion has already aggregated neighborhood state into the target
        # and explicit sources. Using every alternative in the subgoal here lets
        # one hot distractor dominate all candidate operations and repeatedly
        # starve downstream regions.
        node_ids = list(dict.fromkeys([operation.target_id, *operation.source_ids]))
        states = [graph.belief_states[node_id] for node_id in node_ids if node_id in graph.belief_states]
        heat = max((state.computation_heat for state in states), default=0.0)
        uncertainty = max((state.uncertainty for state in states), default=1.0)
        impact = max((state.downstream_answer_impact for state in states), default=0.0)
        unlock = max((state.dependency_unlock_value for state in states), default=0.0)
        novelty = (
            0.25 if assignment_branch
            else 1.0 if operation.operation_type == OperationType.RETRIEVE
            else 0.4
        )
        recovery = 1.0 if operation.operation_type == OperationType.REVISE else 0.0
        base_cost = 0.05 if assignment_branch else {
            OperationType.COMMIT: 0.02,
            OperationType.PRUNE: 0.02,
            OperationType.MERGE: 0.55,
            OperationType.REVISE: 0.10,
            OperationType.RETRIEVE: 0.35,
            OperationType.VERIFY: 0.70,
            OperationType.BRANCH: 0.75,
            OperationType.EXPAND: 1.00,
        }[operation.operation_type]
        growth = min(
            1.0,
            len(operation.payload.get("candidate_ids", []))
            / max(1, graph.limits.max_active_branches),
        ) if assignment_branch else {
            OperationType.EXPAND: 1.0,
            OperationType.BRANCH: 0.8,
            OperationType.RETRIEVE: 0.4,
            OperationType.MERGE: 0.35,
        }.get(operation.operation_type, 0.05)
        family = operation_family(operation)
        rkey = operation_region_key(operation)
        prior = feedback_prior(graph, family, rkey)
        empirical_cost = float(prior.get("mean_cost", 0.0))
        return EVCSignals(
            graph_heat=heat,
            uncertainty_reduction=uncertainty * _operation_reduction(operation.operation_type),
            answer_impact=impact,
            dependency_unlock=unlock,
            evidence_novelty=novelty * max(0.2, float(prior["posterior_success"])),
            recovery_value=recovery,
            observed_value=float(prior["posterior_value"]),
            expected_cost=max(base_cost, empirical_cost),
            graph_growth_risk=growth,
            failure_cooldown=float(prior["cooldown_active"]),
        )

    def _budget_packet(
        self,
        operation: GraphOperation,
        heat: float,
        remaining: dict[str, int],
        prior: dict[str, float],
        fidelity_fraction: float | None = None,
    ) -> dict[str, int]:
        assignment_branch = (
            operation.operation_type == OperationType.BRANCH
            and str(operation.payload.get("mode", "")) == "assignments"
        )
        max_tokens = 0 if assignment_branch else {
            OperationType.EXPAND: self.config.graph_editor_max_tokens,
            OperationType.BRANCH: self.config.typed_extraction_max_tokens,
            OperationType.VERIFY: self.config.soft_verifier_max_tokens,
            OperationType.MERGE: self.config.join_validation_max_tokens,
            OperationType.COMMIT: 0,
            OperationType.REVISE: 0,
            OperationType.RETRIEVE: 0,
            OperationType.PRUNE: 0,
        }[operation.operation_type]
        if fidelity_fraction is not None:
            fraction = max(0.1, min(1.0, float(fidelity_fraction)))
            verifications = max(1, int(round(
                self.config.max_independent_verifications * fraction
            )))
            packet_heat = fraction
        elif self.config.allocator_mode != "adaptive_evc":
            fraction = self.config.allocation_mid_token_fraction
            verifications = 1
            packet_heat = 0.5
        else:
            # With no outcomes, preserve the pre-feedback allocation exactly.
            # Once evidence exists, use a conservative posterior adjustment; a
            # neutral prior must never dilute a genuinely hot graph region.
            if float(prior.get("observations", 0.0)) <= 0.0:
                packet_heat = max(0.0, min(1.0, heat))
            else:
                posterior = float(prior["posterior_value"])
                packet_heat = max(0.0, min(1.0, heat + 0.25 * (posterior - 0.5)))
            if float(prior["cooldown_active"]) >= 1.0:
                packet_heat = min(packet_heat, self.config.allocation_min_token_fraction)
            if packet_heat >= self.config.allocation_high_heat_threshold:
                fraction = 1.0
                verifications = self.config.max_independent_verifications
            elif packet_heat >= self.config.allocation_mid_heat_threshold:
                fraction = self.config.allocation_mid_token_fraction
                verifications = 1
            else:
                fraction = self.config.allocation_min_token_fraction
                verifications = 1
        candidate_cap = max(
            1,
            int(round(self.config.max_extracted_claims_per_round * max(fraction, 0.25))),
        )
        requested_tokens = int(max_tokens * fraction) if max_tokens else 0
        schema_floor = {
            OperationType.BRANCH: 260 + 90 * candidate_cap,
            OperationType.VERIFY: 260 + 90 * max(1, len(operation.source_ids)),
            OperationType.MERGE: 400,
            OperationType.EXPAND: 500,
        }.get(operation.operation_type, 0) if not assignment_branch else 0
        requested_tokens = min(max_tokens, max(requested_tokens, schema_floor)) if max_tokens else 0
        tokens = min(remaining["tokens"], requested_tokens)
        top_k = min(
            self.config.max_adaptive_top_k,
            max(1, int(round(self.config.top_k * (0.5 + 0.5 * packet_heat)))),
        )
        branch_width = max(
            1, int(round(self.config.max_active_branches * max(fraction, 0.34))),
        )
        if assignment_branch:
            branch_width = min(
                len(operation.payload.get("candidate_ids", [])),
                max(2, branch_width),
            )
        return {
            "max_tokens": tokens,
            "retrieval_top_k": top_k,
            "candidate_cap": candidate_cap,
            "verification_samples": verifications,
            "branch_width": branch_width,
            "revision_allowance": 1 if (
                operation.operation_type == OperationType.REVISE
                or packet_heat >= self.config.allocation_mid_heat_threshold
            ) else 0,
        }


def operation_family(operation: GraphOperation) -> str:
    mode = str(operation.payload.get("mode", ""))
    return f"{operation.operation_type.value.lower()}:{mode or 'default'}"


def operation_region_key(operation: GraphOperation) -> str:
    contextual_fields = {
        key: operation.payload.get(key)
        for key in (
            "query", "event", "extraction_evidence_count", "join_signature",
            "candidate_id", "candidate_ids", "dependency_claim_ids", "premise_ids",
        )
        if key in operation.payload
    }
    return stable_hash({
        "target": operation.target_id,
        "branch": operation.branch_id,
        "family": operation_family(operation),
        "sources": sorted(str(value) for value in operation.source_ids),
        "context": contextual_fields,
    })[:16]


def feedback_key(family: str, region_key: str = "*") -> str:
    return f"{family}|{region_key}"


def feedback_prior(
    graph: DynamicReasoningHypergraphV2, family: str, region_key: str,
) -> dict[str, float]:
    exact = graph.operation_feedback.get(feedback_key(family, region_key))
    # Different subgoals can have radically different evidence and topology.
    # Family-wide statistics remain serialized for audit, but only causal
    # outcomes from the same target region control later allocation.
    stats = exact or OperationFeedbackStats()
    mean_cost = stats.cumulative_cost / max(1, stats.observations)
    return {
        "observations": float(stats.observations),
        "posterior_value": float(stats.posterior_value),
        "posterior_success": float(stats.posterior_success),
        "consecutive_failures": float(stats.consecutive_failures),
        "cooldown_until_step": float(stats.cooldown_until_step),
        "cooldown_active": float(graph.step < stats.cooldown_until_step),
        "mean_cost": max(0.0, min(1.0, mean_cost)),
    }


def operation_region_node_ids(
    graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
) -> list[str]:
    node_ids = list(dict.fromkeys([operation.target_id, *operation.source_ids]))
    for node in graph.nodes.values():
        if isinstance(node, (ClaimNode, EvidenceNode)):
            if node.target_subgoal == operation.target_id and node.branch_id == operation.branch_id:
                node_ids.append(node.node_id)
        elif isinstance(node, SubgoalNode) and node.node_id == operation.target_id:
            node_ids.append(node.node_id)
    return list(dict.fromkeys(node_ids))


def summarize_operation_region(
    graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
) -> dict[str, float]:
    node_ids = operation_region_node_ids(graph, operation)
    states = [graph.belief_states[node_id] for node_id in node_ids if node_id in graph.belief_states]
    claims = [graph.nodes[node_id] for node_id in node_ids if isinstance(graph.nodes.get(node_id), ClaimNode)]
    evidence = [graph.nodes[node_id] for node_id in node_ids if isinstance(graph.nodes.get(node_id), EvidenceNode)]
    active_claims = [
        node for node in claims
        if node.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
    ]
    joined = [
        node for node in claims
        if graph.claim_semantics.get(node.node_id)
        and graph.claim_semantics[node.node_id].join_depth > 0
    ]
    grounded_claims = [
        node for node in active_claims
        if node.evidence_refs and node.provenance.metadata.get("source_spans")
    ]
    proof_leaves = {
        leaf_id
        for row in graph.join_attempt_history if row.accepted
        for leaf_id in (row.proof_leaf_ids or row.premise_ids)
    }
    accepted_answers = [row for row in graph.answers() if row.status == AnswerStatus.ACCEPTED]
    completed = [row for row in graph.subgoals() if row.status == SubgoalStatus.RESOLVED]
    total_subgoals = max(1, len(graph.subgoals()))

    def avg(name: str, default: float) -> float:
        return sum(float(getattr(row, name)) for row in states) / len(states) if states else default

    chain_progress = min(1.0, len(completed) / total_subgoals + 0.25 * len(accepted_answers))
    return {
        "node_count": float(len(node_ids)),
        "claim_count": float(len(claims)),
        "active_claim_count": float(len(active_claims)),
        "evidence_count": float(len(evidence)),
        "join_count": float(len(joined)),
        "grounded_claim_count": float(len(grounded_claims)),
        "proof_leaf_count": float(len(proof_leaves)),
        "accepted_answer_count": float(len(accepted_answers)),
        "completed_subgoal_count": float(len(completed)),
        "uncertainty": avg("uncertainty", 1.0),
        "absolute_support": avg("absolute_support", 0.0),
        "evidence_gap": avg("evidence_gap", 1.0),
        "entropy": avg("entropy", 1.0),
        "dependency_unlock": avg("dependency_unlock_value", 0.0),
        "contradiction_pressure": avg("contradiction_pressure", 0.0),
        "answer_chain_progress": chain_progress,
    }


def _fixed_priority(value: OperationType) -> int:
    return {
        OperationType.REVISE: 0,
        OperationType.COMMIT: 1,
        OperationType.VERIFY: 2,
        OperationType.MERGE: 3,
        OperationType.RETRIEVE: 4,
        OperationType.BRANCH: 5,
        OperationType.EXPAND: 6,
        OperationType.PRUNE: 7,
    }[value]


def _operation_reduction(value: OperationType) -> float:
    return {
        OperationType.VERIFY: 0.9,
        OperationType.RETRIEVE: 0.8,
        OperationType.MERGE: 0.8,
        OperationType.REVISE: 0.7,
        OperationType.BRANCH: 0.6,
        OperationType.EXPAND: 0.5,
        OperationType.COMMIT: 0.2,
        OperationType.PRUNE: 0.2,
    }[value]


def _minmax(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [max(0.0, min(1.0, values[0]))] if values else []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        # Preserve absolute signal when the ready set has no variance. Returning
        # 0.5 erased cooldowns and made identical graph regions look neutral.
        value = max(0.0, min(1.0, values[0]))
        return [value for _ in values]
    return [(value - low) / (high - low) for value in values]
