from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from ..budget import Budget
from ..dynamic.graph import GraphOperation, OperationType
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2


@dataclass(frozen=True)
class EVCSignals:
    graph_heat: float = 0.0
    uncertainty_reduction: float = 0.0
    answer_impact: float = 0.0
    dependency_unlock: float = 0.0
    evidence_novelty: float = 0.0
    recovery_value: float = 0.0
    expected_cost: float = 0.0
    graph_growth_risk: float = 0.0


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

    def trace(self) -> dict:
        return {
            "allocation_id": self.allocation_id,
            "operation_id": self.operation.operation_id,
            "target_region": list(self.target_region),
            "predicted_evc": self.predicted_evc,
            "evc_components_raw": self.raw.__dict__,
            "evc_components_normalized": self.normalized.__dict__,
            "requested_budget": dict(self.requested_budget),
            "remaining_global_budget": dict(self.remaining_global_budget),
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
        raw_rows = [self._signals(graph, operation) for operation in operations]
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
        packets = []
        serial_start = self.allocation_serial
        for index, operation in enumerate(operations):
            normalized = EVCSignals(**{name: columns[name][index] for name in names})
            evc = max(0.0, (
                self.config.evc_weight_heat * normalized.graph_heat
                + self.config.evc_weight_uncertainty_reduction * normalized.uncertainty_reduction
                + self.config.evc_weight_answer_impact * normalized.answer_impact
                + self.config.evc_weight_unlock * normalized.dependency_unlock
                + self.config.evc_weight_novelty * normalized.evidence_novelty
                + self.config.evc_weight_recovery * normalized.recovery_value
                - self.config.evc_weight_cost * normalized.expected_cost
                - self.config.evc_weight_growth_risk * normalized.graph_growth_risk
            ))
            region = tuple(dict.fromkeys([operation.target_id, *operation.source_ids]))
            packets.append(ComputationPacket(
                allocation_id=f"allocation_{serial_start + index + 1:06d}",
                operation=operation,
                target_region=region,
                predicted_evc=evc,
                raw=raw_rows[index],
                normalized=normalized,
                requested_budget=self._budget_packet(operation, raw_rows[index].graph_heat, remaining),
                remaining_global_budget=dict(remaining),
            ))
        self.allocation_serial += len(operations)
        self.last_packets = sorted(
            packets,
            key=lambda row: (-row.predicted_evc, row.operation.operation_type.value, row.operation.operation_id),
        )
        return self.last_packets

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
        region = list(dict.fromkeys([operation.target_id, *operation.source_ids]))
        states = [graph.belief_states[node_id] for node_id in region if node_id in graph.belief_states]
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
        return EVCSignals(
            graph_heat=heat,
            uncertainty_reduction=uncertainty * _operation_reduction(operation.operation_type),
            answer_impact=impact,
            dependency_unlock=unlock,
            evidence_novelty=novelty,
            recovery_value=recovery,
            expected_cost=base_cost,
            graph_growth_risk=growth,
        )

    def _budget_packet(
        self, operation: GraphOperation, heat: float, remaining: dict[str, int],
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
        if heat >= self.config.allocation_high_heat_threshold:
            fraction = 1.0
            verifications = self.config.max_independent_verifications
        elif heat >= self.config.allocation_mid_heat_threshold:
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
        # Adaptive allocation may reduce the number of objects requested from a
        # structured call, but it must not make the selected schema impossible to
        # serialize.  These deterministic floors scale with output cardinality;
        # they are protocol constraints rather than task- or relation-specific
        # accuracy patches.
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
            max(1, int(round(self.config.top_k * (0.5 + 0.5 * heat)))),
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
                or heat >= self.config.allocation_mid_heat_threshold
            ) else 0,
        }


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
        # Absolute information is retained for a singleton; unlike v1 ranking,
        # meta-stop must compare its EVC with marginal compute cost.
        return [max(0.0, min(1.0, values[0]))] if values else []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]
