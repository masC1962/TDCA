from __future__ import annotations

from dataclasses import dataclass

from ..budget import Budget
from ..dynamic.graph import AnswerStatus, CandidateStatus, ClaimNode
from .allocator import ComputationPacket
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2, TerminationKind


@dataclass(frozen=True)
class MetaDecision:
    outcome: TerminationKind
    reason: str
    best_predicted_evc: float
    answer_node_id: str | None = None


class MetaStopPolicy:
    def __init__(self, config: DynamicV2ResearchConfig) -> None:
        self.config = config

    def decide(
        self,
        graph: DynamicReasoningHypergraphV2,
        packets: list[ComputationPacket],
        budget: Budget,
    ) -> MetaDecision:
        answer = self._supported_answer(graph)
        if answer is not None:
            return MetaDecision(
                TerminationKind.ANSWER, "accepted_graph_grounded_answer",
                max((row.predicted_evc for row in packets), default=0.0), answer.node_id,
            )
        if not packets:
            return MetaDecision(TerminationKind.ABSTAIN, "no_executable_computation", 0.0)
        best = packets[0]
        if not self._affordable(best, graph, budget):
            return MetaDecision(
                TerminationKind.BUDGET_EXHAUSTED,
                "positive_value_computation_exceeds_remaining_budget",
                best.predicted_evc,
            )
        if best.predicted_evc <= self.config.meta_stop_evc_threshold:
            return MetaDecision(
                TerminationKind.ABSTAIN,
                "best_expected_value_not_above_marginal_cost_threshold",
                best.predicted_evc,
            )
        return MetaDecision(TerminationKind.CONTINUE, "positive_expected_value", best.predicted_evc)

    @staticmethod
    def _supported_answer(graph: DynamicReasoningHypergraphV2):
        valid = []
        for answer in graph.answers():
            if answer.status != AnswerStatus.ACCEPTED or answer.derivation_edge in graph.invalidated_hyperedges:
                continue
            claims = [graph.node(node_id, ClaimNode) for node_id in answer.supporting_claims]
            if any(claim.status in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED} for claim in claims):
                continue
            if not answer.supporting_evidence:
                continue
            valid.append(answer)
        return max(valid, key=lambda value: (value.confidence, value.node_id), default=None)

    @staticmethod
    def _affordable(
        packet: ComputationPacket,
        graph: DynamicReasoningHypergraphV2,
        budget: Budget,
    ) -> bool:
        request = packet.requested_budget
        if request.get("max_tokens", 0) > 0 and not budget.can_call(request["max_tokens"]):
            return False
        if packet.operation.operation_type.value == "RETRIEVE" and graph.retrieval_calls >= graph.limits.max_retrieval_calls:
            return False
        return len(graph.operation_history) < graph.limits.max_graph_operations
