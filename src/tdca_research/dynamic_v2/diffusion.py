from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..dynamic.graph import ClaimNode, EvidenceNode, SubgoalNode
from .config import DynamicV2ResearchConfig
from .graph import DiffusionSnapshot, DynamicReasoningHypergraphV2


@dataclass(frozen=True)
class TypedMessage:
    source: str
    target: str
    channel: str
    weight: float
    edge_type: str


class TypedDirectionalDiffusion:
    """Deterministic, typed and directional multi-channel message passing."""

    def __init__(self, config: DynamicV2ResearchConfig) -> None:
        self.config = config

    def propagate(
        self,
        graph: DynamicReasoningHypergraphV2,
        seed_node_ids: list[str],
        diffusion_id: str,
    ) -> DiffusionSnapshot:
        messages = self._messages(graph)
        base = {
            node_id: {
                "support_influence": state.support_influence,
                "contradiction_pressure": state.contradiction_pressure,
                "downstream_answer_impact": state.downstream_answer_impact,
                "computation_heat": self._initial_heat(state),
            }
            for node_id, state in graph.belief_states.items()
        }
        values = {node_id: dict(channels) for node_id, channels in base.items()}
        max_delta = 0.0
        converged = False
        iterations = 0
        for iteration in range(1, self.config.diffusion_steps + 1):
            incoming: dict[tuple[str, str], list[float]] = defaultdict(list)
            for message in messages:
                source_value = values.get(message.source, {}).get(message.channel, 0.0)
                incoming[(message.target, message.channel)].append(source_value * message.weight)
            next_values: dict[str, dict[str, float]] = {}
            max_delta = 0.0
            for node_id, channels in values.items():
                next_channels = {}
                for channel, old_value in channels.items():
                    propagated = _mean(incoming.get((node_id, channel), []))
                    new_value = _unit(
                        self.config.diffusion_restart * base[node_id][channel]
                        + (1.0 - self.config.diffusion_restart)
                        * (
                            (1.0 - self.config.diffusion_decay) * old_value
                            + self.config.diffusion_decay * propagated
                        )
                    )
                    next_channels[channel] = new_value
                    max_delta = max(max_delta, abs(new_value - old_value))
                next_values[node_id] = next_channels
            values = next_values
            iterations = iteration
            if max_delta <= self.config.diffusion_min_delta:
                converged = True
                break
        for node_id, channels in values.items():
            state = graph.belief_states[node_id]
            state.support_influence = channels["support_influence"]
            state.contradiction_pressure = channels["contradiction_pressure"]
            state.downstream_answer_impact = channels["downstream_answer_impact"]
            state.computation_heat = channels["computation_heat"]
        snapshot = DiffusionSnapshot(
            diffusion_id=diffusion_id,
            step=graph.step,
            seed_node_ids=sorted(set(seed_node_ids)),
            iterations=iterations,
            converged=converged,
            max_delta=max_delta,
            channels_by_node=values,
            typed_messages=[message.__dict__ for message in messages],
        )
        graph.diffusion_history.append(snapshot)
        return snapshot

    def _initial_heat(self, state) -> float:
        numerator = (
            self.config.heat_weight_uncertainty * state.uncertainty
            + self.config.heat_weight_answer_impact * state.downstream_answer_impact
            + self.config.heat_weight_evidence_gap * state.evidence_gap
            + self.config.heat_weight_contradiction * state.contradiction_pressure
            + self.config.heat_weight_unlock * state.dependency_unlock_value
        )
        denominator = (
            self.config.heat_weight_uncertainty
            + self.config.heat_weight_answer_impact
            + self.config.heat_weight_evidence_gap
            + self.config.heat_weight_contradiction
            + self.config.heat_weight_unlock
        ) or 1.0
        return _unit(numerator / denominator) if state.valid else 0.0

    @staticmethod
    def _messages(graph: DynamicReasoningHypergraphV2) -> list[TypedMessage]:
        rows: list[TypedMessage] = []
        for claim in graph.claims():
            if not graph.belief_states.get(claim.node_id) or not graph.belief_states[claim.node_id].valid:
                continue
            for evidence_id in claim.evidence_refs:
                if (
                    evidence_id in graph.nodes
                    and isinstance(graph.nodes[evidence_id], EvidenceNode)
                    and graph.belief_states.get(evidence_id)
                    and graph.belief_states[evidence_id].valid
                ):
                    rows.append(TypedMessage(evidence_id, claim.node_id, "support_influence", 0.90, "evidence_support"))
                    rows.append(TypedMessage(claim.node_id, evidence_id, "downstream_answer_impact", 0.55, "evidence_demand"))
            rows.append(TypedMessage(claim.node_id, claim.target_subgoal, "support_influence", 0.80, "claim_resolves_subgoal"))
            rows.append(TypedMessage(claim.target_subgoal, claim.node_id, "downstream_answer_impact", 0.85, "subgoal_demands_claim"))
            rows.append(TypedMessage(claim.target_subgoal, claim.node_id, "computation_heat", 0.70, "subgoal_allocates_claim"))
            for other_id in claim.contradiction_links:
                if (
                    other_id in graph.nodes
                    and graph.belief_states.get(other_id)
                    and graph.belief_states[other_id].valid
                ):
                    rows.append(TypedMessage(other_id, claim.node_id, "contradiction_pressure", 0.90, "claim_contradiction"))
        for edge in graph.hyperedges.values():
            if edge.edge_id in graph.invalidated_hyperedges:
                continue
            members = [*edge.source_node_set, edge.target_node]
            if any(
                not graph.belief_states.get(node_id)
                or not graph.belief_states[node_id].valid
                for node_id in members
            ):
                continue
            for source_id in edge.source_node_set:
                rows.append(TypedMessage(source_id, edge.target_node, "support_influence", 0.80, "hyperedge_forward_support"))
                rows.append(TypedMessage(edge.target_node, source_id, "downstream_answer_impact", 0.85, "hyperedge_backward_impact"))
                rows.append(TypedMessage(edge.target_node, source_id, "computation_heat", 0.70, "hyperedge_backward_heat"))
        for subgoal in graph.subgoals():
            if not graph.belief_states.get(subgoal.node_id) or not graph.belief_states[subgoal.node_id].valid:
                continue
            for dependency in subgoal.dependencies:
                if (
                    dependency in graph.nodes
                    and isinstance(graph.nodes[dependency], SubgoalNode)
                    and graph.belief_states.get(dependency)
                    and graph.belief_states[dependency].valid
                ):
                    rows.append(TypedMessage(dependency, subgoal.node_id, "support_influence", 0.65, "execution_forward"))
                    rows.append(TypedMessage(subgoal.node_id, dependency, "downstream_answer_impact", 0.80, "execution_backward"))
                    rows.append(TypedMessage(subgoal.node_id, dependency, "computation_heat", 0.70, "execution_heat"))
        return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
