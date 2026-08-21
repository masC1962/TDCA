from __future__ import annotations

from collections import deque

from ..dynamic.graph import (
    AnswerNode,
    AnswerStatus,
    CandidateStatus,
    ClaimNode,
    EvidenceNode,
    SubgoalNode,
)
from .graph import BeliefState, DynamicReasoningHypergraphV2


class GraphBeliefUpdater:
    """Deterministic local belief update after a controller transaction."""

    def recompute(
        self,
        graph: DynamicReasoningHypergraphV2,
        seed_node_ids: list[str] | None,
        reason: str,
    ) -> list[str]:
        affected = self._affected_closure(graph, seed_node_ids)
        impacts = _downstream_impacts(graph)
        changed: list[str] = []
        for node_id in sorted(affected):
            node = graph.nodes[node_id]
            previous = graph.belief_states.get(node_id, BeliefState())
            current = self._local_state(graph, node_id, impacts.get(node_id, 0.0))
            current.version = previous.version + int(_belief_payload(previous) != _belief_payload(current))
            current.updated_at_step = graph.step
            current.update_reason = reason
            graph.belief_states[node_id] = current
            if _belief_payload(previous) != _belief_payload(current):
                changed.append(node_id)
        return changed

    def _local_state(
        self, graph: DynamicReasoningHypergraphV2, node_id: str, answer_impact: float,
    ) -> BeliefState:
        node = graph.nodes[node_id]
        if isinstance(node, EvidenceNode):
            rank_support = 1.0 / max(1, node.retrieval_rank)
            return BeliefState(
                absolute_support=_unit(rank_support),
                relative_weight=0.0,
                entropy=0.0,
                evidence_gap=0.0,
                support_influence=_unit(rank_support),
                contradiction_pressure=0.0,
                downstream_answer_impact=answer_impact,
                dependency_unlock_value=0.0,
                uncertainty=0.0,
                valid=True,
            )
        if isinstance(node, ClaimNode):
            valid = node.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
            linked = [
                graph.nodes[value] for value in node.contradiction_links
                if value in graph.nodes and isinstance(graph.nodes[value], ClaimNode)
            ]
            linked_pressure = max(
                (value.score.absolute_support for value in linked if value.status != CandidateStatus.INVALID),
                default=0.0,
            )
            contradiction = max(node.score.raw.contradiction_risk, linked_pressure)
            unlock = _claim_unlock_value(graph, node.node_id)
            return BeliefState(
                absolute_support=node.score.absolute_support,
                relative_weight=node.score.relative_weight,
                entropy=node.score.set_entropy,
                evidence_gap=node.score.evidence_gap,
                support_influence=node.score.absolute_support if valid else 0.0,
                contradiction_pressure=_unit(contradiction),
                downstream_answer_impact=answer_impact,
                dependency_unlock_value=unlock,
                uncertainty=_unit(max(node.score.set_entropy, node.score.evidence_gap)),
                valid=valid,
            )
        if isinstance(node, SubgoalNode):
            claims = [
                claim for claim in graph.claims(node.node_id)
                if claim.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
            ]
            best = max(claims, key=lambda value: value.score.absolute_support, default=None)
            children = sum(node.node_id in child.dependencies for child in graph.subgoals())
            return BeliefState(
                absolute_support=best.score.absolute_support if best else node.confidence,
                relative_weight=best.score.relative_weight if best else 0.0,
                entropy=max((claim.score.set_entropy for claim in claims), default=node.uncertainty),
                evidence_gap=best.score.evidence_gap if best else 1.0,
                support_influence=best.score.absolute_support if best else 0.0,
                contradiction_pressure=max(
                    (claim.score.raw.contradiction_risk for claim in claims), default=0.0,
                ),
                downstream_answer_impact=answer_impact,
                dependency_unlock_value=_unit(children / 3.0),
                uncertainty=node.uncertainty,
                valid=node.status.value != "archived",
            )
        assert isinstance(node, AnswerNode)
        valid = node.status == AnswerStatus.ACCEPTED
        return BeliefState(
            absolute_support=node.confidence,
            relative_weight=1.0 if valid else 0.0,
            entropy=_unit(1.0 - node.confidence),
            evidence_gap=0.0 if node.supporting_evidence else 1.0,
            support_influence=node.confidence if valid else 0.0,
            contradiction_pressure=node.contradiction_risk,
            downstream_answer_impact=1.0,
            dependency_unlock_value=0.0,
            uncertainty=_unit(1.0 - node.confidence),
            valid=valid,
        )

    @staticmethod
    def _affected_closure(
        graph: DynamicReasoningHypergraphV2, seed_node_ids: list[str] | None,
    ) -> set[str]:
        if not seed_node_ids:
            return set(graph.nodes)
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in graph.nodes}
        for edge in graph.hyperedges.values():
            members = edge.source_node_set + [edge.target_node]
            for left in members:
                adjacency.setdefault(left, set()).update(value for value in members if value != left)
        for claim in graph.claims():
            adjacency[claim.node_id].add(claim.target_subgoal)
            adjacency.setdefault(claim.target_subgoal, set()).add(claim.node_id)
            for related in claim.evidence_refs + claim.dependency_claim_ids + claim.contradiction_links:
                if related in graph.nodes:
                    adjacency[claim.node_id].add(related)
                    adjacency.setdefault(related, set()).add(claim.node_id)
        for subgoal in graph.subgoals():
            for dependency in subgoal.dependencies:
                if dependency in graph.nodes:
                    adjacency[subgoal.node_id].add(dependency)
                    adjacency.setdefault(dependency, set()).add(subgoal.node_id)
        queue = deque(value for value in seed_node_ids if value in graph.nodes)
        affected = set(queue)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in affected:
                    affected.add(neighbor)
                    queue.append(neighbor)
        return affected


def _downstream_impacts(graph: DynamicReasoningHypergraphV2) -> dict[str, float]:
    roots = [node.node_id for node in graph.subgoals() if node.terminal or node.node_id == "subgoal_root"]
    roots += [node.node_id for node in graph.answers()]
    reverse: dict[str, set[str]] = {node_id: set() for node_id in graph.nodes}
    for edge in graph.hyperedges.values():
        if edge.edge_id in graph.invalidated_hyperedges:
            continue
        for source in edge.source_node_set:
            reverse.setdefault(edge.target_node, set()).add(source)
    for subgoal in graph.subgoals():
        for dependency in subgoal.dependencies:
            reverse.setdefault(subgoal.node_id, set()).add(dependency)
    for claim in graph.claims():
        reverse.setdefault(claim.target_subgoal, set()).add(claim.node_id)
        for evidence_id in claim.evidence_refs:
            reverse.setdefault(claim.node_id, set()).add(evidence_id)
    impact = {node_id: 0.0 for node_id in graph.nodes}
    queue = deque((root, 0) for root in roots if root in graph.nodes)
    best_distance: dict[str, int] = {}
    while queue:
        current, distance = queue.popleft()
        if distance >= best_distance.get(current, 10**9):
            continue
        best_distance[current] = distance
        impact[current] = max(impact[current], 1.0 / (distance + 1.0))
        for source in reverse.get(current, set()):
            queue.append((source, distance + 1))
    return impact


def _claim_unlock_value(graph: DynamicReasoningHypergraphV2, claim_id: str) -> float:
    claim = graph.node(claim_id, ClaimNode)
    children = sum(claim.target_subgoal in subgoal.dependencies for subgoal in graph.subgoals())
    joined = sum(claim_id in edge.source_node_set for edge in graph.hyperedges.values())
    return _unit((children + joined) / 3.0)


def _belief_payload(value: BeliefState) -> tuple:
    return (
        value.absolute_support, value.relative_weight, value.entropy, value.evidence_gap,
        value.support_influence, value.contradiction_pressure, value.downstream_answer_impact,
        value.dependency_unlock_value, value.uncertainty, value.valid,
    )


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
