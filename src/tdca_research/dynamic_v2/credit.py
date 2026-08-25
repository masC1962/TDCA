from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..dynamic.graph import (
    AnswerNode,
    AnswerStatus,
    CandidateStatus,
    ClaimNode,
    EvidenceNode,
    OperationType,
    SubgoalNode,
)
from .config import DynamicV2ResearchConfig
from .graph import (
    AllocationRecord,
    CreditAssignmentRecord,
    DynamicReasoningHypergraphV2,
)


@dataclass(frozen=True)
class DelayedCreditSnapshot:
    seed_node_ids: tuple[str, ...]
    causal_descendant_ids: tuple[str, ...]
    causal_distance_by_node: dict[str, int]
    delayed_components_raw: dict[str, float]
    delayed_components_normalized: dict[str, float]
    delayed_realized_proof_return: float
    causal_event_ids: tuple[str, ...]


def refresh_delayed_credit(
    graph: DynamicReasoningHypergraphV2,
    config: DynamicV2ResearchConfig,
    *,
    terminal: bool = False,
) -> None:
    """Append changed causal-credit observations and refresh ledger caches.

    This function mutates graph-owned accounting state and must therefore only
    be called by ``V2GraphController`` while the controller seal is open.
    Attribution is structural and gold-free: temporal order alone never creates
    an eligibility edge.
    """

    if not config.delayed_credit_assignment:
        return
    outcomes = {row.allocation_id: row for row in graph.operation_outcome_history}
    for allocation in graph.allocation_history:
        snapshot = delayed_credit_snapshot(graph, allocation, config)
        previous = _latest_credit(graph, allocation.allocation_id)
        changed = previous is None or _credit_signature(previous) != _snapshot_signature(snapshot)
        needs_terminal_marker = terminal and (previous is None or not previous.terminal)
        if changed or needs_terminal_marker:
            serial = 1 + sum(
                row.allocation_id == allocation.allocation_id
                for row in graph.credit_assignment_history
            )
            graph.credit_assignment_history.append(CreditAssignmentRecord(
                credit_id=f"credit_{allocation.allocation_id}_{serial:04d}",
                allocation_id=allocation.allocation_id,
                operation_id=allocation.operation_id,
                source_step=allocation.step,
                observed_at_step=graph.step,
                gamma=float(config.delayed_credit_gamma),
                seed_node_ids=list(snapshot.seed_node_ids),
                causal_descendant_ids=list(snapshot.causal_descendant_ids),
                causal_distance_by_node=dict(snapshot.causal_distance_by_node),
                delayed_components_raw=dict(snapshot.delayed_components_raw),
                delayed_components_normalized=dict(snapshot.delayed_components_normalized),
                delayed_realized_proof_return=snapshot.delayed_realized_proof_return,
                causal_event_ids=list(snapshot.causal_event_ids),
                terminal=bool(terminal),
            ))
        allocation.delayed_realized_proof_return = snapshot.delayed_realized_proof_return
        allocation.combined_realized_utility = _combined_utility(
            allocation.actual_immediate_utility,
            snapshot.delayed_realized_proof_return,
            allocation.actual_normalized_cost,
            config,
        )
        allocation.credit_finalized = bool(terminal)
        outcome = outcomes.get(allocation.allocation_id)
        if outcome is not None:
            outcome.delayed_realized_proof_return = snapshot.delayed_realized_proof_return
            outcome.combined_realized_utility = allocation.combined_realized_utility


def delayed_credit_snapshot(
    graph: DynamicReasoningHypergraphV2,
    allocation: AllocationRecord,
    config: DynamicV2ResearchConfig,
) -> DelayedCreditSnapshot:
    seeds = _credit_seeds(graph, allocation)
    distances = _causal_distances(graph, seeds)
    descendants = tuple(sorted(distances))
    gamma = float(config.delayed_credit_gamma)
    source_step = int(allocation.step)
    active_claims = [
        node for node_id, node in graph.nodes.items()
        if node_id in distances
        and isinstance(node, ClaimNode)
        and node.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
    ]
    future_claims = [node for node in active_claims if node.created_at_step > source_step]
    answer_claims = [
        node for node in future_claims
        if bool(node.provenance.metadata.get("answers_subgoal", False))
    ]
    candidate_availability = max((
        _discount(node.score.absolute_support, distances[node.node_id], gamma)
        for node in answer_claims
    ), default=0.0)

    descendant_evidence = {
        node_id for node_id, node in graph.nodes.items()
        if node_id in distances and isinstance(node, EvidenceNode)
    }
    accepted_evidence_scores = []
    accepted_evidence_ids: set[str] = set()
    for claim in future_claims:
        if claim.status == CandidateStatus.PROPOSED:
            continue
        used = descendant_evidence.intersection(claim.evidence_refs)
        if not used:
            continue
        accepted_evidence_ids.update(used)
        accepted_evidence_scores.append(_discount(
            len(used) / max(1, len(descendant_evidence)),
            distances[claim.node_id], gamma,
        ))
    accepted_evidence = max(accepted_evidence_scores, default=0.0)

    joined_claims = [
        node for node in future_claims
        if graph.claim_semantics.get(node.node_id) is not None
        and graph.claim_semantics[node.node_id].join_depth > 0
    ]
    successful_join = max((
        _discount(1.0, distances[node.node_id], gamma) for node in joined_claims
    ), default=0.0)

    accepted_answers = [
        node for node_id, node in graph.nodes.items()
        if node_id in distances
        and isinstance(node, AnswerNode)
        and node.created_at_step > source_step
        and node.status == AnswerStatus.ACCEPTED
    ]
    supported_terminal_answer = max((
        _discount(1.0, distances[node.node_id], gamma) for node in accepted_answers
    ), default=0.0)
    proof_completeness = max((
        _discount(
            graph.terminal_beliefs[node.node_id].chain_coverage,
            distances[node.node_id], gamma,
        )
        for node in accepted_answers if node.node_id in graph.terminal_beliefs
    ), default=0.0)

    raw = {
        "proof_completeness": _unit(proof_completeness),
        "candidate_availability": _unit(candidate_availability),
        "accepted_evidence": _unit(accepted_evidence),
        "successful_join": _unit(successful_join),
        "supported_terminal_answer": _unit(supported_terminal_answer),
    }
    # Components already have an absolute [0,1] interpretation.  Preserve that
    # interpretation instead of min-max scaling within a question.
    normalized = dict(raw)
    weights = {
        "proof_completeness": config.delayed_credit_weight_proof_completeness,
        "candidate_availability": config.delayed_credit_weight_candidate_availability,
        "accepted_evidence": config.delayed_credit_weight_accepted_evidence,
        "successful_join": config.delayed_credit_weight_successful_join,
        "supported_terminal_answer": config.delayed_credit_weight_supported_terminal_answer,
    }
    denominator = max(1e-12, sum(float(value) for value in weights.values()))
    realized = _unit(sum(
        float(weights[name]) * normalized[name] for name in weights
    ) / denominator)
    events = {
        node.node_id for node in answer_claims + joined_claims + accepted_answers
    } | accepted_evidence_ids
    return DelayedCreditSnapshot(
        seed_node_ids=tuple(sorted(seeds)),
        causal_descendant_ids=descendants,
        causal_distance_by_node={key: distances[key] for key in descendants},
        delayed_components_raw=raw,
        delayed_components_normalized=normalized,
        delayed_realized_proof_return=realized,
        causal_event_ids=tuple(sorted(events)),
    )


def _credit_seeds(
    graph: DynamicReasoningHypergraphV2, allocation: AllocationRecord,
) -> set[str]:
    audit = next((
        row for row in graph.operation_history if row.operation_id == allocation.operation_id
    ), None)
    if audit is None:
        return set()
    created = {node_id for node_id in audit.created_nodes if node_id in graph.nodes}
    updated = {node_id for node_id in audit.updated_nodes if node_id in graph.nodes}
    pruned = {node_id for node_id in audit.pruned_nodes if node_id in graph.nodes}
    operation_type = audit.operation_type
    if operation_type in {
        OperationType.RETRIEVE, OperationType.EXPAND, OperationType.MERGE,
    }:
        return created
    if operation_type == OperationType.BRANCH:
        return created or {
            node_id for node_id in allocation.target_region
            if isinstance(graph.nodes.get(node_id), ClaimNode)
        }
    if operation_type == OperationType.VERIFY:
        return {node_id for node_id in updated if isinstance(graph.nodes[node_id], ClaimNode)}
    if operation_type == OperationType.COMMIT:
        answer_nodes = {
            node_id for node_id in created if isinstance(graph.nodes[node_id], AnswerNode)
        }
        if answer_nodes:
            return answer_nodes
        return {
            node_id for node_id in updated | set(allocation.target_region)
            if isinstance(graph.nodes.get(node_id), ClaimNode)
        }
    if operation_type in {OperationType.REVISE, OperationType.PRUNE}:
        return {
            node_id for node_id in updated | pruned
            if isinstance(graph.nodes.get(node_id), (ClaimNode, AnswerNode))
        }
    return created | updated


def _causal_distances(
    graph: DynamicReasoningHypergraphV2, seeds: set[str],
) -> dict[str, int]:
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in graph.nodes}
    for node_id, node in graph.nodes.items():
        sources = set(node.provenance.source_node_ids) | set(node.provenance.evidence_ids)
        if isinstance(node, ClaimNode):
            sources.update(node.evidence_refs)
            sources.update(node.dependency_claim_ids)
        elif isinstance(node, AnswerNode):
            sources.update(node.supporting_claims)
            sources.update(node.supporting_evidence)
        # Subgoal execution edges describe plan order, not evidence causality.
        if isinstance(node, SubgoalNode):
            sources = set(node.provenance.source_node_ids)
        for source in sources:
            if source in adjacency and source != node_id:
                adjacency[source].add(node_id)
    for edge in graph.hyperedges.values():
        for source in edge.source_node_set:
            if source in adjacency and edge.target_node in graph.nodes:
                adjacency[source].add(edge.target_node)
    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in sorted(seeds))
    while queue:
        node_id, distance = queue.popleft()
        if node_id in distances and distances[node_id] <= distance:
            continue
        distances[node_id] = distance
        for target in sorted(adjacency.get(node_id, ())):
            queue.append((target, distance + 1))
    return distances


def _combined_utility(
    immediate: float, delayed: float, cost: float,
    config: DynamicV2ResearchConfig,
) -> float:
    value = (
        float(config.evc_immediate_horizon_weight) * _unit(immediate)
        + float(config.evc_delayed_horizon_weight) * _unit(delayed)
        - _unit(cost)
    )
    return max(-1.0, min(1.0, value))


def _latest_credit(
    graph: DynamicReasoningHypergraphV2, allocation_id: str,
) -> CreditAssignmentRecord | None:
    return next((
        row for row in reversed(graph.credit_assignment_history)
        if row.allocation_id == allocation_id
    ), None)


def _credit_signature(row: CreditAssignmentRecord) -> tuple:
    return (
        tuple(row.seed_node_ids), tuple(row.causal_descendant_ids),
        tuple(sorted(row.causal_distance_by_node.items())),
        tuple(sorted(row.delayed_components_raw.items())),
        float(row.delayed_realized_proof_return), tuple(row.causal_event_ids),
    )


def _snapshot_signature(row: DelayedCreditSnapshot) -> tuple:
    return (
        row.seed_node_ids, row.causal_descendant_ids,
        tuple(sorted(row.causal_distance_by_node.items())),
        tuple(sorted(row.delayed_components_raw.items())),
        float(row.delayed_realized_proof_return), row.causal_event_ids,
    )


def _discount(value: float, distance: int, gamma: float) -> float:
    return _unit(float(value)) * float(gamma) ** max(0, int(distance))


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
