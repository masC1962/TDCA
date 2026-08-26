from __future__ import annotations

import math
from typing import Any

from ..dynamic.graph import (
    AnswerStatus,
    BranchStatus,
    CandidateStatus,
    ClaimNode,
    EvidenceNode,
    GraphOperation,
    OperationType,
    SubgoalNode,
    SubgoalStatus,
)
from ..utils import normalize_text, stable_hash
from .graph import DynamicReasoningHypergraphV2
from .join import deterministic_join_derivation, join_candidate_from_operation
from .obligations import terminal_dependency_distance
from .proof import audit_graph_proof, claim_closure


_VIABLE_CLAIM_STATES = {
    CandidateStatus.SCORED,
    CandidateStatus.RETAINED,
    CandidateStatus.REOPENED,
    CandidateStatus.REVISED,
}


def certified_transition_value(
    graph: DynamicReasoningHypergraphV2,
    operation: GraphOperation,
    config: Any | None = None,
) -> dict[str, Any]:
    """Return a gold-free certificate for a guaranteed provider-free transition.

    This is deliberately narrower than operation-family membership.  The
    concrete payload and current graph state must prove that applying the
    operation changes the branch/execution state and exposes a successor.  A
    stale or hand-crafted certificate is therefore harmless: the stop policy
    recomputes this value from the sealed graph.
    """

    certificate: dict[str, Any] = {
        "certificate_version": "certified-transition-option-v2.4.3.2",
        "kind": "",
        "mandatory": False,
        "deterministic": False,
        "provider_calls": 0,
        "target_subgoal": operation.target_id,
        "branch_id": operation.branch_id,
        "source_claim_ids": [],
        "successor_subgoal_ids": [],
        "transition_certainty": 0.0,
        "successor_reachability_gain": 0.0,
        "successor_option_value": 0.0,
        "transition_redundancy": 1.0,
        "predicted_transition_value": 0.0,
        "preconditions": {},
        "promised_effect": {},
    }
    subgoal = graph.nodes.get(operation.target_id)
    branch = graph.branches.get(operation.branch_id)
    if not isinstance(subgoal, SubgoalNode) or branch is None:
        return _seal(certificate)
    if branch.status != BranchStatus.ACTIVE:
        return _seal(certificate)

    successor_ids = sorted(
        row.node_id for row in graph.subgoals()
        if operation.target_id in row.dependencies
        and row.status != SubgoalStatus.ARCHIVED
    )
    virtual_terminal = bool(
        subgoal.terminal or subgoal.node_id == "subgoal_root"
    )

    if (
        operation.operation_type == OperationType.COMMIT
        and str(operation.payload.get("mode", "")) == "answer"
        and bool(getattr(config, "certified_terminal_materialization", False))
    ):
        terminal = _terminal_materialization_certificate(graph, operation, config)
        if terminal is None:
            return _seal(certificate)
        certificate.update(terminal)
    elif (
        operation.operation_type == OperationType.COMMIT
        and str(operation.payload.get("mode", "")) != "answer"
    ):
        candidate_id = str(operation.payload.get("candidate_id", ""))
        claim = graph.nodes.get(candidate_id)
        valid = (
            isinstance(claim, ClaimNode)
            and claim.target_subgoal == subgoal.node_id
            and claim.branch_id == branch.branch_id
            and claim.status in _VIABLE_CLAIM_STATES
            and branch.assignments.get(subgoal.node_id) != candidate_id
            and subgoal.node_id not in branch.completed_subgoals
            and subgoal.status != SubgoalStatus.RESOLVED
            and operation.source_ids == [candidate_id]
        )
        if not valid:
            return _seal(certificate)
        certificate.update({
            "kind": "branch_assignment_commit",
            "mandatory": True,
            "deterministic": True,
            "source_claim_ids": [candidate_id],
            "successor_subgoal_ids": successor_ids,
            "preconditions": {
                "candidate_status": claim.status.value,
                "candidate_target_matches": True,
                "candidate_branch_matches": True,
                "target_unresolved": True,
                "branch_active": True,
                "assignment_absent": True,
            },
            "promised_effect": {
                "assignments_added": 1,
                "completed_subgoals_added": 1,
                "resolved_subgoals_added": 1,
                "successor_dependencies_reduced": len(successor_ids),
                "virtual_terminal_readout_exposed": int(virtual_terminal),
            },
        })
    elif (
        operation.operation_type == OperationType.BRANCH
        and str(operation.payload.get("mode", "")) == "assignments"
    ):
        candidate_ids = [str(value) for value in operation.payload.get("candidate_ids", [])]
        claims = [graph.nodes.get(value) for value in candidate_ids]
        valid = (
            len(candidate_ids) >= 2
            and len(candidate_ids) == len(set(candidate_ids))
            and all(
                isinstance(claim, ClaimNode)
                and claim.target_subgoal == subgoal.node_id
                and claim.branch_id == branch.branch_id
                and claim.status in _VIABLE_CLAIM_STATES
                for claim in claims
            )
            and subgoal.node_id not in branch.completed_subgoals
            and len(graph.active_branches()) - 1 + len(candidate_ids)
            <= graph.limits.max_active_branches
            and operation.source_ids == candidate_ids
        )
        if not valid:
            return _seal(certificate)
        certificate.update({
            "kind": "ambiguity_branch_materialization",
            "mandatory": True,
            "deterministic": True,
            "source_claim_ids": candidate_ids,
            "successor_subgoal_ids": successor_ids,
            "preconditions": {
                "candidate_count": len(candidate_ids),
                "all_candidates_viable": True,
                "target_incomplete": True,
                "parent_branch_active": True,
                "branch_capacity_available": True,
            },
            "promised_effect": {
                "parent_branches_archived": 1,
                "child_branches_created": len(candidate_ids),
                "branch_assignments_added": len(candidate_ids),
                "successor_dependencies_reduced": len(successor_ids),
                "virtual_terminal_readout_exposed": int(virtual_terminal),
            },
        })
    elif (
        operation.operation_type == OperationType.MERGE
        and str(operation.payload.get("mode", "")) == "validate_join"
        and bool(getattr(config, "certified_deterministic_join_allocation", False))
    ):
        candidate = join_candidate_from_operation(operation)
        derivation = deterministic_join_derivation(graph, candidate, config)
        existing = sum(
            row.accepted and row.signature == candidate.signature
            for row in graph.join_attempt_history
        )
        valid = bool(
            derivation is not None
            and operation.source_ids == list(candidate.premise_ids)
            and int(operation.payload.get("predicted_provider_calls", 1)) == 0
        )
        if not valid:
            return _seal(certificate)
        certificate.update({
            "kind": "deterministic_join_materialization",
            "mandatory": True,
            "deterministic": True,
            "provider_calls": 0,
            "source_claim_ids": list(candidate.premise_ids),
            "successor_subgoal_ids": successor_ids,
            "join_signature": candidate.signature,
            "preconditions": {
                "provider_free_derivation_proved": True,
                "premise_ids_match_operation_sources": True,
                "accepted_same_signature_before": existing,
                "validation_rule": derivation.validation_rule,
            },
            "promised_effect": {
                "accepted_join_attempts_added": 1,
                "join_claims_added": 1,
                "hyperedges_added": 1,
                "virtual_terminal_readout_exposed": int(virtual_terminal),
            },
        })
    else:
        return _seal(certificate)

    reachability = _successor_reachability_gain(
        graph, operation.target_id, successor_ids, virtual_terminal,
    )
    option = _successor_option_value(
        graph, operation.target_id, successor_ids, virtual_terminal,
    )
    certificate.update({
        "transition_certainty": 1.0,
        "successor_reachability_gain": reachability,
        "successor_option_value": option,
        "transition_redundancy": 0.0,
        "predicted_transition_value": max(0.0, min(1.0, reachability * option)),
    })
    return _seal(certificate)


def realized_transition_value(
    graph: DynamicReasoningHypergraphV2,
    certificate: dict[str, Any],
) -> tuple[bool, float, dict[str, Any]]:
    """Audit whether the promised deterministic state change actually occurred."""

    kind = str(certificate.get("kind", ""))
    target_id = str(certificate.get("target_subgoal", ""))
    branch_id = str(certificate.get("branch_id", ""))
    source_ids = [str(value) for value in certificate.get("source_claim_ids", [])]
    target = graph.nodes.get(target_id)
    if kind == "branch_assignment_commit":
        branch = graph.branches.get(branch_id)
        realized = bool(
            isinstance(target, SubgoalNode)
            and target.status == SubgoalStatus.RESOLVED
            and branch is not None
            and source_ids
            and branch.assignments.get(target_id) == source_ids[0]
            and target_id in branch.completed_subgoals
        )
        observations = {
            "target_resolved": bool(
                isinstance(target, SubgoalNode)
                and target.status == SubgoalStatus.RESOLVED
            ),
            "assignment_materialized": bool(
                branch is not None and source_ids
                and branch.assignments.get(target_id) == source_ids[0]
            ),
            "completion_materialized": bool(
                branch is not None and target_id in branch.completed_subgoals
            ),
        }
    elif kind == "ambiguity_branch_materialization":
        children = [
            row for row in graph.branches.values()
            if row.parent_branch_id == branch_id
            and row.assignments.get(target_id) in set(source_ids)
        ]
        parent = graph.branches.get(branch_id)
        realized = bool(
            parent is not None
            and parent.status == BranchStatus.ARCHIVED
            and len(children) == len(source_ids)
        )
        observations = {
            "parent_archived": bool(
                parent is not None and parent.status == BranchStatus.ARCHIVED
            ),
            "child_branches_created": len(children),
            "expected_child_branches": len(source_ids),
        }
    elif kind == "deterministic_join_materialization":
        signature = str(certificate.get("join_signature", ""))
        before = int((certificate.get("preconditions") or {}).get(
            "accepted_same_signature_before", 0,
        ))
        accepted = [
            row for row in graph.join_attempt_history
            if row.accepted and row.signature == signature
        ]
        conclusions = [
            graph.nodes.get(row.conclusion_node_id) for row in accepted
            if row.conclusion_node_id
        ]
        realized = bool(
            len(accepted) > before
            and any(
                isinstance(row, ClaimNode)
                and row.target_subgoal == target_id
                for row in conclusions
            )
        )
        observations = {
            "accepted_same_signature_before": before,
            "accepted_same_signature_after": len(accepted),
            "target_join_claim_materialized": any(
                isinstance(row, ClaimNode)
                and row.target_subgoal == target_id
                for row in conclusions
            ),
        }
    elif kind == "accepted_terminal_materialization":
        answer_id = str(certificate.get("answer_node_id", ""))
        answer = graph.nodes.get(answer_id)
        terminal = graph.terminal_beliefs.get(answer_id)
        branch = graph.branches.get(branch_id)
        realized = bool(
            answer is not None
            and getattr(answer, "status", None) == AnswerStatus.ACCEPTED
            and set(getattr(answer, "supporting_claims", [])) == set(source_ids)
            and terminal is not None
            and terminal.accepted
            and terminal.sufficient_chain
            and not terminal.rejection_reasons
            and branch is not None
            and branch.status == BranchStatus.COMPLETED
        )
        observations = {
            "answer_materialized": answer is not None,
            "answer_accepted": bool(
                answer is not None
                and getattr(answer, "status", None) == AnswerStatus.ACCEPTED
            ),
            "terminal_belief_materialized": terminal is not None,
            "terminal_belief_accepted": bool(
                terminal is not None and terminal.accepted
                and terminal.sufficient_chain and not terminal.rejection_reasons
            ),
            "branch_completed": bool(
                branch is not None and branch.status == BranchStatus.COMPLETED
            ),
        }
    else:
        realized = False
        observations = {"unsupported_certificate_kind": kind}
    predicted = float(certificate.get("predicted_transition_value", 0.0))
    return realized, predicted if realized else 0.0, observations


def _terminal_materialization_certificate(
    graph: DynamicReasoningHypergraphV2,
    operation: GraphOperation,
    config: Any,
) -> dict[str, Any] | None:
    """Recompute every terminal channel before certifying answer COMMIT.

    Absolute channels and proof coverage come from the sealed graph.  Relative
    channels are reconstructed from the complete, gold-free competition
    snapshot emitted by :class:`TerminalBeliefReadout`.  Consequently neither
    an operation family label nor a hand-written ``accepted`` flag is enough.
    """

    answer = operation.payload.get("answer")
    if not isinstance(answer, dict):
        return None
    profile = answer.get("terminal_belief")
    competition = answer.get("terminal_competition")
    if not isinstance(profile, dict) or not isinstance(competition, dict):
        return None

    answer_id = str(answer.get("node_id", ""))
    edge_id = str(answer.get("derivation_edge", ""))
    candidate_answer = str(answer.get("candidate_answer", "")).strip()
    claim_ids = [str(value) for value in answer.get("supporting_claims", [])]
    evidence_ids = [str(value) for value in answer.get("supporting_evidence", [])]
    initial_claim_ids = [str(value) for value in operation.source_ids]
    if any((
        not answer_id,
        answer_id in graph.nodes,
        not edge_id,
        edge_id in graph.hyperedges,
        not normalize_text(candidate_answer),
        str(answer.get("status", "")) != AnswerStatus.ACCEPTED.value,
        not claim_ids,
        len(claim_ids) != len(set(claim_ids)),
        not evidence_ids,
        len(evidence_ids) != len(set(evidence_ids)),
        not initial_claim_ids,
    )):
        return None

    claims = [graph.nodes.get(node_id) for node_id in claim_ids]
    viable = {
        CandidateStatus.COMMITTED,
        CandidateStatus.SCORED,
        CandidateStatus.RETAINED,
        CandidateStatus.REVISED,
    }
    if not all(
        isinstance(claim, ClaimNode)
        and claim.status in viable
        and (
            claim.branch_id == operation.branch_id
            or (
                bool(getattr(
                    config, "terminal_certificate_accepts_ancestor_claims", False,
                ))
                and _is_branch_ancestor(
                    graph, claim.branch_id, operation.branch_id,
                )
            )
        )
        for claim in claims
    ):
        return None
    if not all(isinstance(graph.nodes.get(node_id), EvidenceNode) for node_id in evidence_ids):
        return None
    closure = claim_closure(graph, initial_claim_ids)
    graph_evidence = list(dict.fromkeys(
        evidence_id
        for claim in claims
        for evidence_id in claim.evidence_refs
        if isinstance(graph.nodes.get(evidence_id), EvidenceNode)
    ))
    if set(closure) != set(claim_ids) or set(graph_evidence) != set(evidence_ids):
        return None
    if any(not claim.evidence_refs for claim in claims):
        return None

    proof = audit_graph_proof(
        graph, operation.target_id, operation.branch_id, initial_claim_ids,
    )
    absolute_support = min(float(claim.score.absolute_support) for claim in claims)
    evidence_gap = max(float(claim.score.evidence_gap) for claim in claims)
    contradiction = max(max(
        float(claim.score.raw.contradiction_risk),
        float(getattr(graph.belief_states.get(claim.node_id), "contradiction_pressure", 0.0)),
    ) for claim in claims)
    type_consistency = float(answer.get("answer_type_consistency", 0.0))

    rows = competition.get("candidate_values")
    if not isinstance(rows, list) or not rows:
        return None
    by_value: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        key = normalize_text(str(row.get("normalized_answer", "")))
        try:
            support = float(row.get("absolute_support"))
        except (TypeError, ValueError):
            return None
        if not key or key in by_value or not math.isfinite(support) or not 0.0 <= support <= 1.0:
            return None
        by_value[key] = support
    selected_key = normalize_text(candidate_answer)
    if selected_key not in by_value or not _close(by_value[selected_key], absolute_support):
        return None
    total = sum(by_value.values())
    weights = (
        {key: 1.0 / len(by_value) for key in by_value}
        if total <= 0.0 else {key: value / total for key, value in by_value.items()}
    )
    ranked = sorted(weights, key=lambda key: (-weights[key], key))
    if selected_key != ranked[0]:
        return None
    relative_weight = weights[selected_key]
    relative_margin = (
        relative_weight - weights[ranked[1]] if len(ranked) > 1 else 1.0
    )
    entropy = _normalized_entropy(list(weights.values()))

    unresolved = competition.get("unresolved_branch_ids", [])
    if not isinstance(unresolved, list) or any(
        _credible_unresolved_branch(
            graph, str(branch_id), operation.target_id, config,
        )
        for branch_id in sorted(set(str(value) for value in unresolved))
        if str(branch_id) != operation.branch_id
    ):
        return None

    profile_checks = {
        "answer_node_id": str(profile.get("answer_node_id", "")) == answer_id,
        "candidate_answer": normalize_text(str(profile.get("candidate_answer", ""))) == selected_key,
        "branch_id": str(profile.get("branch_id", "")) == operation.branch_id,
        "supporting_claims": set(str(value) for value in profile.get("supporting_claims", [])) == set(claim_ids),
        "supporting_evidence": set(str(value) for value in profile.get("supporting_evidence", [])) == set(evidence_ids),
        "absolute_support": _close(profile.get("absolute_support"), absolute_support),
        "relative_weight": _close(profile.get("relative_weight"), relative_weight),
        "entropy": _close(profile.get("entropy"), entropy),
        "competition_entropy": _close(profile.get("competition_entropy"), entropy),
        "evidence_gap": _close(profile.get("evidence_gap"), evidence_gap),
        "relative_margin": _close(profile.get("relative_margin"), relative_margin),
        "contradiction_pressure": _close(profile.get("contradiction_pressure"), contradiction),
        "answer_type_consistency": _close(profile.get("answer_type_consistency"), type_consistency),
        "chain_coverage": _close(profile.get("chain_coverage"), proof.dependency_coverage),
        "sufficient_chain": bool(profile.get("sufficient_chain", False)),
        "accepted": bool(profile.get("accepted", False)),
        "rejection_reasons": not profile.get("rejection_reasons", []),
    }
    configured_gate = all((
        proof.proof_connected,
        proof.dependency_coverage >= config.terminal_min_chain_coverage,
        absolute_support >= config.terminal_min_absolute_support,
        relative_margin >= config.terminal_min_relative_margin,
        entropy <= config.terminal_max_entropy,
        evidence_gap <= config.terminal_max_evidence_gap,
        contradiction < config.terminal_max_contradiction,
        type_consistency >= config.terminal_min_type_consistency,
    ))
    if not all(profile_checks.values()) or not configured_gate:
        return None

    return {
        "certificate_version": (
            "certified-transition-option-v2.4.3.5"
            if getattr(config, "terminal_certificate_accepts_ancestor_claims", False)
            else "certified-transition-option-v2.4.3.4"
        ),
        "kind": "accepted_terminal_materialization",
        "mandatory": True,
        "deterministic": True,
        "answer_node_id": answer_id,
        "derivation_edge_id": edge_id,
        "source_claim_ids": claim_ids,
        "successor_subgoal_ids": [],
        "preconditions": {
            "branch_active": True,
            "answer_and_edge_ids_new": True,
            "all_source_claims_viable": True,
            "all_evidence_exists": True,
            "proof_connected": True,
            "terminal_channels_recomputed": profile_checks,
            "configured_terminal_gate_passed": True,
            "competition_candidate_count": len(by_value),
        },
        "promised_effect": {
            "answer_nodes_added": 1,
            "derivation_hyperedges_added": 1,
            "terminal_beliefs_added": 1,
            "branches_completed": 1,
        },
    }


def _credible_unresolved_branch(
    graph: DynamicReasoningHypergraphV2,
    branch_id: str,
    target_subgoal: str,
    config: Any,
) -> bool:
    branch = graph.branches.get(branch_id)
    if branch is None:
        return True
    for claim in graph.claims():
        if claim.target_subgoal != target_subgoal or claim.branch_id != branch_id:
            continue
        if claim.status not in {
            CandidateStatus.SCORED, CandidateStatus.RETAINED,
            CandidateStatus.REVISED, CandidateStatus.COMMITTED,
        }:
            continue
        semantics = graph.claim_semantics.get(claim.node_id)
        is_projection = bool(
            claim.provenance.metadata.get("answers_subgoal", False)
            or (semantics is not None and semantics.qualifiers.get("projection_premise_id"))
        )
        if is_projection and all((
            claim.score.absolute_support >= config.terminal_min_absolute_support,
            claim.score.evidence_gap <= config.terminal_max_evidence_gap,
            claim.score.raw.grounding >= config.join_min_premise_support,
            claim.score.raw.type_match >= config.terminal_min_type_consistency,
            claim.score.raw.contradiction_risk < config.terminal_max_contradiction,
            bool(claim.evidence_refs),
        )):
            return True
    return False


def _is_branch_ancestor(
    graph: DynamicReasoningHypergraphV2,
    ancestor_id: str,
    descendant_id: str,
) -> bool:
    """Return whether a claim-owning branch is on the child's sealed lineage."""

    seen = set()
    current = graph.branches.get(descendant_id)
    while current is not None and current.branch_id not in seen:
        seen.add(current.branch_id)
        parent_id = current.parent_branch_id
        if parent_id == ancestor_id:
            return True
        current = graph.branches.get(parent_id) if parent_id else None
    return False


def _normalized_entropy(weights: list[float]) -> float:
    if len(weights) <= 1:
        return 0.0
    value = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
    return max(0.0, min(1.0, value / math.log(len(weights))))


def _close(left: Any, right: float, tolerance: float = 1e-8) -> bool:
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and abs(value - float(right)) <= tolerance


def _successor_reachability_gain(
    graph: DynamicReasoningHypergraphV2,
    target_id: str,
    successor_ids: list[str],
    virtual_terminal: bool,
) -> float:
    gains = []
    for successor_id in successor_ids:
        successor = graph.nodes.get(successor_id)
        if not isinstance(successor, SubgoalNode):
            continue
        unresolved = [
            value for value in successor.dependencies
            if value in graph.nodes
            and getattr(graph.nodes[value], "status", None) != SubgoalStatus.RESOLVED
        ]
        if target_id in unresolved:
            gains.append(1.0 / max(1, len(unresolved)))
    if virtual_terminal:
        gains.append(1.0)
    return max(gains, default=0.0)


def _successor_option_value(
    graph: DynamicReasoningHypergraphV2,
    target_id: str,
    successor_ids: list[str],
    virtual_terminal: bool,
) -> float:
    values = [1.0] if virtual_terminal else []
    for node_id in successor_ids or [target_id]:
        belief = graph.belief_states.get(node_id)
        distance = terminal_dependency_distance(graph, node_id)
        reachability = 0.0 if distance is None else 1.0 / (1.0 + distance)
        values.append(max(
            reachability,
            float(getattr(belief, "computation_heat", 0.0)),
            float(getattr(belief, "downstream_answer_impact", 0.0)),
            float(getattr(belief, "dependency_unlock_value", 0.0)),
        ))
    return max(0.0, min(1.0, max(values, default=0.0)))


def _seal(certificate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(certificate)
    payload["state_certificate_hash"] = stable_hash({
        key: value for key, value in payload.items()
        if key != "state_certificate_hash"
    })
    return payload
