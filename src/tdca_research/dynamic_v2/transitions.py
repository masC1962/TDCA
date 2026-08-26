from __future__ import annotations

from typing import Any

from ..dynamic.graph import (
    BranchStatus,
    CandidateStatus,
    ClaimNode,
    GraphOperation,
    OperationType,
    SubgoalNode,
    SubgoalStatus,
)
from ..utils import stable_hash
from .graph import DynamicReasoningHypergraphV2
from .obligations import terminal_dependency_distance


_VIABLE_CLAIM_STATES = {
    CandidateStatus.SCORED,
    CandidateStatus.RETAINED,
    CandidateStatus.REOPENED,
    CandidateStatus.REVISED,
}


def certified_transition_value(
    graph: DynamicReasoningHypergraphV2,
    operation: GraphOperation,
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
    else:
        realized = False
        observations = {"unsupported_certificate_kind": kind}
    predicted = float(certificate.get("predicted_transition_value", 0.0))
    return realized, predicted if realized else 0.0, observations


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
