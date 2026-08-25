from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..dynamic.graph import (
    AnswerStatus,
    BranchStatus,
    CandidateStatus,
    ClaimNode,
    EvidenceNode,
    GraphOperation,
    OperationType,
    SubgoalStatus,
)
from ..utils import stable_hash
from .graph import (
    DynamicReasoningHypergraphV2,
    ProofObligationSnapshot,
    ProofObligationState,
)


OPEN = "OPEN"
CLOSED = "CLOSED"
BLOCKED = "BLOCKED"


def refresh_proof_obligations(
    graph: DynamicReasoningHypergraphV2,
    trigger_operation_id: str,
    *,
    max_retrieval_rounds: int,
) -> None:
    """Project proof deficits from controller-owned graph state.

    The projection is deterministic and contains no answer labels.  Existing
    rows remain addressable after closure; a snapshot provides the append-only
    audit trail needed to explain later scheduling and termination decisions.
    """

    graph.proof_obligation_version = "proof-obligation-state-v2.4.3"
    previous = graph.proof_obligations
    current: dict[str, ProofObligationState] = {}
    active_branches = [
        row.branch_id for row in graph.branches.values()
        if row.status == BranchStatus.ACTIVE
    ] or sorted(graph.branches) or ["branch_root"]
    for subgoal in sorted(graph.subgoals(), key=lambda row: row.node_id):
        if subgoal.status == SubgoalStatus.ARCHIVED:
            continue
        branch_ids = sorted(set(active_branches) | {
            row.branch_id for row in graph.claims(subgoal.node_id)
        } | {
            row.branch_id for row in graph.evidence()
            if row.target_subgoal == subgoal.node_id
        })
        for branch_id in branch_ids:
            _project_subgoal_branch(
                graph, current, previous, subgoal, branch_id,
                max_retrieval_rounds=max_retrieval_rounds,
            )

    # Preserve closed rows so an allocation can prove that it discharged a
    # previously visible obligation without rewriting history.
    for obligation_id, old in previous.items():
        if obligation_id in current:
            continue
        current[obligation_id] = ProofObligationState(
            **{
                **asdict(old),
                "status": CLOSED,
                "severity": 0.0,
                "updated_at_step": graph.step,
                "reason_codes": sorted(set(old.reason_codes + ["graph_condition_satisfied"])),
                "provenance_event_ids": sorted(set(
                    old.provenance_event_ids + [trigger_operation_id]
                )),
            }
        )
    graph.proof_obligations = current
    graph.proof_obligation_history.append(ProofObligationSnapshot(
        snapshot_id=f"obligation_snapshot_{graph.step:04d}_{trigger_operation_id}",
        step=graph.step,
        trigger_operation_id=trigger_operation_id,
        obligations=[asdict(current[key]) for key in sorted(current)],
    ))


def operation_obligation_targets(
    graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
) -> list[str]:
    closes = {
        OperationType.RETRIEVE: {"missing_evidence", "retrieval_exhausted"},
        OperationType.BRANCH: {"missing_claim", "missing_binding", "extraction_exhausted"},
        OperationType.VERIFY: {"missing_verification", "contradiction"},
        OperationType.MERGE: {"missing_join_premise", "terminal_disconnected_join"},
        OperationType.EXPAND: {"missing_binding", "missing_claim"},
        OperationType.REVISE: {"contradiction", "terminal_disconnected_join"},
        OperationType.PRUNE: {"contradiction"},
        OperationType.COMMIT: {
            "missing_evidence", "missing_claim", "missing_verification",
            "missing_join_premise", "terminal_disconnected_join",
        },
    }[operation.operation_type]
    return sorted(
        obligation_id for obligation_id, row in graph.proof_obligations.items()
        if row.status == OPEN
        and row.obligation_type in closes
        and row.target_subgoal == operation.target_id
        and (row.branch_id == operation.branch_id or not operation.branch_id)
    )


def graph_local_operation_value(
    graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
) -> dict[str, float]:
    target_ids = operation_obligation_targets(graph, operation)
    obligations = [graph.proof_obligations[value] for value in target_ids]
    closure = max((row.severity for row in obligations), default=0.0)
    missing_premise = max((
        row.severity for row in obligations
        if row.obligation_type in {"missing_binding", "missing_join_premise"}
    ), default=0.0)
    payload_reducibility = _unit(float(operation.payload.get("proof_gap_reducibility", 0.0)))
    payload_unlock = _unit(float(operation.payload.get("feasibility_unlock", 0.0)))
    missing_premise = max(missing_premise, payload_reducibility, payload_unlock)
    distance = terminal_dependency_distance(graph, operation.target_id)
    terminal_reachability = 0.0 if distance is None else 1.0 / (1.0 + distance)
    if obligations and not any(row.terminal_reachable for row in obligations):
        terminal_reachability = 0.0
    region_claims = [
        row for row in graph.claims(operation.target_id)
        if row.branch_id == operation.branch_id
        and row.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
    ]
    candidate_reachability = terminal_reachability * max((
        graph.belief_states.get(row.node_id).absolute_support
        if graph.belief_states.get(row.node_id) is not None else row.score.absolute_support
        for row in region_claims
    ), default=0.0)
    evidence = [
        row for row in graph.evidence()
        if row.target_subgoal == operation.target_id and row.branch_id == operation.branch_id
    ]
    evidence_path = terminal_reachability * _unit(len(evidence) / 2.0)
    disconnected_join = any(
        row.obligation_type == "terminal_disconnected_join" for row in obligations
    )
    dead_end_risk = 1.0 if disconnected_join else _unit(sum(
        row.status == BLOCKED for row in graph.proof_obligations.values()
        if row.target_subgoal == operation.target_id and row.branch_id == operation.branch_id
    ) / 2.0)
    return {
        "obligation_closure": _unit(closure),
        "terminal_reachability": _unit(terminal_reachability),
        "missing_premise_reduction": _unit(missing_premise),
        "candidate_reachability": _unit(candidate_reachability),
        "evidence_path": _unit(evidence_path),
        "dead_end_risk": _unit(dead_end_risk),
    }


def terminal_dependency_distance(
    graph: DynamicReasoningHypergraphV2, source_id: str,
) -> int | None:
    terminal_ids = {
        row.node_id for row in graph.subgoals()
        if row.terminal or row.node_id == "subgoal_root"
    }
    if source_id in terminal_ids:
        return 0
    # dependency -> dependent is the forward proof direction.
    forward: dict[str, set[str]] = {}
    for subgoal in graph.subgoals():
        for dependency in subgoal.dependencies:
            forward.setdefault(dependency, set()).add(subgoal.node_id)
    frontier = [(source_id, 0)]
    seen: set[str] = set()
    while frontier:
        current, distance = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        for target in sorted(forward.get(current, set())):
            if target in terminal_ids:
                return distance + 1
            frontier.append((target, distance + 1))
    return None


def dead_end_certificate(
    graph: DynamicReasoningHypergraphV2,
    packets: list[Any],
    remaining_budget: dict[str, int],
) -> dict[str, Any]:
    open_rows = [
        asdict(row) for row in graph.proof_obligations.values()
        if row.status == OPEN
    ]
    blocked_rows = [
        asdict(row) for row in graph.proof_obligations.values()
        if row.status == BLOCKED
    ]
    return {
        "certificate_version": "proof-obligation-dead-end-v2.4.3",
        "step": graph.step,
        "open_obligations": sorted(open_rows, key=lambda row: row["obligation_id"]),
        "blocked_obligations": sorted(blocked_rows, key=lambda row: row["obligation_id"]),
        "remaining_budget": dict(remaining_budget),
        "candidate_operations": [
            {
                "allocation_id": row.allocation_id,
                "operation_id": row.operation.operation_id,
                "operation_type": row.operation.operation_type.value,
                "target_obligation_ids": list(row.target_obligation_ids),
                "gross_opportunity": float(row.predicted_gross_opportunity),
                "absolute_cost": float(row.predicted_normalized_cost),
                "net_evc": float(row.predicted_evc),
            }
            for row in packets
        ],
        "feasible_unselected_join_count": sum(
            row.operation.operation_type == OperationType.MERGE for row in packets[1:]
        ),
        "exhaustion_evidence": sorted({
            reason
            for row in (blocked_rows or open_rows)
            for reason in row.get("reason_codes", [])
        } or ({"no_open_proof_obligation"} if not open_rows else set())),
    }


def _project_subgoal_branch(
    graph: DynamicReasoningHypergraphV2,
    current: dict[str, ProofObligationState],
    previous: dict[str, ProofObligationState],
    subgoal: Any,
    branch_id: str,
    *,
    max_retrieval_rounds: int,
) -> None:
    claims = [
        row for row in graph.claims(subgoal.node_id)
        if row.branch_id == branch_id
        and row.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
    ]
    evidence = [
        row for row in graph.evidence()
        if row.target_subgoal == subgoal.node_id and row.branch_id == branch_id
    ]
    dependencies = [
        value for value in subgoal.dependencies
        if value in graph.nodes
    ]
    unresolved_dependencies = [
        value for value in dependencies
        if getattr(graph.nodes[value], "status", None) != SubgoalStatus.RESOLVED
    ]
    retrieval_attempts = [
        row for row in graph.retrieval_attempt_history
        if row.target_subgoal == subgoal.node_id and row.branch_id == branch_id
    ]
    extraction_failures = [
        row for row in graph.operation_outcome_history
        if row.operation_family == "branch:extract_typed"
        and not row.progressed
        and row.region_key
        and any(
            allocation.allocation_id == row.allocation_id
            and subgoal.node_id in allocation.target_region
            for allocation in graph.allocation_history
        )
    ]
    terminal_reachable = terminal_dependency_distance(graph, subgoal.node_id) is not None
    provenance = [row.attempt_id for row in retrieval_attempts]
    if subgoal.status != SubgoalStatus.RESOLVED and unresolved_dependencies:
        _put(
            current, previous, graph, subgoal.node_id, branch_id, "missing_binding",
            OPEN, 1.0, terminal_reachable, unresolved_dependencies, [],
            ["unresolved_dependency"], provenance,
        )
    if subgoal.status != SubgoalStatus.RESOLVED and not evidence:
        exhausted = len(retrieval_attempts) >= max_retrieval_rounds
        _put(
            current, previous, graph, subgoal.node_id, branch_id,
            "retrieval_exhausted" if exhausted else "missing_evidence",
            BLOCKED if exhausted else OPEN, 1.0, terminal_reachable, [], [],
            ["retrieval_round_cap_reached"] if exhausted else ["no_grounding_evidence"],
            provenance,
        )
    elif subgoal.status != SubgoalStatus.RESOLVED and evidence and not claims:
        exhausted = len(extraction_failures) >= 2
        _put(
            current, previous, graph, subgoal.node_id, branch_id,
            "extraction_exhausted" if exhausted else "missing_claim",
            BLOCKED if exhausted else OPEN, 0.9, terminal_reachable,
            [row.node_id for row in evidence], [],
            ["typed_extraction_failed"] if exhausted else ["evidence_not_extracted"],
            provenance + [row.outcome_id for row in extraction_failures],
        )
    proposed = [row for row in claims if row.status == CandidateStatus.PROPOSED]
    if proposed:
        _put(
            current, previous, graph, subgoal.node_id, branch_id,
            "missing_verification", OPEN, 0.75, terminal_reachable,
            [row.node_id for row in proposed], [], ["unscored_claim"], provenance,
        )
    joined = [
        row for row in claims
        if graph.claim_semantics.get(row.node_id)
        and graph.claim_semantics[row.node_id].join_depth > 0
    ]
    if dependencies and claims and not joined and subgoal.status != SubgoalStatus.RESOLVED:
        _put(
            current, previous, graph, subgoal.node_id, branch_id,
            "missing_join_premise", OPEN, 0.85, terminal_reachable,
            dependencies + [row.node_id for row in claims], [],
            ["dependency_not_composed"], provenance,
        )
    if joined and not terminal_reachable:
        _put(
            current, previous, graph, subgoal.node_id, branch_id,
            "terminal_disconnected_join", OPEN, 1.0, False,
            [row.node_id for row in joined], [], ["no_terminal_dependency_path"],
            provenance,
        )
    contradictory = [
        row.node_id for row in claims
        if graph.belief_states.get(row.node_id) is not None
        and graph.belief_states[row.node_id].contradiction_pressure > 0.0
    ]
    if contradictory:
        _put(
            current, previous, graph, subgoal.node_id, branch_id,
            "contradiction", OPEN,
            max(graph.belief_states[value].contradiction_pressure for value in contradictory),
            terminal_reachable, contradictory, [], ["active_contradiction"], provenance,
        )


def _put(
    current: dict[str, ProofObligationState],
    previous: dict[str, ProofObligationState],
    graph: DynamicReasoningHypergraphV2,
    target: str,
    branch: str,
    kind: str,
    status: str,
    severity: float,
    terminal_reachable: bool,
    required: list[str],
    satisfied: list[str],
    reasons: list[str],
    provenance: list[str],
) -> None:
    obligation_id = "obligation_" + stable_hash({
        "target": target, "branch": branch, "kind": kind,
    })[:16]
    old = previous.get(obligation_id)
    current[obligation_id] = ProofObligationState(
        obligation_id=obligation_id,
        target_subgoal=target,
        branch_id=branch,
        obligation_type=kind,
        status=status,
        severity=_unit(severity),
        terminal_reachable=bool(terminal_reachable),
        required_node_ids=sorted(set(required)),
        satisfied_by_node_ids=sorted(set(satisfied)),
        reason_codes=sorted(set(reasons)),
        provenance_event_ids=sorted(set(provenance)),
        created_at_step=old.created_at_step if old is not None else graph.step,
        updated_at_step=graph.step,
    )


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
