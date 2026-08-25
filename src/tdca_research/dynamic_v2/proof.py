from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..dynamic.graph import CandidateStatus, ClaimNode, EvidenceNode
from .graph import DynamicReasoningHypergraphV2


@dataclass(frozen=True)
class GraphProofAudit:
    """Gold-free structural audit of one proposed answer proof."""

    root_subgoal_id: str
    branch_id: str
    initial_claim_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    required_subgoal_ids: tuple[str, ...]
    covered_subgoal_ids: tuple[str, ...]
    evidence_leaf_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    dependency_coverage: float
    evidence_leaf_coverage: float
    distinct_evidence_leaf_ratio: float
    proof_connected: bool
    graph_proof_completion: bool
    proof_depth: int
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


def claim_closure(
    graph: DynamicReasoningHypergraphV2,
    initial_claim_ids: list[str] | tuple[str, ...],
) -> list[str]:
    """Return the deterministic dependency closure of existing claim nodes."""
    closure: set[str] = set()
    queue = list(dict.fromkeys(str(value) for value in initial_claim_ids))
    while queue:
        claim_id = queue.pop(0)
        if claim_id in closure:
            continue
        node = graph.nodes.get(claim_id)
        if not isinstance(node, ClaimNode):
            continue
        closure.add(claim_id)
        queue.extend(
            dependency for dependency in node.dependency_claim_ids
            if dependency not in closure
        )
    return sorted(closure)


def required_subgoal_closure(
    graph: DynamicReasoningHypergraphV2,
    root_subgoal_id: str,
) -> list[str]:
    required: set[str] = set()
    queue = [root_subgoal_id]
    while queue:
        subgoal_id = queue.pop(0)
        if subgoal_id in required:
            continue
        required.add(subgoal_id)
        queue.extend(graph.execution_graph.dependencies.get(subgoal_id, []))
    return sorted(required or {root_subgoal_id})


def audit_graph_proof(
    graph: DynamicReasoningHypergraphV2,
    root_subgoal_id: str,
    branch_id: str,
    initial_claim_ids: list[str] | tuple[str, ...],
) -> GraphProofAudit:
    """Audit dependency coverage, evidence leaves and hyperedge connectivity.

    The audit consumes only graph state available at inference time.  It never
    reads gold answers or oracle decomposition annotations.
    """
    initial = tuple(dict.fromkeys(str(value) for value in initial_claim_ids))
    claim_ids = claim_closure(graph, initial)
    claim_id_set = set(claim_ids)
    claims = [graph.nodes[value] for value in claim_ids]
    claims = [value for value in claims if isinstance(value, ClaimNode)]
    required = required_subgoal_closure(graph, root_subgoal_id)
    branch = graph.branches.get(branch_id)
    reasons: set[str] = set()

    if not initial:
        reasons.add("empty_initial_claim_set")
    missing_initial = [value for value in initial if value not in claim_id_set]
    if missing_initial:
        reasons.add("missing_initial_claim")
    if branch is None:
        reasons.add("missing_branch")

    covered: set[str] = set()
    if any(claim.target_subgoal == root_subgoal_id for claim in claims):
        covered.add(root_subgoal_id)
    else:
        reasons.add("root_subgoal_uncovered")
    if branch is not None:
        for subgoal_id in required:
            if subgoal_id == root_subgoal_id:
                continue
            assignment = branch.assignments.get(subgoal_id)
            if (
                assignment in claim_id_set
                and isinstance(graph.nodes.get(str(assignment)), ClaimNode)
                and graph.nodes[str(assignment)].target_subgoal == subgoal_id
            ):
                covered.add(subgoal_id)
            else:
                reasons.add("dependency_subgoal_uncovered")

    statuses_valid = all(
        claim.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
        for claim in claims
    )
    if not statuses_valid:
        reasons.add("invalid_or_archived_claim")

    proof_connected = bool(claims) and not missing_initial
    invalid_edges = set(graph.invalidated_hyperedges)
    for claim in claims:
        if any(value not in claim_id_set for value in claim.dependency_claim_ids):
            proof_connected = False
            reasons.add("dependency_claim_outside_closure")
        semantics = graph.claim_semantics.get(claim.node_id)
        if semantics is not None and semantics.join_depth > 0:
            valid_edges = [
                edge for edge in graph.hyperedges.values()
                if edge.target_node == claim.node_id
                and edge.edge_id not in invalid_edges
                and len(edge.source_node_set) >= 2
                and set(edge.source_node_set).issubset(claim_id_set)
            ]
            if not valid_edges:
                proof_connected = False
                reasons.add("joined_claim_lacks_valid_hyperedge")

    leaves = [
        claim for claim in claims
        if graph.claim_semantics.get(claim.node_id) is None
        or graph.claim_semantics[claim.node_id].join_depth == 0
    ]
    grounded_leaves = [
        claim for claim in leaves
        if claim.evidence_refs
        and all(isinstance(graph.nodes.get(value), EvidenceNode) for value in claim.evidence_refs)
    ]
    if not leaves:
        reasons.add("no_evidence_leaves")
    elif len(grounded_leaves) != len(leaves):
        reasons.add("ungrounded_evidence_leaf")
    evidence_ids = sorted({
        evidence_id for claim in grounded_leaves for evidence_id in claim.evidence_refs
    })
    distinct_passages = {
        graph.nodes[evidence_id].passage_id
        for evidence_id in evidence_ids
        if isinstance(graph.nodes.get(evidence_id), EvidenceNode)
    }
    dependency_coverage = len(covered) / max(1, len(required))
    evidence_leaf_coverage = len(grounded_leaves) / max(1, len(leaves))
    distinct_ratio = min(1.0, len(distinct_passages) / max(1, len(leaves)))
    complete = bool(
        claims
        and statuses_valid
        and proof_connected
        and dependency_coverage == 1.0
        and evidence_leaf_coverage == 1.0
    )
    return GraphProofAudit(
        root_subgoal_id=root_subgoal_id,
        branch_id=branch_id,
        initial_claim_ids=initial,
        claim_ids=tuple(claim_ids),
        required_subgoal_ids=tuple(required),
        covered_subgoal_ids=tuple(sorted(covered)),
        evidence_leaf_ids=tuple(sorted(claim.node_id for claim in leaves)),
        evidence_ids=tuple(evidence_ids),
        dependency_coverage=dependency_coverage,
        evidence_leaf_coverage=evidence_leaf_coverage,
        distinct_evidence_leaf_ratio=distinct_ratio,
        proof_connected=proof_connected,
        graph_proof_completion=complete,
        proof_depth=max((
            graph.claim_semantics[claim.node_id].join_depth
            for claim in claims if claim.node_id in graph.claim_semantics
        ), default=0),
        reason_codes=tuple(sorted(reasons)),
    )
