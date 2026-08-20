from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import networkx as nx

from ..utils import stable_hash


class GraphInvariantError(ValueError):
    pass


class GraphBudgetExceeded(GraphInvariantError):
    """Expected safety-cap exhaustion, distinct from an invalid graph."""

    pass


class NodeKind(str, Enum):
    SUBGOAL = "subgoal"
    CLAIM = "claim"
    EVIDENCE = "evidence"
    ANSWER = "answer"


class SubgoalStatus(str, Enum):
    UNRESOLVED = "unresolved"
    ACTIVE = "active"
    PARTIAL = "partial"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    RETAINED = "retained"
    SCORED = "scored"
    COMMITTED = "committed"
    REOPENED = "reopened"
    REVISED = "revised"
    ARCHIVED = "archived"
    INVALID = "invalid"


class AnswerStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class BranchStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class OperationType(str, Enum):
    EXPAND = "EXPAND"
    BRANCH = "BRANCH"
    RETRIEVE = "RETRIEVE"
    VERIFY = "VERIFY"
    MERGE = "MERGE"
    PRUNE = "PRUNE"
    COMMIT = "COMMIT"
    REVISE = "REVISE"


@dataclass
class RevisionRecord:
    step: int
    operation_id: str
    reason: str
    before: dict[str, Any]
    after: dict[str, Any]


@dataclass
class Provenance:
    source: str
    operation_id: str
    created_at_step: int
    source_node_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    retrieval_query: str = ""
    retriever_identity: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationSignals:
    """Independent raw evidence scores, never a relative candidate probability."""

    grounding: float = 0.0
    entailment: float = 0.0
    type_match: float = 0.0
    dependency_consistency: float = 0.0
    retrieval_support: float = 0.0
    contradiction_risk: float = 0.0
    raw_model_confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class CandidateScoreProfile:
    raw: VerificationSignals = field(default_factory=VerificationSignals)
    absolute_support: float = 0.0
    relative_weight: float = 0.0
    set_entropy: float = 0.0
    evidence_gap: float = 1.0
    scoring_version: str = "dh-independent-v1"


@dataclass
class SubgoalNode:
    node_id: str
    question_template: str
    instantiated_question: str
    dependencies: list[str]
    variable_bindings: dict[str, str]
    answer_type: str
    terminal: bool
    status: SubgoalStatus
    confidence: float
    uncertainty: float
    created_at_step: int
    provenance: Provenance
    revision_history: list[RevisionRecord] = field(default_factory=list)

    @property
    def kind(self) -> NodeKind:
        return NodeKind.SUBGOAL


@dataclass
class ClaimNode:
    node_id: str
    subject: str
    relation: str
    value: str
    answer_type: str
    target_subgoal: str
    branch_id: str
    evidence_refs: list[str]
    dependency_claim_ids: list[str]
    score: CandidateScoreProfile
    status: CandidateStatus
    contradiction_links: list[str]
    created_at_step: int
    provenance: Provenance
    revision_history: list[RevisionRecord] = field(default_factory=list)

    @property
    def kind(self) -> NodeKind:
        return NodeKind.CLAIM


@dataclass
class EvidenceNode:
    node_id: str
    document_id: str
    passage_id: str
    title: str
    source_span: str
    retrieval_rank: int
    retrieval_score: float
    retrieval_query: str
    retriever_identity: str
    branch_id: str
    target_subgoal: str
    created_at_step: int
    provenance: Provenance
    revision_history: list[RevisionRecord] = field(default_factory=list)

    @property
    def kind(self) -> NodeKind:
        return NodeKind.EVIDENCE


@dataclass
class AnswerNode:
    node_id: str
    candidate_answer: str
    answer_type: str
    supporting_claims: list[str]
    supporting_evidence: list[str]
    derivation_edge: str
    branch_id: str
    confidence: float
    answer_type_consistency: float
    contradiction_risk: float
    status: AnswerStatus
    created_at_step: int
    provenance: Provenance
    revision_history: list[RevisionRecord] = field(default_factory=list)

    @property
    def kind(self) -> NodeKind:
        return NodeKind.ANSWER


GraphNode: TypeAlias = SubgoalNode | ClaimNode | EvidenceNode | AnswerNode


@dataclass
class Hyperedge:
    edge_id: str
    source_node_set: list[str]
    target_node: str
    inference_type: str
    confidence: float
    supporting_evidence: list[str]
    creation_reason: str
    created_by_operation: str
    created_at_step: int
    provenance: Provenance


@dataclass
class BranchState:
    branch_id: str
    parent_branch_id: str | None
    assignments: dict[str, str]
    completed_subgoals: list[str]
    score: float
    status: BranchStatus
    created_at_step: int
    revision_count: int = 0
    last_revision_step: int = -1
    history: list[RevisionRecord] = field(default_factory=list)


@dataclass
class ExecutionDependencyGraph:
    """Acyclic executable dependencies, separate from structural belief links."""

    dependencies: dict[str, list[str]] = field(default_factory=dict)

    def add_node(self, node_id: str, dependencies: list[str]) -> None:
        if node_id in self.dependencies:
            raise GraphInvariantError(f"duplicate execution node {node_id}")
        self.dependencies[node_id] = list(dict.fromkeys(dependencies))
        self.validate()

    def replace_dependencies(self, node_id: str, dependencies: list[str]) -> None:
        if node_id not in self.dependencies:
            raise GraphInvariantError(f"unknown execution node {node_id}")
        previous = self.dependencies[node_id]
        self.dependencies[node_id] = list(dict.fromkeys(dependencies))
        try:
            self.validate()
        except Exception:
            self.dependencies[node_id] = previous
            raise

    def validate(self) -> None:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.dependencies)
        for target, sources in self.dependencies.items():
            for source in sources:
                if source not in self.dependencies:
                    raise GraphInvariantError(f"execution dependency {source} does not exist")
                if source == target:
                    raise GraphInvariantError("execution self-cycle")
                graph.add_edge(source, target)
        if not nx.is_directed_acyclic_graph(graph):
            raise GraphInvariantError("execution dependency graph must be acyclic")

    def depth(self) -> int:
        """Return the longest executable dependency chain in nodes."""
        self.validate()
        if not self.dependencies:
            return 0
        graph = nx.DiGraph()
        graph.add_nodes_from(self.dependencies)
        for target, sources in self.dependencies.items():
            graph.add_edges_from((source, target) for source in sources)
        return nx.dag_longest_path_length(graph) + 1

    def to_dict(self) -> dict[str, Any]:
        return {"dependencies": {key: list(value) for key, value in sorted(self.dependencies.items())}}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExecutionDependencyGraph":
        graph = cls({str(key): [str(item) for item in values] for key, values in value.get("dependencies", {}).items()})
        graph.validate()
        return graph


@dataclass
class GraphLimits:
    max_candidates_per_subgoal: int
    max_active_branches: int
    max_graph_nodes: int
    max_hyperedges: int
    max_graph_revisions: int
    max_revision_per_candidate: int
    max_graph_depth: int
    max_graph_operations: int
    max_retrieval_calls: int


@dataclass
class GraphOperation:
    operation_id: str
    operation_type: OperationType
    target_id: str
    source_ids: list[str]
    branch_id: str
    payload: dict[str, Any]
    reason: str
    proposed_by: str
    estimated_cost: dict[str, float] = field(default_factory=dict)
    utility_components: dict[str, float] = field(default_factory=dict)


@dataclass
class AppliedOperation:
    operation_id: str
    operation_type: OperationType
    step: int
    branch_id: str
    graph_before_hash: str
    graph_after_hash: str
    created_nodes: list[str]
    updated_nodes: list[str]
    pruned_nodes: list[str]
    created_hyperedges: list[str]
    reason: str
    payload_digest: str


@dataclass
class DynamicReasoningHypergraph:
    question: str
    limits: GraphLimits
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    hyperedges: dict[str, Hyperedge] = field(default_factory=dict)
    branches: dict[str, BranchState] = field(default_factory=dict)
    execution_graph: ExecutionDependencyGraph = field(default_factory=ExecutionDependencyGraph)
    operation_history: list[AppliedOperation] = field(default_factory=list)
    revision_history: list[RevisionRecord] = field(default_factory=list)
    step: int = 0
    retrieval_calls: int = 0

    def node(self, node_id: str, expected: type | None = None) -> GraphNode:
        if node_id not in self.nodes:
            raise GraphInvariantError(f"unknown node {node_id}")
        value = self.nodes[node_id]
        if expected is not None and not isinstance(value, expected):
            raise GraphInvariantError(f"node {node_id} is not {expected.__name__}")
        return value

    def subgoals(self) -> list[SubgoalNode]:
        return [node for node in self.nodes.values() if isinstance(node, SubgoalNode)]

    def claims(self, subgoal_id: str | None = None, branch_id: str | None = None) -> list[ClaimNode]:
        return [
            node for node in self.nodes.values()
            if isinstance(node, ClaimNode)
            and (subgoal_id is None or node.target_subgoal == subgoal_id)
            and (branch_id is None or node.branch_id == branch_id)
        ]

    def evidence(self, subgoal_id: str | None = None, branch_id: str | None = None) -> list[EvidenceNode]:
        return [
            node for node in self.nodes.values()
            if isinstance(node, EvidenceNode)
            and (subgoal_id is None or node.target_subgoal == subgoal_id)
            and (branch_id is None or node.branch_id == branch_id)
        ]

    def answers(self) -> list[AnswerNode]:
        return [node for node in self.nodes.values() if isinstance(node, AnswerNode)]

    def active_branches(self) -> list[BranchState]:
        return [branch for branch in self.branches.values() if branch.status == BranchStatus.ACTIVE]

    def state_payload(self) -> dict[str, Any]:
        """Canonical mutable state, excluding audit history to avoid hash recursion."""
        return {
            "question": self.question,
            "limits": _primitive(self.limits),
            "nodes": {key: _node_dict(value) for key, value in sorted(self.nodes.items())},
            "hyperedges": {key: _primitive(value) for key, value in sorted(self.hyperedges.items())},
            "branches": {key: _primitive(value) for key, value in sorted(self.branches.items())},
            "execution_graph": self.execution_graph.to_dict(),
            "revision_history": _primitive(self.revision_history),
            "step": self.step,
            "retrieval_calls": self.retrieval_calls,
        }

    def state_hash(self) -> str:
        return stable_hash(self.state_payload())

    def to_dict(self) -> dict[str, Any]:
        return self.state_payload() | {
            "operation_history": _primitive(self.operation_history),
            "graph_schema_version": "dynamic-hypergraph-v1",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DynamicReasoningHypergraph":
        limits = GraphLimits(**value["limits"])
        graph = cls(question=str(value["question"]), limits=limits)
        graph.nodes = {
            str(node_id): _node_from_dict(node_value)
            for node_id, node_value in value.get("nodes", {}).items()
        }
        graph.hyperedges = {
            str(edge_id): _hyperedge_from_dict(edge_value)
            for edge_id, edge_value in value.get("hyperedges", {}).items()
        }
        graph.branches = {
            str(branch_id): _branch_from_dict(branch_value)
            for branch_id, branch_value in value.get("branches", {}).items()
        }
        graph.execution_graph = ExecutionDependencyGraph.from_dict(value.get("execution_graph", {}))
        graph.revision_history = [_revision_from_dict(row) for row in value.get("revision_history", [])]
        graph.operation_history = [_operation_from_dict(row) for row in value.get("operation_history", [])]
        graph.step = int(value.get("step", 0))
        graph.retrieval_calls = int(value.get("retrieval_calls", 0))
        graph.validate()
        return graph

    def validate(self) -> None:
        if len(self.nodes) > self.limits.max_graph_nodes:
            raise GraphBudgetExceeded("max_graph_nodes exceeded")
        if len(self.hyperedges) > self.limits.max_hyperedges:
            raise GraphBudgetExceeded("max_hyperedges exceeded")
        if len(self.operation_history) > self.limits.max_graph_operations:
            raise GraphBudgetExceeded("max_graph_operations exceeded")
        if self.retrieval_calls > self.limits.max_retrieval_calls:
            raise GraphBudgetExceeded("max_retrieval_calls exceeded")
        if len(self.revision_history) > self.limits.max_graph_revisions:
            raise GraphBudgetExceeded("max_graph_revisions exceeded")
        if len(self.active_branches()) > self.limits.max_active_branches:
            raise GraphBudgetExceeded("max_active_branches exceeded")
        for node_id, node in self.nodes.items():
            if node_id != node.node_id:
                raise GraphInvariantError(f"node key/id mismatch: {node_id}")
            if not node.provenance.source or not node.provenance.operation_id:
                raise GraphInvariantError(f"node {node_id} lacks provenance")
            if isinstance(node, EvidenceNode):
                if not node.document_id or not node.passage_id or not node.source_span:
                    raise GraphInvariantError(f"evidence {node_id} has invalid retrieval provenance")
                if not node.retrieval_query or not node.retriever_identity:
                    raise GraphInvariantError(f"evidence {node_id} lacks query/retriever identity")
            if isinstance(node, ClaimNode):
                if node.target_subgoal not in self.nodes or not isinstance(self.nodes[node.target_subgoal], SubgoalNode):
                    raise GraphInvariantError(f"claim {node_id} targets unknown subgoal")
                if not node.evidence_refs:
                    raise GraphInvariantError(f"claim {node_id} has no evidence")
                for evidence_id in node.evidence_refs:
                    if evidence_id not in self.nodes or not isinstance(self.nodes[evidence_id], EvidenceNode):
                        raise GraphInvariantError(f"claim {node_id} references invalid evidence {evidence_id}")
                for dependency_id in node.dependency_claim_ids:
                    if dependency_id not in self.nodes or not isinstance(self.nodes[dependency_id], ClaimNode):
                        raise GraphInvariantError(f"claim {node_id} references invalid dependency claim")
                for contradiction_id in node.contradiction_links:
                    if contradiction_id not in self.nodes or not isinstance(self.nodes[contradiction_id], ClaimNode):
                        raise GraphInvariantError(f"claim {node_id} has invalid contradiction link")
                if len(node.revision_history) > self.limits.max_revision_per_candidate:
                    raise GraphBudgetExceeded(f"claim {node_id} revision budget exceeded")
            if isinstance(node, AnswerNode):
                if not node.candidate_answer or not node.derivation_edge:
                    raise GraphInvariantError(f"answer {node_id} lacks derivation")
                if node.derivation_edge not in self.hyperedges:
                    raise GraphInvariantError(f"answer {node_id} derivation edge missing")
                if self.hyperedges[node.derivation_edge].target_node != node_id:
                    raise GraphInvariantError(f"answer {node_id} derivation edge target mismatch")
                if not node.supporting_claims or not node.supporting_evidence:
                    raise GraphInvariantError(f"answer {node_id} lacks supporting provenance")
                for claim_id in node.supporting_claims:
                    if claim_id not in self.nodes or not isinstance(self.nodes[claim_id], ClaimNode):
                        raise GraphInvariantError(f"answer {node_id} references invalid claim {claim_id}")
                for evidence_id in node.supporting_evidence:
                    if evidence_id not in self.nodes or not isinstance(self.nodes[evidence_id], EvidenceNode):
                        raise GraphInvariantError(f"answer {node_id} references invalid evidence {evidence_id}")
        for edge_id, edge in self.hyperedges.items():
            if edge_id != edge.edge_id:
                raise GraphInvariantError(f"hyperedge key/id mismatch: {edge_id}")
            if not edge.source_node_set or len(edge.source_node_set) != len(set(edge.source_node_set)):
                raise GraphInvariantError(f"hyperedge {edge_id} needs unique nonempty sources")
            if edge.target_node not in self.nodes:
                raise GraphInvariantError(f"hyperedge {edge_id} target missing")
            for source in edge.source_node_set:
                if source not in self.nodes:
                    raise GraphInvariantError(f"hyperedge {edge_id} source missing")
            for evidence_id in edge.supporting_evidence:
                if evidence_id not in self.nodes or not isinstance(self.nodes[evidence_id], EvidenceNode):
                    raise GraphInvariantError(f"hyperedge {edge_id} evidence missing")
            if not edge.provenance.operation_id:
                raise GraphInvariantError(f"hyperedge {edge_id} lacks provenance")
        derivation = nx.DiGraph()
        derivation.add_nodes_from(self.nodes)
        for edge in self.hyperedges.values():
            for source in edge.source_node_set:
                derivation.add_edge(source, edge.target_node)
        if not nx.is_directed_acyclic_graph(derivation):
            raise GraphInvariantError("derivation hyperedges must be acyclic within a graph version")
        self.execution_graph.validate()
        if self.execution_graph.depth() > self.limits.max_graph_depth:
            raise GraphBudgetExceeded("max_graph_depth exceeded")
        for branch_id, branch in self.branches.items():
            if branch_id != branch.branch_id:
                raise GraphInvariantError(f"branch key/id mismatch: {branch_id}")
            if branch.parent_branch_id is not None and branch.parent_branch_id not in self.branches:
                raise GraphInvariantError(f"branch {branch_id} parent missing")
            for subgoal_id, claim_id in branch.assignments.items():
                claim = self.node(claim_id, ClaimNode)
                if claim.target_subgoal != subgoal_id:
                    raise GraphInvariantError(f"branch {branch_id} assignment target mismatch")
                if claim.status in {CandidateStatus.ARCHIVED, CandidateStatus.INVALID}:
                    raise GraphInvariantError("archived/invalid candidate cannot remain active in a branch")
        counts: dict[tuple[str, str], int] = {}
        for claim in self.claims():
            if claim.status not in {CandidateStatus.ARCHIVED, CandidateStatus.INVALID}:
                key = (claim.target_subgoal, claim.branch_id)
                counts[key] = counts.get(key, 0) + 1
        if any(count > self.limits.max_candidates_per_subgoal for count in counts.values()):
            raise GraphBudgetExceeded("max_candidates_per_subgoal exceeded")
        operation_ids = [row.operation_id for row in self.operation_history]
        if len(operation_ids) != len(set(operation_ids)):
            raise GraphInvariantError("operation ids must be unique")
        for row in self.operation_history:
            if not row.graph_before_hash or not row.graph_after_hash:
                raise GraphInvariantError(f"operation {row.operation_id} is not auditable")

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


def _node_dict(node: GraphNode) -> dict[str, Any]:
    return {"kind": node.kind.value, **_primitive(node)}


def _provenance(value: dict[str, Any]) -> Provenance:
    return Provenance(**value)


def _revision_from_dict(value: dict[str, Any]) -> RevisionRecord:
    return RevisionRecord(**value)


def _score_from_dict(value: dict[str, Any]) -> CandidateScoreProfile:
    raw = VerificationSignals(**value.get("raw", {}))
    return CandidateScoreProfile(raw=raw, **{key: item for key, item in value.items() if key != "raw"})


def _node_from_dict(value: dict[str, Any]) -> GraphNode:
    data = dict(value)
    kind = NodeKind(data.pop("kind"))
    data["provenance"] = _provenance(data["provenance"])
    data["revision_history"] = [_revision_from_dict(row) for row in data.get("revision_history", [])]
    if kind == NodeKind.SUBGOAL:
        data["status"] = SubgoalStatus(data["status"])
        return SubgoalNode(**data)
    if kind == NodeKind.CLAIM:
        data["status"] = CandidateStatus(data["status"])
        data["score"] = _score_from_dict(data.get("score", {}))
        return ClaimNode(**data)
    if kind == NodeKind.EVIDENCE:
        return EvidenceNode(**data)
    data["status"] = AnswerStatus(data["status"])
    return AnswerNode(**data)


def _hyperedge_from_dict(value: dict[str, Any]) -> Hyperedge:
    data = dict(value)
    data["provenance"] = _provenance(data["provenance"])
    return Hyperedge(**data)


def _branch_from_dict(value: dict[str, Any]) -> BranchState:
    data = dict(value)
    data["status"] = BranchStatus(data["status"])
    data["history"] = [_revision_from_dict(row) for row in data.get("history", [])]
    return BranchState(**data)


def _operation_from_dict(value: dict[str, Any]) -> AppliedOperation:
    data = dict(value)
    data["operation_type"] = OperationType(data["operation_type"])
    return AppliedOperation(**data)
