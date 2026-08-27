from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..utils import normalize_text, stable_hash
from .graph import (
    AppliedOperation,
    AnswerNode,
    AnswerStatus,
    BranchState,
    BranchStatus,
    CandidateScoreProfile,
    CandidateStatus,
    ClaimNode,
    DynamicReasoningHypergraph,
    EvidenceNode,
    GraphInvariantError,
    GraphOperation,
    Hyperedge,
    OperationType,
    Provenance,
    RevisionRecord,
    SubgoalNode,
    SubgoalStatus,
    VerificationSignals,
)


class GraphController:
    """Transactional deterministic validator for LLM or policy proposals."""

    def apply(self, graph: DynamicReasoningHypergraph, operation: GraphOperation) -> DynamicReasoningHypergraph:
        if len(graph.operation_history) >= graph.limits.max_graph_operations:
            raise GraphInvariantError("operation budget exhausted")
        if any(row.operation_id == operation.operation_id for row in graph.operation_history):
            raise GraphInvariantError(f"duplicate operation id {operation.operation_id}")
        graph.validate()
        before = graph.state_hash()
        updated = DynamicReasoningHypergraph.from_dict(graph.to_dict())
        updated.step += 1
        changes = self._apply_mutation(updated, operation)
        updated.validate()
        after = updated.state_hash()
        updated.operation_history.append(AppliedOperation(
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            step=updated.step,
            branch_id=operation.branch_id,
            graph_before_hash=before,
            graph_after_hash=after,
            created_nodes=changes["created_nodes"],
            updated_nodes=changes["updated_nodes"],
            pruned_nodes=changes["pruned_nodes"],
            created_hyperedges=changes["created_hyperedges"],
            reason=operation.reason,
            payload_digest=stable_hash(operation.payload),
        ))
        updated.validate()
        return updated

    def _apply_mutation(self, graph: DynamicReasoningHypergraph, operation: GraphOperation) -> dict[str, list[str]]:
        changes = {
            "created_nodes": [], "updated_nodes": [], "pruned_nodes": [],
            "created_hyperedges": [],
        }
        handlers = {
            OperationType.EXPAND: self._expand,
            OperationType.BRANCH: self._branch,
            OperationType.RETRIEVE: self._retrieve,
            OperationType.VERIFY: self._verify,
            OperationType.MERGE: self._merge,
            OperationType.PRUNE: self._prune,
            OperationType.COMMIT: self._commit,
            OperationType.REVISE: self._revise,
        }
        handlers[operation.operation_type](graph, operation, changes)
        return changes

    @staticmethod
    def _provenance(operation: GraphOperation, step: int, **extra: Any) -> Provenance:
        return Provenance(
            source=operation.proposed_by,
            operation_id=operation.operation_id,
            created_at_step=step,
            source_node_ids=list(operation.source_ids),
            metadata={"reason": operation.reason, **extra},
        )

    def _expand(self, graph, operation, changes) -> None:
        rows = operation.payload.get("subgoals", [])
        if not isinstance(rows, list) or not rows:
            raise GraphInvariantError("EXPAND requires nonempty subgoals")
        pending_ids = {str(row.get("node_id", "")) for row in rows if isinstance(row, dict)}
        if "" in pending_ids or len(pending_ids) != len(rows):
            raise GraphInvariantError("EXPAND subgoal IDs must be unique and nonempty")
        known = set(graph.execution_graph.dependencies) | pending_ids
        for row in rows:
            dependencies = [str(value) for value in row.get("dependencies", [])]
            if any(value not in known for value in dependencies):
                raise GraphInvariantError("EXPAND references unknown execution dependency")
            node_id = str(row["node_id"])
            if node_id in graph.nodes:
                raise GraphInvariantError(f"node already exists: {node_id}")
            bindings = {str(key): str(value) for key, value in row.get("variable_bindings", {}).items()}
            if any(source not in dependencies for source in bindings.values()):
                raise GraphInvariantError("variable binding source must be an execution dependency")
            node = SubgoalNode(
                node_id=node_id,
                question_template=str(row.get("question_template", "")).strip(),
                instantiated_question=str(row.get("instantiated_question", "")).strip(),
                dependencies=dependencies,
                variable_bindings=bindings,
                answer_type=str(row.get("answer_type", "entity")),
                terminal=bool(row.get("terminal", False)),
                status=SubgoalStatus(row.get("status", "unresolved")),
                confidence=_unit(row.get("confidence", 0.0)),
                uncertainty=_unit(row.get("uncertainty", 1.0)),
                created_at_step=graph.step,
                provenance=self._provenance(operation, graph.step),
            )
            if not node.question_template:
                raise GraphInvariantError("subgoal question is required")
            graph.nodes[node_id] = node
            changes["created_nodes"].append(node_id)
        # Install the whole proposed dependency batch atomically.  Valid plans need
        # not be serialized in topological order; the controller validates the
        # complete execution DAG after every referenced node is materialized.
        for row in rows:
            graph.execution_graph.dependencies[str(row["node_id"])] = list(dict.fromkeys(
                str(value) for value in row.get("dependencies", [])
            ))
        graph.execution_graph.validate()
        attach_target = str(operation.payload.get("attach_target", ""))
        attach_node = str(operation.payload.get("attach_node", ""))
        if attach_target or attach_node:
            if attach_target not in graph.execution_graph.dependencies or attach_node not in pending_ids:
                raise GraphInvariantError("EXPAND attachment must connect a new subgoal to a known target")
            target = graph.node(attach_target, SubgoalNode)
            # If the target already binds variables, those source dependencies
            # carry executable semantics and cannot be discarded merely because
            # an editor inserted an additional prerequisite. A binding-free root
            # may be safely replaced by the newly proposed final relation.
            if target.variable_bindings:
                dependencies = list(dict.fromkeys([attach_node] + target.dependencies))
            else:
                dependencies = [attach_node]
            graph.execution_graph.replace_dependencies(attach_target, dependencies)
            target.dependencies = dependencies
            target.status = SubgoalStatus.UNRESOLVED
            target.uncertainty = max(target.uncertainty, 0.8)
            changes["updated_nodes"].append(attach_target)

    def _retrieve(self, graph, operation, changes) -> None:
        if graph.retrieval_calls >= graph.limits.max_retrieval_calls:
            raise GraphInvariantError("retrieval budget exhausted")
        graph.node(operation.target_id, SubgoalNode)
        rows = operation.payload.get("evidence", [])
        if not isinstance(rows, list):
            raise GraphInvariantError("RETRIEVE evidence must be a list")
        for row in rows:
            node_id = str(row.get("node_id", ""))
            if not node_id or node_id in graph.nodes:
                raise GraphInvariantError(f"invalid or duplicate evidence node {node_id}")
            query = str(row.get("retrieval_query", "")).strip()
            retriever = str(row.get("retriever_identity", "")).strip()
            provenance = self._provenance(operation, graph.step)
            provenance.retrieval_query = query
            provenance.retriever_identity = retriever
            node = EvidenceNode(
                node_id=node_id,
                document_id=str(row.get("document_id", "")),
                passage_id=str(row.get("passage_id", row.get("document_id", ""))),
                title=str(row.get("title", "")),
                source_span=str(row.get("source_span", "")),
                retrieval_rank=int(row.get("retrieval_rank", 0)),
                retrieval_score=float(row.get("retrieval_score", 0.0)),
                retrieval_query=query,
                retriever_identity=retriever,
                branch_id=operation.branch_id,
                target_subgoal=operation.target_id,
                created_at_step=graph.step,
                provenance=provenance,
            )
            graph.nodes[node_id] = node
            changes["created_nodes"].append(node_id)
        new_evidence = [graph.node(node_id, EvidenceNode) for node_id in changes["created_nodes"]]
        # A disambiguation retrieval is useful only if preserved candidates can be
        # reconsidered. Attach newly grounding evidence and mark candidates for a
        # fresh independent raw-score pass; no cross-candidate score is reused.
        for claim in graph.claims(operation.target_id, operation.branch_id):
            if claim.status in {CandidateStatus.ARCHIVED, CandidateStatus.INVALID, CandidateStatus.COMMITTED}:
                continue
            grounded = [
                node.node_id for node in new_evidence
                if normalize_text(claim.value) in normalize_text(node.source_span)
            ]
            if grounded:
                claim.evidence_refs = list(dict.fromkeys(claim.evidence_refs + grounded))
                claim.provenance.evidence_ids = list(claim.evidence_refs)
            claim.status = CandidateStatus.PROPOSED
            changes["updated_nodes"].append(claim.node_id)
        graph.retrieval_calls += 1
        subgoal = graph.node(operation.target_id, SubgoalNode)
        subgoal.status = SubgoalStatus.ACTIVE
        subgoal.instantiated_question = str(operation.payload.get("query", subgoal.instantiated_question))
        changes["updated_nodes"].append(subgoal.node_id)

    def _branch(self, graph, operation, changes) -> None:
        mode = str(operation.payload.get("mode", "candidates"))
        if mode == "candidates":
            self._create_candidates(graph, operation, changes)
            return
        if mode != "assignments":
            raise GraphInvariantError(f"unknown BRANCH mode {mode}")
        parent = graph.branches.get(operation.branch_id)
        if parent is None or parent.status != BranchStatus.ACTIVE:
            raise GraphInvariantError("BRANCH parent must be active")
        candidate_ids = [str(value) for value in operation.payload.get("candidate_ids", [])]
        if len(candidate_ids) < 2:
            raise GraphInvariantError("assignment BRANCH requires at least two candidates")
        if len(graph.active_branches()) - 1 + len(candidate_ids) > graph.limits.max_active_branches:
            raise GraphInvariantError("branch budget exceeded")
        subgoal_id = operation.target_id
        weights = []
        for candidate_id in candidate_ids:
            claim = graph.node(candidate_id, ClaimNode)
            if claim.target_subgoal != subgoal_id or claim.status not in {
                CandidateStatus.SCORED, CandidateStatus.RETAINED, CandidateStatus.REOPENED,
            }:
                raise GraphInvariantError("BRANCH candidate is not viable for target subgoal")
            weights.append(claim.score.relative_weight)
        parent.status = BranchStatus.ARCHIVED
        for index, (candidate_id, weight) in enumerate(zip(candidate_ids, weights), start=1):
            branch_id = str(operation.payload.get("branch_ids", [])[index - 1]) if index <= len(operation.payload.get("branch_ids", [])) else f"{operation.branch_id}.b{graph.step}.{index}"
            if branch_id in graph.branches:
                raise GraphInvariantError(f"duplicate branch id {branch_id}")
            graph.branches[branch_id] = BranchState(
                branch_id=branch_id,
                parent_branch_id=parent.branch_id,
                assignments={**parent.assignments, subgoal_id: candidate_id},
                completed_subgoals=list(dict.fromkeys(parent.completed_subgoals + [subgoal_id])),
                score=max(0.0, parent.score * max(weight, 1e-6)),
                status=BranchStatus.ACTIVE,
                created_at_step=graph.step,
            )
        subgoal = graph.node(subgoal_id, SubgoalNode)
        subgoal.status = SubgoalStatus.PARTIAL
        subgoal.uncertainty = max((graph.node(value, ClaimNode).score.set_entropy for value in candidate_ids), default=1.0)
        changes["updated_nodes"].append(subgoal_id)

    def _create_candidates(self, graph, operation, changes) -> None:
        graph.node(operation.target_id, SubgoalNode)
        rows = operation.payload.get("candidates", [])
        if not isinstance(rows, list) or not rows:
            raise GraphInvariantError("candidate BRANCH requires candidates")
        existing = [
            claim for claim in graph.claims(operation.target_id, operation.branch_id)
            if claim.status not in {CandidateStatus.ARCHIVED, CandidateStatus.INVALID}
        ]
        if len(existing) + len(rows) > graph.limits.max_candidates_per_subgoal:
            raise GraphInvariantError("candidate cap exceeded")
        for row in rows:
            node_id = str(row.get("node_id", ""))
            if not node_id or node_id in graph.nodes:
                raise GraphInvariantError(f"invalid or duplicate claim node {node_id}")
            evidence_refs = [str(value) for value in row.get("evidence_refs", [])]
            for evidence_id in evidence_refs:
                evidence = graph.node(evidence_id, EvidenceNode)
                if evidence.target_subgoal != operation.target_id:
                    raise GraphInvariantError("candidate evidence belongs to another subgoal")
            dependency_ids = [str(value) for value in row.get("dependency_claim_ids", [])]
            for dependency_id in dependency_ids:
                graph.node(dependency_id, ClaimNode)
            node = ClaimNode(
                node_id=node_id,
                subject=str(row.get("subject", "")).strip(),
                relation=str(row.get("relation", "")).strip(),
                value=str(row.get("value", "")).strip(),
                answer_type=str(row.get("answer_type", "entity")),
                target_subgoal=operation.target_id,
                branch_id=operation.branch_id,
                evidence_refs=evidence_refs,
                dependency_claim_ids=dependency_ids,
                score=CandidateScoreProfile(raw=VerificationSignals(
                    raw_model_confidence=_unit(row.get("extraction_confidence", 0.0)),
                )),
                status=CandidateStatus.PROPOSED,
                contradiction_links=[],
                created_at_step=graph.step,
                provenance=self._provenance(operation, graph.step, source_spans=row.get("source_spans", [])),
            )
            if not node.value:
                raise GraphInvariantError("candidate value is required")
            node.provenance.evidence_ids = evidence_refs
            graph.nodes[node_id] = node
            changes["created_nodes"].append(node_id)

    def _verify(self, graph, operation, changes) -> None:
        scores = operation.payload.get("scores", {})
        if not isinstance(scores, dict) or not scores:
            raise GraphInvariantError("VERIFY requires candidate score mapping")
        for claim_id, row in scores.items():
            claim = graph.node(str(claim_id), ClaimNode)
            if claim.target_subgoal != operation.target_id:
                raise GraphInvariantError("VERIFY candidate target mismatch")
            raw = VerificationSignals(
                grounding=_unit(row.get("grounding")),
                entailment=_unit(row.get("entailment")),
                type_match=_unit(row.get("type_match")),
                dependency_consistency=_unit(row.get("dependency_consistency")),
                retrieval_support=_unit(row.get("retrieval_support")),
                contradiction_risk=_unit(row.get("contradiction_risk")),
                raw_model_confidence=_unit(row.get("raw_model_confidence")),
                relation_target_alignment=_unit(row.get("relation_target_alignment")),
                subject_binding_coverage=_unit(row.get("subject_binding_coverage")),
                dependency_binding_coverage=_unit(row.get("dependency_binding_coverage")),
                qualifier_coverage=_unit(row.get("qualifier_coverage")),
                output_slot_coverage=_unit(row.get("output_slot_coverage")),
                full_subgoal_coverage=_unit(row.get("full_subgoal_coverage")),
                reasons=[str(value) for value in row.get("reasons", [])][:5],
            )
            claim.score = CandidateScoreProfile(
                raw=raw,
                absolute_support=_unit(row.get("absolute_support")),
                relative_weight=_unit(row.get("relative_weight")),
                set_entropy=_unit(row.get("set_entropy")),
                evidence_gap=_unit(row.get("evidence_gap", 1.0)),
            )
            if isinstance(row.get("scoring_audit"), dict):
                claim.provenance.metadata["verification_scoring_audit"] = row["scoring_audit"]
            claim.status = CandidateStatus(row.get("status", "scored"))
            contradiction_ids = [str(value) for value in row.get("contradiction_links", [])]
            for contradiction_id in contradiction_ids:
                graph.node(contradiction_id, ClaimNode)
            claim.contradiction_links = list(dict.fromkeys(claim.contradiction_links + contradiction_ids))
            changes["updated_nodes"].append(claim.node_id)
        subgoal = graph.node(operation.target_id, SubgoalNode)
        candidates = [graph.node(str(claim_id), ClaimNode) for claim_id in scores]
        subgoal.confidence = max((claim.score.absolute_support for claim in candidates), default=0.0)
        subgoal.uncertainty = max((claim.score.set_entropy for claim in candidates), default=1.0)
        subgoal.status = SubgoalStatus.PARTIAL
        changes["updated_nodes"].append(subgoal.node_id)

    def _merge(self, graph, operation, changes) -> None:
        keep_id = str(operation.payload.get("keep_id", ""))
        merge_ids = [str(value) for value in operation.payload.get("merge_ids", [])]
        keep = graph.node(keep_id, ClaimNode)
        if not merge_ids:
            raise GraphInvariantError("MERGE requires merge_ids")
        before = asdict(keep)
        for merge_id in merge_ids:
            other = graph.node(merge_id, ClaimNode)
            if other.target_subgoal != keep.target_subgoal:
                raise GraphInvariantError("cannot merge candidates from different subgoals")
            keep.evidence_refs = list(dict.fromkeys(keep.evidence_refs + other.evidence_refs))
            keep.dependency_claim_ids = list(dict.fromkeys(keep.dependency_claim_ids + other.dependency_claim_ids))
            keep.provenance.source_node_ids = list(dict.fromkeys(keep.provenance.source_node_ids + [merge_id]))
            other.status = CandidateStatus.ARCHIVED
            changes["pruned_nodes"].append(merge_id)
            for branch in graph.branches.values():
                for subgoal_id, candidate_id in list(branch.assignments.items()):
                    if candidate_id == merge_id:
                        branch.assignments[subgoal_id] = keep_id
        keep.revision_history.append(RevisionRecord(
            graph.step, operation.operation_id, operation.reason, before, asdict(keep),
        ))
        keep.status = CandidateStatus.REVISED
        changes["updated_nodes"].append(keep_id)

    def _prune(self, graph, operation, changes) -> None:
        candidate_ids = [str(value) for value in operation.payload.get("candidate_ids", [])]
        if not candidate_ids:
            raise GraphInvariantError("PRUNE requires candidate_ids")
        for candidate_id in candidate_ids:
            claim = graph.node(candidate_id, ClaimNode)
            before = {"status": claim.status.value}
            claim.status = CandidateStatus.ARCHIVED
            claim.revision_history.append(RevisionRecord(
                graph.step, operation.operation_id, operation.reason,
                before, {"status": claim.status.value},
            ))
            for branch in graph.branches.values():
                for subgoal_id, assigned in list(branch.assignments.items()):
                    if assigned == candidate_id:
                        del branch.assignments[subgoal_id]
                        branch.completed_subgoals = [value for value in branch.completed_subgoals if value != subgoal_id]
            changes["pruned_nodes"].append(candidate_id)

    def _commit(self, graph, operation, changes) -> None:
        if operation.payload.get("mode") == "answer":
            self._commit_answer(graph, operation, changes)
            return
        candidate_id = str(operation.payload.get("candidate_id", ""))
        claim = graph.node(candidate_id, ClaimNode)
        if claim.target_subgoal != operation.target_id:
            raise GraphInvariantError("COMMIT candidate target mismatch")
        if claim.status not in {CandidateStatus.SCORED, CandidateStatus.RETAINED, CandidateStatus.REOPENED, CandidateStatus.REVISED}:
            raise GraphInvariantError("COMMIT candidate is not viable")
        branch = graph.branches.get(operation.branch_id)
        if branch is None or branch.status != BranchStatus.ACTIVE:
            raise GraphInvariantError("COMMIT branch must be active")
        branch.assignments[claim.target_subgoal] = claim.node_id
        branch.completed_subgoals = list(dict.fromkeys(branch.completed_subgoals + [claim.target_subgoal]))
        claim.status = CandidateStatus.COMMITTED
        subgoal = graph.node(claim.target_subgoal, SubgoalNode)
        subgoal.status = SubgoalStatus.RESOLVED
        subgoal.confidence = claim.score.absolute_support
        subgoal.uncertainty = claim.score.set_entropy
        changes["updated_nodes"].extend([claim.node_id, subgoal.node_id])

    def _commit_answer(self, graph, operation, changes) -> None:
        row = operation.payload.get("answer", {})
        if not isinstance(row, dict):
            raise GraphInvariantError("answer COMMIT requires answer payload")
        node_id = str(row.get("node_id", ""))
        edge_id = str(row.get("derivation_edge", ""))
        if not node_id or node_id in graph.nodes or not edge_id or edge_id in graph.hyperedges:
            raise GraphInvariantError("answer/derivation IDs must be new and nonempty")
        claim_ids = [str(value) for value in row.get("supporting_claims", [])]
        evidence_ids = [str(value) for value in row.get("supporting_evidence", [])]
        if not claim_ids or not evidence_ids:
            raise GraphInvariantError("answer requires claims and evidence")
        for claim_id in claim_ids:
            claim = graph.node(claim_id, ClaimNode)
            if claim.status not in {
                CandidateStatus.COMMITTED, CandidateStatus.SCORED,
                CandidateStatus.RETAINED, CandidateStatus.REVISED,
            }:
                raise GraphInvariantError("answer source claim is not viable")
        for evidence_id in evidence_ids:
            graph.node(evidence_id, EvidenceNode)
        provenance = self._provenance(operation, graph.step)
        provenance.evidence_ids = evidence_ids
        answer = AnswerNode(
            node_id=node_id,
            candidate_answer=str(row.get("candidate_answer", "")).strip(),
            answer_type=str(row.get("answer_type", "entity")),
            supporting_claims=claim_ids,
            supporting_evidence=evidence_ids,
            derivation_edge=edge_id,
            branch_id=operation.branch_id,
            confidence=_unit(row.get("confidence")),
            answer_type_consistency=_unit(row.get("answer_type_consistency", 1.0)),
            contradiction_risk=_unit(row.get("contradiction_risk", 0.0)),
            status=AnswerStatus(row.get("status", "accepted")),
            created_at_step=graph.step,
            provenance=provenance,
        )
        edge = Hyperedge(
            edge_id=edge_id,
            source_node_set=claim_ids,
            target_node=node_id,
            inference_type=str(row.get("inference_type", "terminal_derivation")),
            confidence=answer.confidence,
            supporting_evidence=evidence_ids,
            creation_reason=operation.reason,
            created_by_operation=operation.operation_id,
            created_at_step=graph.step,
            provenance=provenance,
        )
        graph.nodes[node_id] = answer
        graph.hyperedges[edge_id] = edge
        branch = graph.branches.get(operation.branch_id)
        if branch is None:
            raise GraphInvariantError("answer COMMIT branch missing")
        branch.status = BranchStatus.COMPLETED
        changes["created_nodes"].append(node_id)
        changes["created_hyperedges"].append(edge_id)

    def _revise(self, graph, operation, changes) -> None:
        if len(graph.revision_history) >= graph.limits.max_graph_revisions:
            raise GraphInvariantError("revision budget exhausted")
        action = str(operation.payload.get("action", ""))
        branch = graph.branches.get(operation.branch_id)
        if branch is None:
            raise GraphInvariantError("REVISE branch missing")
        if action == "reopen":
            claim_id = str(operation.payload.get("claim_id", ""))
            claim = graph.node(claim_id, ClaimNode)
            if claim.status != CandidateStatus.COMMITTED:
                raise GraphInvariantError("only committed claims can be reopened")
            before = {"status": claim.status.value, "assignment": branch.assignments.get(claim.target_subgoal)}
            claim.status = CandidateStatus.REOPENED
            if branch.assignments.get(claim.target_subgoal) == claim_id:
                del branch.assignments[claim.target_subgoal]
            branch.completed_subgoals = [value for value in branch.completed_subgoals if value != claim.target_subgoal]
            branch.revision_count += 1
            branch.last_revision_step = graph.step
            subgoal = graph.node(claim.target_subgoal, SubgoalNode)
            subgoal.status = SubgoalStatus.PARTIAL
            record = RevisionRecord(
                graph.step, operation.operation_id, operation.reason, before,
                {"status": claim.status.value, "assignment": None},
            )
            claim.revision_history.append(record)
            branch.history.append(record)
            graph.revision_history.append(record)
            changes["updated_nodes"].extend([claim.node_id, subgoal.node_id])
            return
        if action == "replace_assignment":
            subgoal_id = operation.target_id
            new_id = str(operation.payload.get("new_candidate_id", ""))
            new_claim = graph.node(new_id, ClaimNode)
            if new_claim.target_subgoal != subgoal_id:
                raise GraphInvariantError("replacement candidate target mismatch")
            old_id = branch.assignments.get(subgoal_id)
            before = {"candidate_id": old_id}
            if old_id:
                old_claim = graph.node(old_id, ClaimNode)
                old_claim.status = CandidateStatus.REOPENED
                changes["updated_nodes"].append(old_id)
            new_claim.status = CandidateStatus.COMMITTED
            branch.assignments[subgoal_id] = new_id
            branch.completed_subgoals = list(dict.fromkeys(branch.completed_subgoals + [subgoal_id]))
            branch.revision_count += 1
            branch.last_revision_step = graph.step
            record = RevisionRecord(
                graph.step, operation.operation_id, operation.reason, before,
                {"candidate_id": new_id},
            )
            branch.history.append(record)
            graph.revision_history.append(record)
            new_claim.revision_history.append(record)
            changes["updated_nodes"].append(new_id)
            return
        if action == "dependencies":
            subgoal = graph.node(operation.target_id, SubgoalNode)
            dependencies = [str(value) for value in operation.payload.get("dependencies", [])]
            before = {"dependencies": list(subgoal.dependencies), "variable_bindings": dict(subgoal.variable_bindings)}
            graph.execution_graph.replace_dependencies(subgoal.node_id, dependencies)
            subgoal.dependencies = dependencies
            subgoal.variable_bindings = {
                str(key): str(value) for key, value in operation.payload.get("variable_bindings", {}).items()
            }
            record = RevisionRecord(
                graph.step, operation.operation_id, operation.reason, before,
                {"dependencies": dependencies, "variable_bindings": dict(subgoal.variable_bindings)},
            )
            subgoal.revision_history.append(record)
            graph.revision_history.append(record)
            changes["updated_nodes"].append(subgoal.node_id)
            return
        raise GraphInvariantError(f"unknown REVISE action {action}")


def _unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
