from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from ..dynamic.controller import GraphController
from ..dynamic.graph import (
    AppliedOperation,
    AnswerNode,
    AnswerStatus,
    CandidateScoreProfile,
    CandidateStatus,
    ClaimNode,
    GraphInvariantError,
    GraphOperation,
    Hyperedge,
    OperationType,
    RevisionRecord,
    SubgoalNode,
    SubgoalStatus,
    VerificationSignals,
)
from ..utils import normalize_text, stable_hash
from .belief import GraphBeliefUpdater
from .config import DynamicV2ResearchConfig
from .diffusion import TypedDirectionalDiffusion
from .graph import (
    AllocationRecord,
    ClaimSemantics,
    DynamicReasoningHypergraphV2,
    JoinAttemptRecord,
    OperationFeedbackStats,
    OperationOutcomeRecord,
    SupersessionRecord,
    TerminationRecord,
)
from .allocator import (
    feedback_key,
    operation_family,
    operation_region_key,
    summarize_operation_region,
)


class V2GraphController(GraphController):
    """Only mutation authority for V2 graph state.

    The controller executes a transaction on a deep copy, performs a local belief
    update and typed diffusion, seals the resulting state, and only then returns it.
    Invalid operations never modify the input graph.
    """

    def __init__(self, config: DynamicV2ResearchConfig) -> None:
        self.config = config
        self.belief_updater = GraphBeliefUpdater()
        self.diffusion = TypedDirectionalDiffusion(config)

    def apply(
        self, graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
    ) -> DynamicReasoningHypergraphV2:
        if not isinstance(graph, DynamicReasoningHypergraphV2):
            raise TypeError("V2GraphController requires DynamicReasoningHypergraphV2")
        if len(graph.operation_history) >= graph.limits.max_graph_operations:
            raise GraphInvariantError("operation budget exhausted")
        if any(row.operation_id == operation.operation_id for row in graph.operation_history):
            raise GraphInvariantError(f"duplicate operation id {operation.operation_id}")
        graph.validate()
        before = graph.state_hash()
        updated = deepcopy(graph)
        updated.controller_state_hash = ""
        updated.step += 1
        changes = self._apply_mutation(updated, operation)
        self._record_allocation(updated, operation)
        seeds = list(dict.fromkeys(
            changes["created_nodes"] + changes["updated_nodes"] + changes["pruned_nodes"]
            + [operation.target_id] + operation.source_ids
        ))
        belief_changed = self.belief_updater.recompute(updated, seeds, operation.reason)
        changes["updated_nodes"] = list(dict.fromkeys(changes["updated_nodes"] + belief_changed))
        self.diffusion.propagate(
            updated, seeds,
            f"diffusion_{updated.step:04d}_{operation.operation_id}",
        )
        after = updated.state_hash()
        updated.operation_history.append(AppliedOperation(
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            step=updated.step,
            branch_id=operation.branch_id,
            graph_before_hash=before,
            graph_after_hash=after,
            created_nodes=list(dict.fromkeys(changes["created_nodes"])),
            updated_nodes=list(dict.fromkeys(changes["updated_nodes"])),
            pruned_nodes=list(dict.fromkeys(changes["pruned_nodes"])),
            created_hyperedges=list(dict.fromkeys(changes["created_hyperedges"])),
            reason=operation.reason,
            payload_digest=stable_hash(operation.payload),
        ))
        updated.seal_controller_state()
        updated.validate()
        return updated

    def terminate(self, graph: DynamicReasoningHypergraphV2, decision: Any, budget: Any):
        """Append a terminal outcome through the same controller-owned seal."""
        graph.validate()
        updated = deepcopy(graph)
        updated.controller_state_hash = ""
        updated.termination_history.append(TerminationRecord(
            step=updated.step,
            outcome=decision.outcome,
            best_predicted_evc=float(decision.best_predicted_evc),
            answer_node_id=decision.answer_node_id,
            reason=str(decision.reason),
            remaining_budget={
                "llm_calls": max(0, budget.max_llm_calls - budget.usage.llm_calls),
                "tokens": max(0, budget.max_total_tokens - budget.usage.total_tokens),
                "retrieval_calls": max(0, updated.limits.max_retrieval_calls - updated.retrieval_calls),
                "graph_operations": max(0, updated.limits.max_graph_operations - len(updated.operation_history)),
            },
        ))
        updated.seal_controller_state()
        updated.validate()
        return updated

    def reconcile_allocation(
        self,
        graph: DynamicReasoningHypergraphV2,
        packet: Any,
        actual_cost: dict[str, float],
        completed: bool,
        failure_reason: str = "",
        outcome_metadata: dict[str, Any] | None = None,
    ) -> DynamicReasoningHypergraphV2:
        """Record the outcome of every selected allocation, including failures.

        The operation transaction records successful selections eagerly.  This
        controller-owned reconciliation replaces its estimated cost with measured
        budget deltas, or creates a ledger row when no graph operation committed.
        """
        graph.validate()
        updated = deepcopy(graph)
        updated.controller_state_hash = ""
        existing = next((
            row for row in updated.allocation_history
            if row.allocation_id == packet.allocation_id
        ), None)
        measured = {str(key): float(value) for key, value in actual_cost.items()}
        trace = packet.trace()
        if existing is None:
            updated.allocation_history.append(AllocationRecord(
                allocation_id=packet.allocation_id,
                operation_id=packet.operation.operation_id,
                step=updated.step,
                target_region=[str(value) for value in trace["target_region"]],
                predicted_evc=float(trace["predicted_evc"]),
                evc_components_raw={
                    str(key): float(value) for key, value in trace["evc_components_raw"].items()
                },
                evc_components_normalized={
                    str(key): float(value) for key, value in trace["evc_components_normalized"].items()
                },
                requested_budget={
                    str(key): int(value) for key, value in trace["requested_budget"].items()
                },
                remaining_global_budget={
                    str(key): int(value) for key, value in trace["remaining_global_budget"].items()
                },
                actual_cost=measured,
                allocator_mode=str(trace.get("allocator_mode", "adaptive_evc")),
                pre_state_summary={
                    str(key): float(value)
                    for key, value in trace.get("pre_state_summary", {}).items()
                },
                feedback_prior={
                    str(key): float(value)
                    for key, value in trace.get("feedback_prior", {}).items()
                },
                selected=True,
                completed=bool(completed),
                failure_reason=str(failure_reason),
            ))
        else:
            existing.actual_cost = measured
            existing.completed = bool(completed)
            existing.failure_reason = str(failure_reason)
        existing = existing or updated.allocation_history[-1]
        self._reconcile_outcome(
            updated, existing, packet, measured, bool(completed), str(failure_reason),
        )
        self._reconcile_join_attempt(
            updated, packet, measured, bool(completed), str(failure_reason),
            outcome_metadata or {},
        )
        updated.seal_controller_state()
        updated.validate()
        return updated

    def _reconcile_outcome(
        self, graph, allocation, packet, measured, progressed, failure_reason,
    ) -> None:
        operation = packet.operation
        pre = {
            str(key): float(value)
            for key, value in packet.pre_state_summary.items()
        }
        post = summarize_operation_region(graph, operation)
        delta = {
            key: float(post.get(key, 0.0) - pre.get(key, 0.0))
            for key in sorted(set(pre) | set(post))
        }
        raw = {
            "uncertainty_reduction": pre.get("uncertainty", 1.0) - post.get("uncertainty", 1.0),
            "support_gain": post.get("absolute_support", 0.0) - pre.get("absolute_support", 0.0),
            "evidence_gap_reduction": pre.get("evidence_gap", 1.0) - post.get("evidence_gap", 1.0),
            "entropy_reduction": pre.get("entropy", 1.0) - post.get("entropy", 1.0),
            "dependency_unlock_gain": post.get("dependency_unlock", 0.0) - pre.get("dependency_unlock", 0.0),
            "evidence_novelty": post.get("evidence_count", 0.0) - pre.get("evidence_count", 0.0),
            "answer_chain_progress": post.get("answer_chain_progress", 0.0) - pre.get("answer_chain_progress", 0.0),
            "contradiction_resolution": (
                pre.get("contradiction_pressure", 0.0) - post.get("contradiction_pressure", 0.0)
            ),
            "cost": self._normalized_actual_cost(packet, measured),
        }
        normalized = {
            key: _unit_positive(value) for key, value in raw.items()
        }
        weights = {
            "uncertainty_reduction": self.config.actual_utility_weight_uncertainty,
            "support_gain": self.config.actual_utility_weight_support,
            "evidence_gap_reduction": self.config.actual_utility_weight_evidence_gap,
            "entropy_reduction": self.config.actual_utility_weight_entropy,
            "dependency_unlock_gain": self.config.actual_utility_weight_unlock,
            "evidence_novelty": self.config.actual_utility_weight_novelty,
            "answer_chain_progress": self.config.actual_utility_weight_chain_progress,
            "contradiction_resolution": self.config.actual_utility_weight_contradiction_resolution,
        }
        benefit_weight = sum(float(value) for value in weights.values())
        benefit = sum(
            float(weights[key]) * normalized[key] for key in weights
        )
        cost_weight = float(self.config.actual_utility_weight_cost)
        denominator = max(1e-12, benefit_weight + cost_weight)
        utility = max(-1.0, min(1.0, (
            benefit - cost_weight * normalized["cost"]
        ) / denominator))
        family = packet.operation_family or operation_family(operation)
        region = packet.region_key or operation_region_key(operation)
        before = self._feedback_snapshot(graph, family, region)
        for key in (feedback_key(family, region), feedback_key(family)):
            self._update_feedback(graph, key, utility, normalized["cost"], progressed)
        after = self._feedback_snapshot(graph, family, region)
        allocation.post_state_summary = post
        allocation.state_delta = delta
        allocation.actual_utility_components_raw = raw
        allocation.actual_utility_components_normalized = normalized
        allocation.actual_utility = utility
        allocation.feedback_applied = True
        graph.operation_outcome_history.append(OperationOutcomeRecord(
            outcome_id=f"outcome_{packet.allocation_id}",
            allocation_id=packet.allocation_id,
            operation_id=operation.operation_id,
            step=graph.step,
            operation_family=family,
            region_key=region,
            pre_state_summary=pre,
            post_state_summary=post,
            state_delta=delta,
            actual_utility_components_raw=raw,
            actual_utility_components_normalized=normalized,
            actual_utility=utility,
            actual_cost=measured,
            progressed=progressed,
            failure_reason=failure_reason,
            statistics_before=before,
            statistics_after=after,
        ))

    def _update_feedback(self, graph, key, utility, normalized_cost, progressed) -> None:
        stats = graph.operation_feedback.setdefault(key, OperationFeedbackStats())
        stats.observations += 1
        stats.successes += int(progressed)
        stats.no_ops += int(not progressed)
        stats.cumulative_utility += (float(utility) + 1.0) / 2.0
        stats.cumulative_cost += float(normalized_cost)
        prior = float(self.config.outcome_feedback_prior_strength)
        stats.posterior_value = (
            prior * 0.5 + stats.cumulative_utility
        ) / (prior + stats.observations)
        stats.posterior_success = (
            prior * 0.5 + stats.successes
        ) / (prior + stats.observations)
        if progressed:
            stats.consecutive_failures = 0
        else:
            stats.consecutive_failures += 1
            if stats.consecutive_failures >= self.config.outcome_feedback_cooldown_failures:
                stats.cooldown_until_step = max(
                    stats.cooldown_until_step,
                    graph.step + self.config.outcome_feedback_cooldown_steps,
                )

    @staticmethod
    def _feedback_snapshot(graph, family, region) -> dict[str, float]:
        stats = graph.operation_feedback.get(
            feedback_key(family, region), OperationFeedbackStats(),
        )
        return {
            "observations": float(stats.observations),
            "successes": float(stats.successes),
            "no_ops": float(stats.no_ops),
            "posterior_value": float(stats.posterior_value),
            "posterior_success": float(stats.posterior_success),
            "consecutive_failures": float(stats.consecutive_failures),
            "cooldown_until_step": float(stats.cooldown_until_step),
        }

    @staticmethod
    def _normalized_actual_cost(packet, measured) -> float:
        request = packet.requested_budget
        parts = []
        if float(measured.get("llm_calls", 0.0)) > 0.0:
            parts.append(min(1.0, float(measured["llm_calls"])))
        if float(measured.get("tokens", 0.0)) > 0.0:
            parts.append(min(1.0, float(measured["tokens"]) / max(1, request.get("max_tokens", 0))))
        if float(measured.get("retrieval_calls", 0.0)) > 0.0:
            parts.append(min(1.0, float(measured["retrieval_calls"])))
        return sum(parts) / len(parts) if parts else 0.0

    def _reconcile_join_attempt(
        self, graph, packet, measured, completed, failure_reason, metadata,
    ) -> None:
        operation = packet.operation
        if operation.operation_type != OperationType.MERGE:
            return
        existing = next((
            row for row in graph.join_attempt_history
            if row.operation_id == operation.operation_id
        ), None)
        outcome = graph.operation_outcome_history[-1]
        if existing is not None:
            existing.creation_cost = measured
            existing.downstream_unlock = _unit_positive(
                outcome.actual_utility_components_raw.get("dependency_unlock_gain", 0.0)
                + outcome.actual_utility_components_raw.get("answer_chain_progress", 0.0)
            )
            return
        premise_ids = [str(value) for value in operation.payload.get("premise_ids", operation.source_ids)]
        graph.join_attempt_history.append(JoinAttemptRecord(
            attempt_id=f"join_attempt_{operation.operation_id}",
            step=graph.step,
            operation_id=operation.operation_id,
            target_subgoal=operation.target_id,
            branch_id=operation.branch_id,
            premise_ids=premise_ids,
            premise_versions={
                node_id: graph.belief_states.get(node_id).version
                if graph.belief_states.get(node_id) is not None else 0
                for node_id in premise_ids
            },
            variable_bindings={
                str(key): [str(item) for item in value]
                for key, value in operation.payload.get("variable_bindings", {}).items()
            },
            constraints=[dict(value) for value in operation.payload.get("constraints", [])],
            join_kind=str(operation.payload.get("join_kind", "relational_path")),
            signature=str(operation.payload.get("join_signature", "")),
            independent_support={
                node_id: float(graph.node(node_id, ClaimNode).score.absolute_support)
                for node_id in premise_ids
            },
            deterministic_validation=dict(operation.payload.get("deterministic_validation", {})),
            model_validation=dict(metadata),
            accepted=bool(completed),
            rejection_reason=failure_reason or "join_rejected",
            creation_cost=measured,
            downstream_unlock=0.0,
        ))

    def _create_candidates(self, graph, operation, changes) -> None:
        before = set(graph.nodes)
        super()._create_candidates(graph, operation, changes)
        rows = {
            str(row.get("node_id")): row
            for row in operation.payload.get("candidates", []) if isinstance(row, dict)
        }
        for node_id in sorted(set(graph.nodes) - before):
            node = graph.node(node_id, ClaimNode)
            row = rows[node_id]
            node.provenance.metadata["answers_subgoal"] = bool(row.get("answers_subgoal", False))
            node.provenance.metadata["answer_position"] = str(row.get("answer_position", "none"))
            node.provenance.metadata["source_triple"] = dict(row.get("source_triple", {}))
            node.provenance.metadata["extraction_evidence_count"] = int(
                row.get("extraction_evidence_count", 0)
            )
            node.provenance.metadata["typed_qualifiers"] = (
                dict(row.get("qualifiers", {})) if isinstance(row.get("qualifiers"), dict) else {}
            )
            graph.claim_semantics[node_id] = ClaimSemantics(
                node_id=node_id,
                subject_type=_canonical_type(row.get("subject_type", "entity")),
                value_type=_canonical_type(row.get("value_type", row.get("answer_type", "entity"))),
                normalized_subject=normalize_text(node.subject),
                normalized_relation=normalize_text(node.relation),
                normalized_value=normalize_text(node.value),
                qualifiers={
                    str(key): str(value) for key, value in row.get("qualifiers", {}).items()
                } if isinstance(row.get("qualifiers"), dict) else {},
                extraction_mode=str(row.get("extraction_mode", "typed_evidence_extraction")),
                join_depth=int(row.get("join_depth", 0)),
                join_signature=str(row.get("join_signature", "")),
            )

    def _merge(self, graph, operation, changes) -> None:
        if operation.payload.get("mode") != "derive_join":
            super()._merge(graph, operation, changes)
            return
        row = operation.payload.get("claim", {})
        if not isinstance(row, dict):
            raise GraphInvariantError("JOIN MERGE requires claim payload")
        source_ids = list(dict.fromkeys(str(value) for value in operation.source_ids))
        if not 2 <= len(source_ids) <= self.config.max_join_arity:
            raise GraphInvariantError("JOIN requires two to max_join_arity unique premises")
        sources = [graph.node(node_id, ClaimNode) for node_id in source_ids]
        if any(
            node.status in {
                CandidateStatus.PROPOSED, CandidateStatus.INVALID, CandidateStatus.ARCHIVED,
            }
            or node.score.absolute_support < self.config.join_min_premise_support
            for node in sources
        ):
            raise GraphInvariantError("JOIN premise is unsupported, unverified, or invalid")
        constraints = operation.payload.get("constraints", [])
        bindings = operation.payload.get("variable_bindings", {})
        if not isinstance(constraints, list) or not isinstance(bindings, dict):
            raise GraphInvariantError("JOIN constraints and variable bindings must be typed collections")
        if len(source_ids) >= 3 and (
            len(constraints) < len(source_ids) - 1
            or not bindings
            or not _join_constraints_connected(source_ids, constraints)
        ):
            raise GraphInvariantError("n-ary JOIN premises must form one constrained connected component")
        node_id = str(row.get("node_id", ""))
        edge_id = str(row.get("edge_id", ""))
        if not node_id or node_id in graph.nodes or not edge_id or edge_id in graph.hyperedges:
            raise GraphInvariantError("JOIN node and edge IDs must be new")
        target_subgoal = str(row.get("target_subgoal", operation.target_id))
        graph.node(target_subgoal, SubgoalNode)
        evidence_ids = list(dict.fromkeys(
            str(value) for source in sources for value in source.evidence_refs
        ))
        if not evidence_ids:
            raise GraphInvariantError("JOIN result must preserve premise evidence")
        semantics_rows = [graph.claim_semantics[source.node_id] for source in sources]
        join_depth = max(value.join_depth for value in semantics_rows) + 1
        if join_depth > self.config.max_join_depth:
            raise GraphInvariantError("JOIN depth exceeded")
        raw = VerificationSignals(
            grounding=min(source.score.raw.grounding for source in sources),
            entailment=min(source.score.raw.entailment for source in sources),
            type_match=_unit(row.get("type_match", 1.0)),
            dependency_consistency=_unit(row.get("dependency_consistency", 1.0)),
            retrieval_support=min(source.score.raw.retrieval_support for source in sources),
            contradiction_risk=max(source.score.raw.contradiction_risk for source in sources),
            raw_model_confidence=_unit(row.get("derivation_confidence", 0.0)),
            reasons=["typed_multi_premise_join"],
        )
        absolute_support = min(
            min(source.score.absolute_support for source in sources),
            raw.raw_model_confidence,
        )
        provenance = self._provenance(
            operation, graph.step,
            join_binding=str(operation.payload.get("binding", "")),
            join_validation=dict(operation.payload.get("validation", {})),
        )
        provenance.evidence_ids = evidence_ids
        claim = ClaimNode(
            node_id=node_id,
            subject=str(row.get("subject", "")).strip(),
            relation=str(row.get("relation", "")).strip(),
            value=str(row.get("value", "")).strip(),
            answer_type=_canonical_type(row.get("value_type", row.get("answer_type", "entity"))),
            target_subgoal=target_subgoal,
            branch_id=operation.branch_id,
            evidence_refs=evidence_ids,
            dependency_claim_ids=source_ids,
            score=CandidateScoreProfile(
                raw=raw,
                absolute_support=_unit(absolute_support),
                relative_weight=0.0,
                set_entropy=max(source.score.set_entropy for source in sources),
                evidence_gap=max(source.score.evidence_gap for source in sources),
                scoring_version="dh-v2-join-v1",
            ),
            status=CandidateStatus.SCORED,
            contradiction_links=[],
            created_at_step=graph.step,
            provenance=provenance,
        )
        if not claim.subject or not claim.relation or not claim.value:
            raise GraphInvariantError("JOIN result requires subject, relation and value")
        edge = Hyperedge(
            edge_id=edge_id,
            source_node_set=source_ids,
            target_node=node_id,
            inference_type="typed_relational_join",
            confidence=claim.score.absolute_support,
            supporting_evidence=evidence_ids,
            creation_reason=operation.reason,
            created_by_operation=operation.operation_id,
            created_at_step=graph.step,
            provenance=provenance,
        )
        graph.nodes[node_id] = claim
        graph.hyperedges[edge_id] = edge
        signature = str(operation.payload.get("join_signature", "")) or stable_hash({
            "sources": sorted(source_ids), "subject": claim.subject,
            "relation": claim.relation, "value": claim.value,
        })
        graph.claim_semantics[node_id] = ClaimSemantics(
            node_id=node_id,
            subject_type=_canonical_type(row.get("subject_type", semantics_rows[0].subject_type)),
            value_type=_canonical_type(row.get("value_type", semantics_rows[-1].value_type)),
            normalized_subject=normalize_text(claim.subject),
            normalized_relation=normalize_text(claim.relation),
            normalized_value=normalize_text(claim.value),
            qualifiers={
                str(key): str(value) for key, value in row.get("qualifiers", {}).items()
            } if isinstance(row.get("qualifiers"), dict) else {},
            extraction_mode="typed_relational_join",
            join_depth=join_depth,
            join_signature=signature,
        )
        graph.join_attempt_history.append(JoinAttemptRecord(
            attempt_id=f"join_attempt_{operation.operation_id}",
            step=graph.step,
            operation_id=operation.operation_id,
            target_subgoal=target_subgoal,
            branch_id=operation.branch_id,
            premise_ids=source_ids,
            premise_versions={
                source.node_id: graph.belief_states.get(source.node_id).version
                if graph.belief_states.get(source.node_id) is not None else 0
                for source in sources
            },
            variable_bindings={
                str(key): [str(item) for item in value]
                for key, value in operation.payload.get("variable_bindings", {}).items()
            },
            constraints=[dict(value) for value in operation.payload.get("constraints", [])],
            join_kind=str(operation.payload.get("join_kind", "relational_path")),
            signature=signature,
            independent_support={
                source.node_id: float(source.score.absolute_support) for source in sources
            },
            deterministic_validation=dict(
                operation.payload.get("deterministic_validation", {})
            ),
            model_validation=dict(operation.payload.get("validation", {})),
            accepted=True,
            conclusion_node_id=node_id,
            creation_cost={
                str(key): float(value) for key, value in operation.estimated_cost.items()
            },
        ))
        changes["created_nodes"].append(node_id)
        changes["created_hyperedges"].append(edge_id)

    def _verify(self, graph, operation, changes) -> None:
        rows = operation.payload.get("scores", {})
        super()._verify(graph, operation, changes)
        for claim_id, row in rows.items():
            claim = graph.node(str(claim_id), ClaimNode)
            semantics = graph.claim_semantics[claim.node_id]
            position = str(row.get("answer_position", "none"))
            if position not in {"subject", "value", "none"}:
                position = "none"
            was_canonical = bool(claim.provenance.metadata.get("answers_subgoal", False))
            if position == "subject" and not was_canonical:
                claim.subject, claim.value = claim.value, claim.subject
                claim.relation = f"inverse_of:{claim.relation}"
                semantics.subject_type, semantics.value_type = (
                    semantics.value_type, semantics.subject_type,
                )
                semantics.normalized_subject = normalize_text(claim.subject)
                semantics.normalized_relation = normalize_text(claim.relation)
                semantics.normalized_value = normalize_text(claim.value)
            claim.provenance.metadata["verified_answer_position"] = position
            claim.provenance.metadata["answers_subgoal"] = position in {"subject", "value"}
            changes["updated_nodes"].append(claim.node_id)

    def _commit(self, graph, operation, changes) -> None:
        super()._commit(graph, operation, changes)
        if operation.payload.get("mode") == "answer":
            return
        claim_id = str(operation.payload.get("candidate_id", ""))
        if claim_id and claim_id in graph.nodes:
            claim = graph.node(claim_id, ClaimNode)
            state = graph.belief_states.get(claim_id)
            claim.provenance.metadata["commit_belief_baseline"] = {
                "absolute_support": claim.score.absolute_support,
                "entropy": claim.score.set_entropy,
                "evidence_gap": claim.score.evidence_gap,
                "contradiction_pressure": state.contradiction_pressure if state else claim.score.raw.contradiction_risk,
                "step": graph.step,
            }

    def _revise(self, graph, operation, changes) -> None:
        if operation.payload.get("action") != "invalidate_cascade":
            super()._revise(graph, operation, changes)
            return
        if len(graph.revision_history) >= graph.limits.max_graph_revisions:
            raise GraphInvariantError("revision budget exhausted")
        target_id = str(operation.payload.get("claim_id", ""))
        target = graph.node(target_id, ClaimNode)
        if target.status in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}:
            raise GraphInvariantError("revision target is already invalid")
        invalid_nodes, invalid_edges = _downstream_cascade(graph, target_id)
        before = {node_id: _node_status(graph.nodes[node_id]) for node_id in invalid_nodes}
        for node_id in invalid_nodes:
            node = graph.nodes[node_id]
            if isinstance(node, ClaimNode):
                node.status = CandidateStatus.INVALID
                for branch in graph.branches.values():
                    for subgoal_id, assigned in list(branch.assignments.items()):
                        if assigned == node_id:
                            del branch.assignments[subgoal_id]
                            branch.completed_subgoals = [
                                value for value in branch.completed_subgoals if value != subgoal_id
                            ]
                            subgoal = graph.node(subgoal_id, SubgoalNode)
                            subgoal.status = SubgoalStatus.PARTIAL
                            changes["updated_nodes"].append(subgoal_id)
            elif isinstance(node, AnswerNode):
                node.status = AnswerStatus.REJECTED
            changes["pruned_nodes"].append(node_id)
        graph.invalidated_hyperedges = list(dict.fromkeys(graph.invalidated_hyperedges + invalid_edges))
        replacement = str(operation.payload.get("replacement_claim_id", "")) or None
        if replacement:
            replacement_claim = graph.node(replacement, ClaimNode)
            if replacement_claim.status in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}:
                raise GraphInvariantError("revision replacement is invalid")
        record = RevisionRecord(
            step=graph.step,
            operation_id=operation.operation_id,
            reason=operation.reason,
            before=before,
            after={node_id: _node_status(graph.nodes[node_id]) for node_id in invalid_nodes},
        )
        target.revision_history.append(record)
        graph.revision_history.append(record)
        graph.supersession_history.append(SupersessionRecord(
            supersession_id=f"supersession_{graph.step:04d}_{target_id}",
            step=graph.step,
            trigger=str(operation.payload.get("trigger", operation.reason)),
            trigger_source=str(operation.payload.get("trigger_source", "belief_event_detector")),
            target_claim_id=target_id,
            invalidated_node_ids=invalid_nodes,
            invalidated_hyperedge_ids=invalid_edges,
            replacement_claim_id=replacement,
            evidence_ids=[str(value) for value in operation.payload.get("evidence_ids", [])],
            natural=bool(operation.payload.get("natural", True)),
            correctness_label=str(operation.payload.get("correctness_label", "pending")),
        ))

    @staticmethod
    def _record_allocation(graph: DynamicReasoningHypergraphV2, operation: GraphOperation) -> None:
        row = operation.payload.get("_allocation")
        if not isinstance(row, dict):
            return
        graph.allocation_history.append(AllocationRecord(
            allocation_id=str(row["allocation_id"]),
            operation_id=operation.operation_id,
            step=graph.step,
            target_region=[str(value) for value in row.get("target_region", [operation.target_id])],
            predicted_evc=float(row.get("predicted_evc", 0.0)),
            evc_components_raw={str(key): float(value) for key, value in row.get("evc_components_raw", {}).items()},
            evc_components_normalized={
                str(key): float(value) for key, value in row.get("evc_components_normalized", {}).items()
            },
            requested_budget={str(key): int(value) for key, value in row.get("requested_budget", {}).items()},
            remaining_global_budget={
                str(key): int(value) for key, value in row.get("remaining_global_budget", {}).items()
            },
            actual_cost={
                str(key): float(value) for key, value in (
                    operation.estimated_cost or {"llm_calls": 0.0, "tokens": 0.0, "retrieval_calls": 0.0}
                ).items()
            },
            allocator_mode=str(row.get("allocator_mode", "adaptive_evc")),
            pre_state_summary={
                str(key): float(value) for key, value in row.get("pre_state_summary", {}).items()
            },
            feedback_prior={
                str(key): float(value) for key, value in row.get("feedback_prior", {}).items()
            },
            selected=True,
            completed=True,
        ))


def _downstream_cascade(
    graph: DynamicReasoningHypergraphV2, target_id: str,
) -> tuple[list[str], list[str]]:
    invalid_nodes = {target_id}
    invalid_edges: set[str] = set()
    changed = True
    while changed:
        changed = False
        for edge in graph.hyperedges.values():
            if edge.edge_id in invalid_edges:
                continue
            if any(source in invalid_nodes for source in edge.source_node_set):
                invalid_edges.add(edge.edge_id)
                if edge.target_node not in invalid_nodes:
                    invalid_nodes.add(edge.target_node)
                    changed = True
        for claim in graph.claims():
            if claim.node_id not in invalid_nodes and any(
                source in invalid_nodes for source in claim.dependency_claim_ids
            ):
                invalid_nodes.add(claim.node_id)
                changed = True
        for answer in graph.answers():
            if answer.node_id not in invalid_nodes and any(
                source in invalid_nodes for source in answer.supporting_claims
            ):
                invalid_nodes.add(answer.node_id)
                changed = True
    return sorted(invalid_nodes), sorted(invalid_edges)


def _node_status(node: Any) -> str:
    status = getattr(node, "status", None)
    return status.value if hasattr(status, "value") else str(status)


def _join_constraints_connected(
    premise_ids: list[str], constraints: list[dict[str, Any]],
) -> bool:
    adjacency = {node_id: set() for node_id in premise_ids}
    for row in constraints:
        if not isinstance(row, dict):
            return False
        left = str(row.get("left_premise", ""))
        right = str(row.get("right_premise", ""))
        if left not in adjacency or right not in adjacency or left == right:
            return False
        if not bool(row.get("type_compatible", False)):
            return False
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = {premise_ids[0]}
    queue = [premise_ids[0]]
    while queue:
        for node_id in adjacency[queue.pop()]:
            if node_id not in seen:
                seen.add(node_id)
                queue.append(node_id)
    return seen == set(premise_ids)


def _canonical_type(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "human": "person", "people": "person", "individual": "person",
        "nation": "country", "city": "location", "place": "location",
        "year": "date", "time": "date", "count": "number", "quantity": "number",
    }
    return aliases.get(normalized, normalized or "entity")


def _unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _unit_positive(value: Any) -> float:
    """Normalize a signed improvement without erasing its raw audit value."""
    return _unit(max(0.0, float(value)))
