from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from ..budget import Budget, BudgetExceeded
from ..dynamic.candidates import SoftCandidateVerifier
from ..dynamic.engine import DynamicInferenceInput, DynamicHypergraphReasoner, _legacy_claims, _legacy_plan, _safe
from ..dynamic.graph import (
    AnswerStatus,
    BranchState,
    BranchStatus,
    CandidateStatus,
    ClaimNode,
    EvidenceNode,
    GraphBudgetExceeded,
    GraphInvariantError,
    GraphLimits,
    GraphOperation,
    OperationType,
    SubgoalNode,
)
from ..dynamic.planner import DynamicPlanner, direct_fallback_operation
from ..dynamic.terminal import GraphGroundedTerminalReasoner
from ..llm import BaseLLM, InfrastructureError, ProviderRefusalError, StructuredOutputError
from ..models import Prediction, RetrievalHit, RunStatus, Usage
from ..retrieval import BaseRetriever
from ..utils import normalize_text, stable_hash
from .allocator import AdaptiveComputationAllocator, ComputationPacket
from .config import DynamicV2ResearchConfig
from .controller import V2GraphController
from .editor import EventTriggeredGraphEditorV2, editor_preallocation_preflight
from .extraction import TypedClaimExtractor
from .graph import DynamicReasoningHypergraphV2, TerminationKind
from .join import JoinCandidate, MultiHopJoinEngine
from .memory import RelationLightCorpusMemory
from .recovery import (
    claim_projects_target as _claim_answers_subgoal,
    diagnose_proof_gap,
    proof_gap_recovery_query,
    proof_usable_target_claim,
    strict_output_type_compatible as _strict_output_type_compatible,
)
from .revision import BeliefRevisionDetector
from .termination import MetaDecision, MetaStopPolicy, TerminalBeliefReadout
from .transitions import certified_transition_value
from .verifier import MultiSampleIndependentVerifier


class DynamicHypergraphV2Reasoner:
    """Graph-state-driven training-free reasoning and computation allocation."""

    def __init__(
        self, llm: BaseLLM, retriever: BaseRetriever, config: DynamicV2ResearchConfig,
        corpus_memory: RelationLightCorpusMemory | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.corpus_memory = corpus_memory or (
            RelationLightCorpusMemory.from_retriever(retriever)
            if config.relation_light_memory else None
        )

    def solve(self, value: DynamicInferenceInput | dict[str, Any]) -> tuple[Prediction, list[dict], list[dict]]:
        example = value if isinstance(value, DynamicInferenceInput) else DynamicInferenceInput.from_view(value)
        started = time.perf_counter()
        usage = Usage()
        budget = Budget(
            self.config.max_llm_calls, self.config.max_total_tokens,
            self.config.final_reserve_tokens, usage,
        )
        graph = self._new_graph(example.question)
        controller = V2GraphController(self.config)
        retrieval_trace: list[dict] = []
        reasoning_trace: list[dict] = []
        all_hits: list[RetrievalHit] = []
        failed_extractions: set[tuple[str, str, int]] = set()
        attempted_extractions: set[tuple[str, str, str, str]] = set()
        goal_projection_attempts: set[tuple[str, str, int]] = set()
        attempted_joins: set[str] = set()
        editor_events: set[tuple[str, str, str]] = set()
        decision = MetaDecision(TerminationKind.ABSTAIN, "not_started", 0.0)
        try:
            planner = DynamicPlanner(self.llm, budget, self.config)
            try:
                initial = self._normalize_initial_plan(
                    planner.initial_expand(example.question),
                    self.config.terminal_dependency_closure,
                )
                graph = self._apply(controller, graph, initial, reasoning_trace, None)
            except (GraphInvariantError, ProviderRefusalError, StructuredOutputError) as exc:
                self._record_model_or_plan_failure(reasoning_trace, budget, exc, "initial_plan")
                graph = self._apply(
                    controller, graph, direct_fallback_operation(example.question), reasoning_trace, None,
                )
            extractor = TypedClaimExtractor(self.llm, budget, self.config)
            verifier = MultiSampleIndependentVerifier(self.llm, budget, self.config)
            join_engine = MultiHopJoinEngine(self.llm, budget, self.config)
            editor = EventTriggeredGraphEditorV2(self.llm, budget, self.config)
            revision_detector = BeliefRevisionDetector(self.config)
            terminal = GraphGroundedTerminalReasoner(self.llm, budget, self.config)
            terminal_readout = TerminalBeliefReadout(self.config)
            allocator = AdaptiveComputationAllocator(self.config)
            stop_policy = MetaStopPolicy(self.config)

            for _ in range(self.config.max_policy_iterations):
                operations = self._ready_operations(
                    graph, terminal, revision_detector, join_engine,
                    failed_extractions, attempted_extractions, attempted_joins, editor_events,
                    terminal_readout, reasoning_trace, goal_projection_attempts,
                )
                packets = allocator.allocate(graph, operations, budget)
                decision = stop_policy.decide(graph, packets, budget)
                reasoning_trace.append({
                    "event": "meta_decision",
                    "step": graph.step,
                    "outcome": decision.outcome.value,
                    "reason": decision.reason,
                    "best_predicted_evc": decision.best_predicted_evc,
                    "selected_allocation_id": decision.selected_allocation_id,
                    "dead_end_certificate": decision.dead_end_certificate,
                    "allocation_candidates": [row.trace() for row in packets],
                })
                if decision.outcome != TerminationKind.CONTINUE:
                    break
                selected_packet = next((
                    row for row in packets
                    if row.allocation_id == decision.selected_allocation_id
                ), packets[0])
                packet = _execution_packet(selected_packet)
                if (
                    self.config.bounded_extraction_recovery
                    and packet.operation.operation_type == OperationType.BRANCH
                    and packet.operation.payload.get("mode") == "extract_typed"
                ):
                    attempted_extractions.add((
                        packet.operation.target_id,
                        packet.operation.branch_id,
                        str(packet.operation.payload.get("extraction_state_fingerprint", "")),
                        str(packet.operation.payload.get("extraction_focus_mode", "coverage")),
                    ))
                if (
                    packet.operation.operation_type == OperationType.BRANCH
                    and packet.operation.payload.get("extraction_focus_mode") == "direct_answer"
                ):
                    goal_projection_attempts.add((
                        packet.operation.target_id,
                        packet.operation.branch_id,
                        int(packet.operation.payload.get("extraction_evidence_count", 0)),
                    ))
                usage_before = (
                    budget.usage.llm_calls,
                    budget.usage.total_tokens,
                    budget.usage.retrieval_calls,
                )
                failure_reason = ""
                outcome_metadata: dict[str, Any] = {}
                allocation_exhausted = False
                try:
                    graph, progressed = self._execute(
                        example, graph, controller, extractor, verifier, join_engine, editor,
                        packet, retrieval_trace, reasoning_trace, all_hits,
                        failed_extractions, attempted_joins, editor_events, budget,
                    )
                except (BudgetExceeded, GraphBudgetExceeded) as exc:
                    failure_reason = type(exc).__name__
                    progressed = False
                    allocation_exhausted = True
                    decision = MetaDecision(
                        TerminationKind.BUDGET_EXHAUSTED,
                        "selected_computation_exhausted_safety_cap",
                        packet.predicted_evc,
                    )
                    reasoning_trace.append({
                        "event": "selected_allocation_budget_exhausted",
                        "error_type": type(exc).__name__,
                        "allocation": packet.trace(),
                    })
                except (ProviderRefusalError, StructuredOutputError) as exc:
                    failure_reason = type(exc).__name__
                    self._record_model_or_plan_failure(
                        reasoning_trace, budget, exc, packet.operation.operation_type.value.lower(),
                        packet=packet,
                    )
                    progressed = False
                    if packet.operation.operation_type == OperationType.VERIFY:
                        fallback = SoftCandidateVerifier(self.llm, budget, self.config)
                        actual, _ = fallback.deterministic_propose(
                            graph, packet.operation.target_id, packet.operation.branch_id,
                            f"{packet.operation.operation_id}_deterministic_fallback",
                        )
                        if actual is not None:
                            actual.reason = "model_failure_deterministic_verification_fallback"
                            graph = self._apply(
                                controller, graph, allocator.attach(actual, packet), reasoning_trace, packet,
                            )
                            progressed = True
                if packet.operation.operation_type == OperationType.MERGE:
                    outcome_metadata.update(join_engine.last_diagnostics)
                outcome_metadata["terminal_state_after"] = _terminal_frontier_context(
                    graph, terminal, terminal_readout, packet.operation.branch_id,
                )
                actual_cost = {
                    "llm_calls": float(budget.usage.llm_calls - usage_before[0]),
                    "tokens": float(budget.usage.total_tokens - usage_before[1]),
                    "retrieval_calls": float(budget.usage.retrieval_calls - usage_before[2]),
                }
                graph = controller.reconcile_allocation(
                    graph, packet, actual_cost, progressed,
                    failure_reason or ("operation_produced_no_commit" if not progressed else ""),
                    outcome_metadata=outcome_metadata,
                )
                outcome = graph.operation_outcome_history[-1]
                credit = next((
                    row for row in reversed(graph.credit_assignment_history)
                    if row.allocation_id == packet.allocation_id
                ), None)
                reasoning_trace.append({
                    "event": "allocation_reconciled",
                    "allocation": packet.trace(),
                    "actual_cost": actual_cost,
                    "progressed": progressed,
                    "post_state_summary": outcome.post_state_summary,
                    "state_delta": outcome.state_delta,
                    "actual_utility_components_raw": outcome.actual_utility_components_raw,
                    "actual_utility_components_normalized": (
                        outcome.actual_utility_components_normalized
                    ),
                    "actual_utility": outcome.actual_utility,
                    "actual_immediate_utility": outcome.actual_immediate_utility,
                    "actual_normalized_cost": outcome.actual_normalized_cost,
                    "delayed_realized_proof_return": (
                        outcome.delayed_realized_proof_return
                    ),
                    "combined_realized_utility": outcome.combined_realized_utility,
                    "credit_assignment": None if credit is None else {
                        "credit_id": credit.credit_id,
                        "source_step": credit.source_step,
                        "observed_at_step": credit.observed_at_step,
                        "seed_node_ids": list(credit.seed_node_ids),
                        "causal_descendant_ids": list(credit.causal_descendant_ids),
                        "causal_distance_by_node": dict(credit.causal_distance_by_node),
                        "delayed_components_raw": dict(credit.delayed_components_raw),
                        "delayed_components_normalized": dict(
                            credit.delayed_components_normalized
                        ),
                        "delayed_realized_proof_return": (
                            credit.delayed_realized_proof_return
                        ),
                        "causal_event_ids": list(credit.causal_event_ids),
                        "terminal": credit.terminal,
                        "attribution_version": credit.attribution_version,
                    },
                    "statistics_before": outcome.statistics_before,
                    "statistics_after": outcome.statistics_after,
                    "failure_reason": (
                        failure_reason
                        or ("operation_produced_no_commit" if not progressed else "")
                    ),
                })
                if allocation_exhausted:
                    break
                if not progressed:
                    if packet.operation.operation_type == OperationType.BRANCH:
                        failed_extractions.add((
                            packet.operation.target_id,
                            packet.operation.branch_id,
                            int(packet.operation.payload.get("extraction_evidence_count", 0)),
                        ))
                    # The exact failed action is never selected again on unchanged
                    # state; this makes failure a deterministic abstain candidate.
                    if packet.operation.operation_type == OperationType.MERGE:
                        attempted_joins.add(str(
                            packet.operation.payload.get("join_attempt_key")
                            or packet.operation.payload.get("join_signature", "")
                        ))
                    # Failure changes the engine's attempted/recovery state even
                    # when it does not mutate the graph. Recompute the ready set so
                    # a second retrieval or another join can receive budget.
                    continue
            else:
                decision = MetaDecision(
                    TerminationKind.BUDGET_EXHAUSTED,
                    "policy_iteration_safety_cap_exhausted", 0.0,
                )
        except BudgetExceeded:
            decision = MetaDecision(
                TerminationKind.BUDGET_EXHAUSTED,
                "llm_call_or_token_safety_cap_exhausted", 0.0,
            )
        except GraphBudgetExceeded:
            decision = MetaDecision(
                TerminationKind.BUDGET_EXHAUSTED,
                "graph_or_retrieval_safety_cap_exhausted", 0.0,
            )
        except InfrastructureError as exc:
            budget.record_infrastructure_failure(exc)
            usage.wall_seconds = time.perf_counter() - started
            return self._prediction(
                example, graph, all_hits, budget,
                MetaDecision(TerminationKind.ABSTAIN, "infrastructure_failure", 0.0),
                error=exc, infrastructure=True,
            ), retrieval_trace, reasoning_trace
        except (GraphInvariantError, TypeError, ValueError, KeyError) as exc:
            usage.wall_seconds = time.perf_counter() - started
            return self._prediction(
                example, graph, all_hits, budget,
                MetaDecision(TerminationKind.ABSTAIN, "invalid_graph_or_payload", 0.0),
                error=exc, infrastructure=True,
            ), retrieval_trace, reasoning_trace

        graph = controller.terminate(graph, decision, budget)
        reasoning_trace.append({
            "event": "termination",
            "outcome": decision.outcome.value,
            "reason": decision.reason,
            "best_predicted_evc": decision.best_predicted_evc,
            "dead_end_certificate": decision.dead_end_certificate,
            "graph_snapshot": graph.to_dict(),
        })
        usage.wall_seconds = time.perf_counter() - started
        prediction = self._prediction(example, graph, all_hits, budget, decision)
        prediction.usage = usage
        return prediction, retrieval_trace, reasoning_trace

    def _new_graph(self, question: str) -> DynamicReasoningHypergraphV2:
        graph = DynamicReasoningHypergraphV2(question, GraphLimits(
            self.config.max_candidates_per_subgoal,
            self.config.max_active_branches,
            self.config.max_graph_nodes,
            self.config.max_hyperedges,
            self.config.max_graph_revisions,
            self.config.max_revision_per_candidate,
            self.config.max_graph_depth,
            self.config.max_graph_operations,
            self.config.max_retrieval_calls,
        ))
        graph.branches["branch_root"] = BranchState(
            "branch_root", None, {}, [], 1.0, BranchStatus.ACTIVE, 0,
        )
        graph.seal_controller_state()
        return graph

    def _ready_operations(
        self,
        graph,
        terminal,
        revision_detector,
        join_engine,
        failed_extractions,
        attempted_extractions,
        attempted_joins,
        editor_events,
        terminal_readout,
        reasoning_trace,
        goal_projection_attempts,
    ):
        operations: list[GraphOperation] = []
        for trigger in revision_detector.detect(graph):
            claim = graph.node(trigger.claim_id, ClaimNode)
            branch_id = claim.branch_id
            operations.append(revision_detector.operation(
                graph, trigger, branch_id,
                f"op_v2_{graph.step + 1:04d}_revise_{_safe(trigger.claim_id)}",
            ))
        direct, unresolved_terminal_branches = terminal.direct_operations(
            graph, graph.active_branches(), f"op_v2_{graph.step + 1:04d}_answer",
        )
        direct, terminal_diagnostics = terminal_readout.evaluate(
            graph, direct,
            [branch.branch_id for branch in unresolved_terminal_branches],
        )
        if terminal_diagnostics or unresolved_terminal_branches:
            reasoning_trace.append({
                "event": "terminal_belief_readout",
                "step": graph.step,
                "scoring_version": "terminal-belief-readout-v2.2",
                "candidates": terminal_diagnostics,
                "accepted_answer_node_ids": [
                    operation.payload["answer"]["node_id"] for operation in direct
                ],
                "unresolved_branch_ids": [
                    branch.branch_id for branch in unresolved_terminal_branches
                ],
            })
        terminal_contexts = _terminal_contexts(
            terminal_diagnostics,
            [branch.branch_id for branch in unresolved_terminal_branches],
        )
        operations.extend(direct)
        if direct:
            return _unique_operations(_attach_terminal_context(operations, terminal_contexts))
        for branch in sorted(graph.active_branches(), key=lambda value: value.branch_id):
            for subgoal in sorted(graph.subgoals(), key=lambda value: value.node_id):
                if subgoal.node_id in branch.completed_subgoals:
                    continue
                if not all(dependency in branch.assignments for dependency in subgoal.dependencies):
                    continue
                question, dependencies = DynamicHypergraphReasoner._instantiate(graph, subgoal, branch)
                evidence = graph.evidence(subgoal.node_id, branch.branch_id)
                claims = [
                    claim for claim in graph.claims(subgoal.node_id, branch.branch_id)
                    if claim.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
                ]
                region_attempts = [
                    row for row in graph.retrieval_attempt_history
                    if row.target_subgoal == subgoal.node_id
                    and row.branch_id == branch.branch_id
                ]
                pending_recovery = (
                    _pending_recovery_extraction(
                        graph, subgoal.node_id, branch.branch_id, evidence,
                        claims, dependencies, attempted_extractions,
                    )
                    if self.config.proof_recovery_extraction_priority
                    else None
                )
                if pending_recovery is not None:
                    recovery_attempt, recovery_fingerprint = pending_recovery
                    reasoning_trace.append({
                        "event": "proof_recovery_extraction_priority",
                        "step": graph.step,
                        "target_id": subgoal.node_id,
                        "branch_id": branch.branch_id,
                        "retrieval_attempt_id": recovery_attempt.attempt_id,
                        "target_obligation_ids": list(
                            recovery_attempt.recovery_target_obligation_ids
                        ),
                        "new_evidence_count": recovery_attempt.new_evidence_count,
                    })
                    operations.append(self._placeholder(
                        graph, OperationType.BRANCH, subgoal, branch,
                        {
                            "mode": "extract_typed",
                            "question": question,
                            "dependency_claim_ids": dependencies,
                            "extraction_evidence_count": len(evidence),
                            "extraction_state_fingerprint": recovery_fingerprint,
                            "recovery_policy": recovery_attempt.recovery_policy,
                            "recovery_target_obligation_ids": list(
                                recovery_attempt.recovery_target_obligation_ids
                            ),
                        },
                        sources=[node.node_id for node in evidence] + dependencies,
                    ))
                    continue
                if not evidence:
                    if not self.config.region_level_retrieval_stopping:
                        query = question
                        if (
                            subgoal.node_id == "subgoal_root"
                            and normalize_text(query) != normalize_text(graph.question)
                        ):
                            query = f"{query} Original objective: {graph.question}"
                        operations.append(self._placeholder(
                            graph, OperationType.RETRIEVE, subgoal, branch,
                            {"query": query, "dependency_claim_ids": dependencies},
                        ))
                        continue
                    query = ""
                    gate = {"allowed": False, "reason_code": "retrieval_round_cap"}
                    if not region_attempts:
                        query = question
                        if (
                            subgoal.node_id == "subgoal_root"
                            and normalize_text(query) != normalize_text(graph.question)
                        ):
                            query = f"{query} Original objective: {graph.question}"
                        gate = {"allowed": True, "reason_code": "initial_region_retrieval"}
                    elif len(region_attempts) < self.config.max_retrieval_rounds_per_subgoal:
                        query = _missing_binding_query(
                            graph, subgoal, branch, question, dependencies, [], [],
                            attempted_queries={row.normalized_query for row in region_attempts},
                        )
                        if query:
                            gate = _retrieval_retry_gate(
                                graph, subgoal, branch, query, dependencies, [],
                                region_attempts, attempted_extractions,
                                self.config.retrieval_query_max_token_overlap,
                                self.config,
                            )
                    if region_attempts:
                        reasoning_trace.append({
                            "event": "region_retrieval_gate",
                            "step": graph.step,
                            "target_id": subgoal.node_id,
                            "branch_id": branch.branch_id,
                            "retrieval_round": len(region_attempts) + 1,
                            **gate,
                        })
                    if query and gate["allowed"]:
                        operations.append(self._placeholder(
                            graph, OperationType.RETRIEVE, subgoal, branch,
                            {
                                "query": query,
                                "dependency_claim_ids": dependencies,
                                "retrieval_gate_reason": gate["reason_code"],
                            },
                        ))
                    elif self.config.enable_adaptive_planning:
                        event_name = "zero_yield_retrieval_stopped"
                        event = (branch.branch_id, subgoal.node_id, event_name)
                        if event not in editor_events:
                            if self.config.no_diff_editor_preallocation_gate:
                                reasoning_trace.append({
                                    "event": "graph_editor_preallocation_filtered",
                                    "step": graph.step,
                                    "target_id": subgoal.node_id,
                                    "branch_id": branch.branch_id,
                                    **editor_preallocation_preflight(
                                        graph, event_name, subgoal.node_id,
                                    ),
                                })
                                editor_events.add(event)
                            else:
                                operations.append(self._placeholder(
                                    graph, OperationType.EXPAND, subgoal, branch,
                                    {"event": event_name},
                                ))
                    continue
                proposed = [claim for claim in claims if claim.status == CandidateStatus.PROPOSED]
                if proposed:
                    operations.append(self._placeholder(
                        graph, OperationType.VERIFY, subgoal, branch, {"question": question},
                        sources=[claim.node_id for claim in proposed],
                    ))
                    continue
                last_extracted_evidence_count = max((
                    int(claim.provenance.metadata.get("extraction_evidence_count", 0))
                    for claim in claims
                ), default=0)
                has_new_evidence_batch = len(evidence) > last_extracted_evidence_count
                extraction_key = (subgoal.node_id, branch.branch_id, len(evidence))
                projection_key = (subgoal.node_id, branch.branch_id, len(evidence))
                extraction_fingerprint = _extraction_state_fingerprint(
                    graph, subgoal.node_id, branch.branch_id, evidence, dependencies,
                )
                coverage_attempt_key = (
                    subgoal.node_id, branch.branch_id, extraction_fingerprint, "coverage",
                )
                direct_attempt_key = (
                    subgoal.node_id, branch.branch_id, extraction_fingerprint, "direct_answer",
                )
                coverage_available = (
                    coverage_attempt_key not in attempted_extractions
                    if self.config.bounded_extraction_recovery
                    else extraction_key not in failed_extractions
                )
                direct_available = (
                    coverage_attempt_key in attempted_extractions
                    and direct_attempt_key not in attempted_extractions
                    if self.config.bounded_extraction_recovery
                    else projection_key not in goal_projection_attempts
                )
                if not claims and coverage_available:
                    operations.append(self._placeholder(
                        graph, OperationType.BRANCH, subgoal, branch,
                        {
                            "mode": "extract_typed", "question": question,
                            "dependency_claim_ids": dependencies,
                            "extraction_evidence_count": len(evidence),
                            "extraction_state_fingerprint": extraction_fingerprint,
                        },
                        sources=[node.node_id for node in evidence] + dependencies,
                    ))
                    continue
                if (
                    self.config.focused_empty_extraction_recovery
                    and not claims
                    and (
                        coverage_attempt_key in attempted_extractions
                        if self.config.bounded_extraction_recovery
                        else extraction_key in failed_extractions
                    )
                    and direct_available
                ):
                    operations.append(self._placeholder(
                        graph, OperationType.BRANCH, subgoal, branch,
                        {
                            "mode": "extract_typed", "question": question,
                            "dependency_claim_ids": dependencies,
                            "extraction_evidence_count": len(evidence),
                            "extraction_focus_mode": "direct_answer",
                            "extraction_state_fingerprint": extraction_fingerprint,
                        },
                        sources=[node.node_id for node in evidence] + dependencies,
                    ))
                    continue
                semantic_direct_claims = [
                    claim for claim in claims
                    if claim.status in {CandidateStatus.SCORED, CandidateStatus.RETAINED, CandidateStatus.REVISED}
                    and _claim_answers_subgoal(graph, claim)
                ]
                proof_verdicts = {
                    claim.node_id: proof_usable_target_claim(
                        graph, claim, subgoal, self.config, projects_target=True,
                    )
                    for claim in semantic_direct_claims
                } if self.config.proof_usable_target_gate else {}
                direct_claims = (
                    [
                        claim for claim in semantic_direct_claims
                        if proof_verdicts[claim.node_id].usable
                    ]
                    if self.config.proof_usable_target_gate
                    else semantic_direct_claims
                )
                proof_gap = None
                if self.config.proof_gap_conditioned_recovery and not direct_claims:
                    proof_gap = diagnose_proof_gap(
                        graph, subgoal, claims, semantic_direct_claims,
                        proof_verdicts, dependencies,
                    )
                    if not self.config.feasibility_reasoned_recovery:
                        proof_gap = replace(proof_gap, feasibility_unlock=0.0)
                    reasoning_trace.append({
                        "event": "proof_gap_diagnosed",
                        "step": graph.step,
                        "target_id": subgoal.node_id,
                        "branch_id": branch.branch_id,
                        **proof_gap.to_payload(),
                        "claim_verdicts": {
                            claim_id: verdict.to_dict()
                            for claim_id, verdict in sorted(proof_verdicts.items())
                        },
                    })
                raw_direct_ids = {
                    claim.node_id for claim in direct_claims
                    if graph.claim_semantics[claim.node_id].join_depth == 0
                }
                if (
                    claims and not direct_claims
                    and direct_available
                ):
                    operations.append(self._placeholder(
                        graph, OperationType.BRANCH, subgoal, branch,
                        {
                            "mode": "extract_typed", "question": question,
                            "dependency_claim_ids": dependencies,
                            "extraction_evidence_count": len(evidence),
                            "extraction_focus_mode": "direct_answer",
                            "extraction_state_fingerprint": extraction_fingerprint,
                            **(proof_gap.to_payload() if proof_gap is not None else {}),
                        },
                        sources=[node.node_id for node in evidence] + dependencies,
                    ))
                    continue
                # A compositional proof must connect the committed dependency to
                # independently grounded target-local evidence.  The raw claim
                # need not itself answer the question: requiring that would make
                # genuine multi-hop joins impossible whenever no sentence states
                # the final relation directly. JOIN validation and the terminal
                # conjunctive readout still gate the derived answer.
                target_local_raw_ids = {
                    claim.node_id for claim in claims
                    if graph.claim_semantics[claim.node_id].join_depth == 0
                    and claim.evidence_refs
                    and claim.status in {
                        CandidateStatus.SCORED,
                        CandidateStatus.RETAINED,
                        CandidateStatus.REVISED,
                    }
                }
                if dependencies and target_local_raw_ids:
                    dependency_ids = set(dependencies)
                    sufficient_chains = [
                        claim for claim in direct_claims
                        if graph.claim_semantics[claim.node_id].join_depth > 0
                        and _premise_closure(graph, (claim.node_id,)) & dependency_ids
                        and _premise_closure(graph, (claim.node_id,)) & target_local_raw_ids
                    ]
                    sufficient_chains.sort(key=lambda value: (
                        graph.claim_semantics[value.node_id].join_depth,
                        -value.score.absolute_support,
                        value.node_id,
                    ))
                    if sufficient_chains and _committable(
                        sufficient_chains[0], sufficient_chains, graph, self.config,
                    ):
                        chosen = sufficient_chains[0]
                        operations.append(self._placeholder(
                            graph, OperationType.COMMIT, subgoal, branch,
                            {"candidate_id": chosen.node_id}, sources=[chosen.node_id],
                        ))
                        continue
                discovered_joins = join_engine.discover(
                    graph, branch.branch_id, subgoal.node_id,
                )
                if self.config.join_preallocation_feasibility_filter:
                    feasible_joins = []
                    filtered_joins = []
                    for candidate in discovered_joins:
                        attempt_key = _join_attempt_key(graph, candidate)
                        if attempt_key in attempted_joins:
                            continue
                        feasibility = join_engine.check_feasible(graph, candidate)
                        if feasibility.feasible:
                            feasible_joins.append(candidate)
                            continue
                        attempted_joins.add(attempt_key)
                        filtered_joins.append({
                            "join_signature": candidate.signature,
                            "join_attempt_key": attempt_key,
                            "premise_ids": list(candidate.premise_ids),
                            "reason_codes": list(feasibility.reason_codes),
                            "failed_premise_ids": list(feasibility.premise_ids),
                        })
                    if filtered_joins:
                        reasoning_trace.append({
                            "event": "join_preallocation_filtered",
                            "step": graph.step,
                            "target_id": subgoal.node_id,
                            "branch_id": branch.branch_id,
                            "filtered": filtered_joins,
                            "provider_calls": 0,
                            "provider_tokens": 0,
                        })
                        if (
                            proof_gap is not None
                            and self.config.feasibility_reasoned_recovery
                        ):
                            failed_ids = tuple(sorted({
                                str(claim_id)
                                for row in filtered_joins
                                for claim_id in row["failed_premise_ids"]
                            }))
                            feasibility_reasons = tuple(sorted({
                                str(reason)
                                for row in filtered_joins
                                for reason in row["reason_codes"]
                            }))
                            proof_gap = replace(
                                proof_gap,
                                reason_code=(
                                    f"join_{feasibility_reasons[0]}"
                                    if feasibility_reasons else "join_precondition_gap"
                                ),
                                reason_codes=tuple(sorted(set(
                                    proof_gap.reason_codes + feasibility_reasons
                                ))),
                                target_claim_ids=failed_ids or proof_gap.target_claim_ids,
                                proof_gap_reducibility=max(
                                    proof_gap.proof_gap_reducibility, 0.90,
                                ),
                                feasibility_unlock=1.0,
                            )
                            reasoning_trace.append({
                                "event": "proof_gap_feasibility_update",
                                "step": graph.step,
                                "target_id": subgoal.node_id,
                                "branch_id": branch.branch_id,
                                **proof_gap.to_payload(),
                                "filtered_join_count": len(filtered_joins),
                            })
                    discovered_joins = feasible_joins
                joins = [
                    row for row in discovered_joins
                    if _join_attempt_key(graph, row) not in attempted_joins
                    and _nary_relevant(graph, row, set(dependencies))
                ]
                if self.config.goal_conditioned_join_frontier:
                    joins = [
                        row for row in joins
                        if _join_can_answer_subgoal(graph, row, subgoal, branch)
                    ]
                if _charged_join_attempt_count(
                    graph,
                ) >= self.config.max_join_attempts_per_question:
                    joins = []
                if dependencies:
                    dependency_ids = set(dependencies)
                    # A sequential decomposition is still an explicit multi-hop
                    # composition: materialize the dependency claim + current
                    # relation before committing the subgoal value. Intermediate
                    # bridge joins are allowed when their transitive premise closure
                    # contains a dependency; a later JOIN must still connect that
                    # state to a target-local grounded claim. Entity/type equality remains
                    # the only discovery rule and Qwen validates every derivation.
                    direct_ids = target_local_raw_ids
                    direct_endpoints = {
                        normalize_text(endpoint)
                        for claim in claims if claim.node_id in direct_ids
                        for endpoint in (claim.subject, claim.value)
                    }
                    chain_joins = [
                        row for row in joins
                        if _premise_closure(graph, row.premise_ids) & dependency_ids
                        and _premise_closure(graph, row.premise_ids) & direct_ids
                    ]
                    joins = sorted(chain_joins, key=lambda row: (
                        -int(row.join_kind in {"numeric_argmax", "numeric_argmin"}),
                        -int(bool(row.projection_premise_id)),
                        -float(row.deterministic_validation.get("goal_alignment", 0.0)),
                        -len(_premise_closure(graph, row.premise_ids) & dependency_ids),
                        -int(bool(_premise_closure(graph, row.premise_ids) & direct_ids)),
                        -sum(
                            normalize_text(endpoint) in direct_endpoints
                            for endpoint in row.open_endpoints
                        ),
                        len(row.premise_ids),
                        row.join_depth,
                        row.premise_ids,
                    ))
                joins = joins[: self.config.max_join_proposals_per_step]
                if joins and (not direct_claims or dependencies):
                    join = joins[0]
                    operations.append(self._placeholder(
                        graph, OperationType.MERGE, subgoal, branch,
                        {
                            "mode": "validate_join", "premise_ids": list(join.premise_ids),
                            "binding": join.binding, "join_signature": join.signature,
                            "join_attempt_key": _join_attempt_key(graph, join),
                            "join_depth": join.join_depth,
                            "orientation": join.orientation,
                            "open_endpoints": list(join.open_endpoints),
                            "projection_premise_id": join.projection_premise_id,
                            "variable_bindings": join.variable_bindings,
                            "constraints": [dict(value) for value in join.constraints],
                            "join_kind": join.join_kind,
                            "deterministic_validation": join.deterministic_validation,
                        },
                        sources=list(join.premise_ids),
                    ))
                    if not self.config.goal_conditioned_join_frontier:
                        continue
                if direct_claims and not dependencies:
                    chosen = max(direct_claims, key=lambda value: (
                        graph.claim_semantics[value.node_id].join_depth,
                        value.score.absolute_support,
                        value.node_id,
                    ))
                    if _committable(chosen, direct_claims, graph, self.config):
                        operations.append(self._placeholder(
                            graph, OperationType.COMMIT, subgoal, branch,
                            {"candidate_id": chosen.node_id}, sources=[chosen.node_id],
                        ))
                        continue
                    branch_candidates = []
                    seen_values = set()
                    for claim in sorted(direct_claims, key=lambda value: (
                        -value.score.relative_weight,
                        -value.score.absolute_support,
                        value.node_id,
                    )):
                        normalized = normalize_text(claim.value)
                        if (
                            normalized in seen_values
                            or claim.status not in {
                                CandidateStatus.SCORED,
                                CandidateStatus.RETAINED,
                                CandidateStatus.REOPENED,
                            }
                        ):
                            continue
                        seen_values.add(normalized)
                        branch_candidates.append(claim.node_id)
                    available_width = (
                        graph.limits.max_active_branches
                        - len(graph.active_branches()) + 1
                    )
                    if (
                        not dependencies and not subgoal.terminal
                        and min(len(branch_candidates), available_width) >= 2
                    ):
                        candidate_ids = branch_candidates[:available_width]
                        operations.append(self._placeholder(
                            graph, OperationType.BRANCH, subgoal, branch,
                            {
                                "mode": "assignments",
                                "candidate_ids": candidate_ids,
                            },
                            sources=candidate_ids,
                        ))
                        continue
                if has_new_evidence_batch and coverage_available:
                    operations.append(self._placeholder(
                        graph, OperationType.BRANCH, subgoal, branch,
                        {
                            "mode": "extract_typed", "question": question,
                            "dependency_claim_ids": dependencies,
                            "extraction_evidence_count": len(evidence),
                            "extraction_state_fingerprint": extraction_fingerprint,
                        },
                        sources=[node.node_id for node in evidence] + dependencies,
                    ))
                    continue
                region_attempts = [
                    row for row in graph.retrieval_attempt_history
                    if row.target_subgoal == subgoal.node_id
                    and row.branch_id == branch.branch_id
                ]
                # Legacy v2.2 graphs have no explicit attempt ledger.  Their
                # evidence queries remain a conservative lower bound.
                retrieval_rounds = (
                    len(region_attempts) if (
                        self.config.retrieval_attempt_aware_scheduling
                        and graph.retrieval_attempt_history
                    )
                    else len({node.retrieval_query for node in evidence})
                )
                if retrieval_rounds < self.config.max_retrieval_rounds_per_subgoal:
                    attempted_queries = (
                        {row.normalized_query for row in region_attempts}
                        if self.config.retrieval_attempt_aware_scheduling else set()
                    )
                    if self.config.proof_gap_conditioned_recovery and proof_gap is not None:
                        query = proof_gap_recovery_query(
                            graph, subgoal, question, dependencies, claims,
                            proof_gap, attempted_queries,
                            compact_objective=(
                                self.config.compact_objective_recovery_query
                            ),
                            anchored_objective=(
                                self.config.anchored_proof_recovery_query
                            ),
                        )
                    else:
                        query = _missing_binding_query(
                            graph, subgoal, branch, question, dependencies, claims, evidence,
                            attempted_queries=attempted_queries,
                        )
                    retrieval_gate = {
                        "allowed": True,
                        "reason_code": "legacy_or_initial_retrieval",
                    }
                    if (
                        query
                        and retrieval_rounds > 0
                        and self.config.region_level_retrieval_stopping
                    ):
                        retrieval_gate = _retrieval_retry_gate(
                            graph, subgoal, branch, query, dependencies, claims,
                            region_attempts, attempted_extractions,
                            self.config.retrieval_query_max_token_overlap,
                            self.config,
                        )
                        reasoning_trace.append({
                            "event": "region_retrieval_gate",
                            "step": graph.step,
                            "target_id": subgoal.node_id,
                            "branch_id": branch.branch_id,
                            "retrieval_round": retrieval_rounds + 1,
                            **retrieval_gate,
                        })
                        if not retrieval_gate["allowed"]:
                            query = ""
                    if query:
                        operations.append(self._placeholder(
                            graph, OperationType.RETRIEVE, subgoal, branch,
                            {
                                "query": query,
                                "dependency_claim_ids": dependencies,
                                "retrieval_gate_reason": retrieval_gate["reason_code"],
                                **(proof_gap.to_payload() if proof_gap is not None else {}),
                            },
                            sources=[claim.node_id for claim in claims],
                        ))
                        continue
                if self.config.enable_adaptive_planning:
                    event_name = "high_uncertainty_no_join" if claims else "missing_terminal_path"
                    event = (branch.branch_id, subgoal.node_id, event_name)
                    if event not in editor_events:
                        if self.config.no_diff_editor_preallocation_gate:
                            reasoning_trace.append({
                                "event": "graph_editor_preallocation_filtered",
                                "step": graph.step,
                                "target_id": subgoal.node_id,
                                "branch_id": branch.branch_id,
                                **editor_preallocation_preflight(
                                    graph, event_name, subgoal.node_id,
                                ),
                            })
                            editor_events.add(event)
                        else:
                            operations.append(self._placeholder(
                                graph, OperationType.EXPAND, subgoal, branch,
                                {"event": event_name}, sources=[claim.node_id for claim in claims],
                            ))
        operations = _suppress_terminal_expansion_when_commit_ready(operations)
        return _unique_operations(_attach_terminal_context(operations, terminal_contexts))

    def _execute(
        self,
        example,
        graph,
        controller,
        extractor,
        verifier,
        join_engine,
        editor,
        packet,
        retrieval_trace,
        reasoning_trace,
        all_hits,
        failed_extractions,
        attempted_joins,
        editor_events,
        budget,
    ):
        operation = packet.operation
        request = packet.requested_budget
        if operation.operation_type == OperationType.RETRIEVE:
            query = str(operation.payload["query"])
            budget.record_retrieval()
            hits = self.retriever.search(query, request["retrieval_top_k"])
            existing_hits = {value.passage.passage_id for value in all_hits}
            all_hits.extend(value for value in hits if value.passage.passage_id not in existing_hits)
            rows = []
            activated_hits = _novel_retrieval_hits_for_region(
                graph, hits, operation.target_id, operation.branch_id,
            )
            for hit in activated_hits:
                rows.append({
                    "node_id": f"evidence_v2_{graph.step + 1}_{len(rows) + 1}_{_safe(hit.passage.passage_id)}",
                    "document_id": hit.passage.passage_id,
                    "passage_id": hit.passage.passage_id,
                    "title": hit.passage.title,
                    "source_span": hit.passage.text,
                    "retrieval_rank": hit.rank,
                    "retrieval_score": hit.raw_score,
                    "retrieval_query": query,
                    "retriever_identity": hit.retriever,
                })
            memory_activation = {}
            if self.corpus_memory is not None and rows:
                memory_activation = self.corpus_memory.activate(
                    activated_hits,
                    [str(row["node_id"]) for row in rows],
                    graph.question,
                    operation.target_id,
                    operation.branch_id,
                    query,
                ).to_payload()
                memory_activation["corpus_memory_fingerprint"] = self.corpus_memory.fingerprint
            actual = GraphOperation(
                operation.operation_id, OperationType.RETRIEVE, operation.target_id,
                operation.source_ids, operation.branch_id,
                {
                    "query": query,
                    "evidence": rows,
                    "memory_activation": memory_activation,
                    "allocated_top_k": int(request["retrieval_top_k"]),
                    "hit_count": len(hits),
                },
                "adaptive_retrieval_for_heated_region", "adaptive_retriever_v2",
                {"retrieval_calls": 1.0, "top_k": float(request["retrieval_top_k"])},
            )
            graph = self._apply(controller, graph, AdaptiveComputationAllocator.attach(actual, packet), reasoning_trace, packet)
            retrieval_trace.append({
                "step": graph.step, "operation_id": actual.operation_id,
                "subgoal_id": actual.target_id, "branch_id": actual.branch_id,
                "query": query, "allocated_top_k": request["retrieval_top_k"],
                "hits": [value.to_dict() for value in hits],
            })
            return graph, (
                bool(rows) if self.config.retrieval_attempt_aware_scheduling else True
            )
        if operation.operation_type == OperationType.BRANCH:
            if str(operation.payload.get("mode", "")) == "assignments":
                if request.get("branch_width", 0) < 2:
                    return graph, False
                available_width = (
                    graph.limits.max_active_branches
                    - len(graph.active_branches()) + 1
                )
                candidate_ids = list(operation.payload.get("candidate_ids", []))[
                    :min(int(request["branch_width"]), available_width)
                ]
                if len(candidate_ids) < 2:
                    return graph, False
                actual = GraphOperation(
                    operation.operation_id,
                    OperationType.BRANCH,
                    operation.target_id,
                    candidate_ids,
                    operation.branch_id,
                    {"mode": "assignments", "candidate_ids": candidate_ids},
                    "adaptive_branch_width_for_ambiguous_typed_candidates",
                    "adaptive_computation_allocator_v2",
                    {
                        "graph_operations": 1.0,
                        "branches_created": float(len(candidate_ids)),
                    },
                )
                execution_packet = _bind_transition_to_execution(
                    graph, packet, actual, self.config,
                )
                graph = self._apply(
                    controller, graph,
                    AdaptiveComputationAllocator.attach(actual, execution_packet),
                    reasoning_trace, execution_packet,
                )
                return graph, True
            actual = extractor.propose(
                graph, operation.target_id, operation.branch_id,
                str(operation.payload["question"]),
                list(operation.payload.get("dependency_claim_ids", [])),
                operation.operation_id, request["max_tokens"], request["candidate_cap"],
                str(operation.payload.get("extraction_focus_mode", "coverage")),
            )
            reasoning_trace.append({
                "event": "typed_extraction_diagnostic",
                "operation_id": operation.operation_id,
                "target_id": operation.target_id,
                "branch_id": operation.branch_id,
                "focus_mode": str(
                    operation.payload.get("extraction_focus_mode", "coverage")
                ),
                "extraction_evidence_count": int(
                    operation.payload.get("extraction_evidence_count", 0)
                ),
                "extraction_state_fingerprint": str(
                    operation.payload.get("extraction_state_fingerprint", "")
                ),
                "accepted": actual is not None,
                "diagnostics": dict(extractor.last_diagnostics),
            })
            if actual is None:
                reasoning_trace.append({
                    "event": "typed_extraction_rejected",
                    "operation_id": operation.operation_id,
                    "target_id": operation.target_id,
                    "branch_id": operation.branch_id,
                    "focus_mode": str(
                        operation.payload.get("extraction_focus_mode", "coverage")
                    ),
                    "extraction_evidence_count": int(
                        operation.payload.get("extraction_evidence_count", 0)
                    ),
                    "extraction_state_fingerprint": str(
                        operation.payload.get("extraction_state_fingerprint", "")
                    ),
                    "diagnostics": extractor.last_diagnostics,
                })
                return graph, False
            graph = self._apply(controller, graph, AdaptiveComputationAllocator.attach(actual, packet), reasoning_trace, packet)
            return graph, True
        if operation.operation_type == OperationType.VERIFY:
            actual = verifier.propose(
                graph, operation.target_id, operation.branch_id,
                str(operation.payload["question"]), operation.operation_id,
                request["verification_samples"], request["max_tokens"],
            )
            if actual is None:
                return graph, False
            graph = self._apply(controller, graph, AdaptiveComputationAllocator.attach(actual, packet), reasoning_trace, packet)
            return graph, True
        if operation.operation_type == OperationType.MERGE:
            candidate = JoinCandidate(
                tuple(str(value) for value in operation.payload["premise_ids"]),
                str(operation.payload["binding"]), operation.target_id,
                str(operation.payload["join_signature"]), int(operation.payload["join_depth"]),
                str(operation.payload.get("orientation", "value_subject")),
                tuple(str(value) for value in operation.payload.get("open_endpoints", [])),
                str(operation.payload.get("projection_premise_id", "")),
                {
                    str(key): [str(item) for item in value]
                    for key, value in operation.payload.get("variable_bindings", {}).items()
                },
                tuple(dict(value) for value in operation.payload.get("constraints", [])),
                str(operation.payload.get("join_kind", "relational_path")),
                dict(operation.payload.get("deterministic_validation", {})),
            )
            attempted_joins.add(str(
                operation.payload.get("join_attempt_key") or candidate.signature
            ))
            actual = join_engine.propose(
                graph, candidate, operation.operation_id, request["max_tokens"],
            )
            if actual is None:
                reasoning_trace.append({
                    "event": "join_rejected", "join_signature": candidate.signature,
                    "premise_ids": list(candidate.premise_ids),
                    "diagnostics": join_engine.last_diagnostics,
                })
                return graph, False
            graph = self._apply(controller, graph, AdaptiveComputationAllocator.attach(actual, packet), reasoning_trace, packet)
            return graph, True
        if operation.operation_type == OperationType.EXPAND:
            event_name = str(operation.payload.get("event", "missing_terminal_path"))
            event = (operation.branch_id, operation.target_id, event_name)
            editor_events.add(event)
            actual = editor.propose(
                graph, event_name, graph.branches[operation.branch_id],
                operation.operation_id, operation.target_id, request["max_tokens"],
            )
            if actual is None:
                reasoning_trace.append({
                    "event": "graph_editor_rejected", "target_id": operation.target_id,
                    "branch_id": operation.branch_id, "trigger": event_name,
                    "diagnostics": editor.last_diagnostics,
                    "allocation": packet.trace(),
                })
                return graph, False
            graph = self._apply(
                controller, graph, AdaptiveComputationAllocator.attach(actual, packet),
                reasoning_trace, packet,
            )
            return graph, True
        if (
            operation.operation_type == OperationType.REVISE
            and request.get("revision_allowance", 0) < 1
        ):
            return graph, False
        graph = self._apply(
            controller, graph, AdaptiveComputationAllocator.attach(operation, packet), reasoning_trace, packet,
        )
        return graph, True

    @staticmethod
    def _placeholder(graph, kind, subgoal, branch, payload, sources=None):
        return GraphOperation(
            f"op_v2_{graph.step + 1:04d}_{kind.value.lower()}_{_safe(subgoal.node_id)}_{_safe(branch.branch_id)}",
            kind, subgoal.node_id, sources or [], branch.branch_id, payload,
            "graph_state_generated_computation_candidate", "v2_ready_operation_builder",
            {"llm_calls": 0.0, "tokens": 0.0},
        )
    @staticmethod
    def _normalize_initial_plan(
        operation: GraphOperation,
        terminal_dependency_closure: bool = False,
    ) -> GraphOperation:
        """Collapse a near-duplicate final subgoal/root pair into an alias.

        This prevents applying the same relation twice when a planner emits both a
        final decomposition step and a lightly rephrased root.  A root that
        literally consumes its dependency variable is never collapsed: that
        binding proves an additional outer relation even under high lexical
        overlap.  Unconsumed/malformed aliases retain the bounded lexical guard.
        """
        if operation.operation_type != OperationType.EXPAND:
            return operation
        rows = operation.payload.get("subgoals", [])
        if not isinstance(rows, list):
            return operation
        by_id = {str(row.get("node_id")): row for row in rows if isinstance(row, dict)}
        root = by_id.get("subgoal_root")
        if not root:
            return _close_terminal_dependency_gap(operation) if terminal_dependency_closure else operation
        dependencies = [str(value) for value in root.get("dependencies", [])]
        if len(dependencies) != 1 or dependencies[0] not in by_id:
            return _close_terminal_dependency_gap(operation) if terminal_dependency_closure else operation
        source = by_id[dependencies[0]]
        if not _strict_output_type_compatible(
            str(root.get("answer_type", "entity")),
            str(source.get("answer_type", "entity")),
        ):
            return _close_terminal_dependency_gap(operation) if terminal_dependency_closure else operation
        root_template = str(root.get("question_template", ""))
        consumes_dependency = any(
            str(source_id) == dependencies[0] and str(variable) in root_template
            for variable, source_id in (root.get("variable_bindings", {}) or {}).items()
        )
        if consumes_dependency:
            return _close_terminal_dependency_gap(operation) if terminal_dependency_closure else operation
        similarity = _template_similarity(
            root_template, str(source.get("question_template", "")),
        )
        if similarity < 0.55:
            return _close_terminal_dependency_gap(operation) if terminal_dependency_closure else operation
        value = GraphOperation(
            operation.operation_id, operation.operation_type, operation.target_id,
            list(operation.source_ids), operation.branch_id,
            {**operation.payload, "subgoals": [dict(row) for row in rows]},
            operation.reason, operation.proposed_by,
            dict(operation.estimated_cost), dict(operation.utility_components),
        )
        normalized_rows = value.payload["subgoals"]
        normalized_root = next(row for row in normalized_rows if row.get("node_id") == "subgoal_root")
        normalized_root["question_template"] = str(source.get("question_template", ""))
        normalized_root["instantiated_question"] = str(source.get("instantiated_question", ""))
        normalized_root["variable_bindings"] = dict(source.get("variable_bindings", {}))
        normalized_root["dependencies"] = list(source.get("dependencies", []))
        value.payload["subgoals"] = [
            row for row in normalized_rows if row.get("node_id") != dependencies[0]
        ]
        return _close_terminal_dependency_gap(value) if terminal_dependency_closure else value

    @staticmethod
    def _apply(controller, graph, operation, trace, packet):
        updated = controller.apply(graph, operation)
        audit = updated.operation_history[-1]
        trace.append({
            "event": "graph_operation",
            "step_id": audit.step,
            "operation": audit.operation_type.value,
            "operation_id": audit.operation_id,
            "graph_before_hash": audit.graph_before_hash,
            "graph_after_hash": audit.graph_after_hash,
            "created_nodes": audit.created_nodes,
            "updated_nodes": audit.updated_nodes,
            "pruned_nodes": audit.pruned_nodes,
            "created_hyperedges": audit.created_hyperedges,
            "allocation": packet.trace() if packet is not None else None,
            "actual_cost": dict(operation.estimated_cost),
            "belief_channels": {
                node_id: state.__dict__ for node_id, state in updated.belief_states.items()
            },
            "diffusion": updated.diffusion_history[-1].__dict__ if updated.diffusion_history else None,
            "revision_reason": operation.reason if operation.operation_type == OperationType.REVISE else None,
        })
        return updated

    @staticmethod
    def _record_model_or_plan_failure(trace, budget, error, stage, packet=None):
        generation = None
        if isinstance(error, ProviderRefusalError):
            budget.record_provider_failure(error)
        elif isinstance(error, StructuredOutputError):
            generation = error.generation
            try:
                budget.record_generation(generation)
            except BudgetExceeded:
                pass
        row = {
            "event": "recoverable_model_failure", "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        if generation is not None:
            row["generation"] = {
                "finish_reason": generation.finish_reason,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "cached": generation.cached,
            }
        if packet is not None:
            row["allocation"] = packet.trace()
            row["actual_cost"] = {
                "llm_calls": 1.0 if generation is not None else 0.0,
                "tokens": float(
                    generation.prompt_tokens + generation.completion_tokens
                    if generation is not None else 0
                ),
            }
        trace.append(row)

    @staticmethod
    def _prediction(example, graph, hits, budget, decision, error=None, infrastructure=False):
        answer = (
            graph.nodes.get(decision.answer_node_id)
            if decision.answer_node_id is not None else None
        )
        if answer is not None and (
            not hasattr(answer, "status") or answer.status != AnswerStatus.ACCEPTED
        ):
            answer = None
        if infrastructure:
            status = RunStatus.INFRASTRUCTURE_FAILURE
        elif decision.outcome == TerminationKind.ANSWER and answer is not None:
            status = RunStatus.ANSWER
        elif decision.outcome == TerminationKind.BUDGET_EXHAUSTED:
            status = RunStatus.BUDGET_EXHAUSTED
        else:
            status = RunStatus.ABSTAIN
        best = max(
            (claim for claim in graph.claims() if claim.status != CandidateStatus.INVALID),
            key=lambda value: value.score.absolute_support,
            default=None,
        )
        return Prediction(
            qid=example.qid,
            question=example.question,
            status=status,
            answer=answer.candidate_answer if status == RunStatus.ANSWER else None,
            confidence=answer.confidence if status == RunStatus.ANSWER else 0.0,
            stop_reason=decision.reason,
            best_unverified_candidate=best.value if best is not None and status != RunStatus.ANSWER else None,
            rejection_reasons=[] if status == RunStatus.ANSWER else [decision.reason],
            claims=_legacy_claims(graph),
            plan=_legacy_plan(graph),
            retrieved=hits,
            usage=budget.usage,
            error=None if error is None else f"{type(error).__name__}: {error}",
        )


def _execution_packet(packet: ComputationPacket) -> ComputationPacket:
    """Give every selected action an ID unique across no-op graph states.

    Ready-operation IDs describe state candidates and may repeat when a rejected
    model proposal leaves `graph.step` unchanged. Allocation IDs, in contrast,
    are monotonic within the question, so they safely namespace the actual action.
    """
    operation = deepcopy(packet.operation)
    operation.operation_id = f"{operation.operation_id}_{packet.allocation_id}"
    return replace(packet, operation=operation)


def _bind_transition_to_execution(
    graph: DynamicReasoningHypergraphV2,
    packet: ComputationPacket,
    operation: GraphOperation,
    config: DynamicV2ResearchConfig,
) -> ComputationPacket:
    """Bind an allocation promise to the concrete post-fidelity operation."""

    if not config.bind_transition_certificate_to_execution:
        return packet
    certificate = certified_transition_value(graph, operation, config)
    return replace(
        packet,
        operation=operation,
        transition_certificate=certificate,
        predicted_transition_value=float(
            certificate.get("predicted_transition_value", 0.0)
        ),
    )


def _unique_operations(values: list[GraphOperation]) -> list[GraphOperation]:
    rows = {}
    for value in values:
        rows.setdefault(value.operation_id, value)
    return list(rows.values())


def _charged_join_attempt_count(graph: DynamicReasoningHypergraphV2) -> int:
    """Count only JOINs that consumed model compute or produced a conclusion.

    Deterministic precondition rejections are still version-keyed and audited,
    but do not consume the semantic JOIN budget.  Graph-operation and policy
    caps remain the hard bound on these zero-call checks.
    """
    return sum(
        bool(row.accepted)
        or float((row.creation_cost or {}).get("llm_calls", 0.0)) > 0.0
        for row in graph.join_attempt_history
    )


def _suppress_terminal_expansion_when_commit_ready(
    operations: list[GraphOperation],
) -> list[GraphOperation]:
    """Do not mutate a shared terminal schema while a proof is commit-ready."""
    ready_targets = {
        operation.target_id for operation in operations
        if operation.operation_type == OperationType.COMMIT
        and operation.payload.get("candidate_id")
    }
    if not ready_targets:
        return operations
    return [
        operation for operation in operations
        if not (
            operation.operation_type == OperationType.EXPAND
            and operation.target_id in ready_targets
        )
    ]


def _terminal_contexts(
    diagnostics: list[dict], unresolved_branch_ids: list[str],
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for row in sorted(diagnostics, key=lambda value: (
        float(value.get("terminal_gap", 1.0)),
        -float(value.get("absolute_support", 0.0)),
        str(value.get("candidate_answer", "")),
    )):
        branch_id = str(row.get("branch_id", ""))
        contexts.setdefault(branch_id, {
            "candidate_answer": str(row.get("candidate_answer", "")),
            "terminal_gap": float(row.get("terminal_gap", 1.0)),
            "absolute_support": float(row.get("absolute_support", 0.0)),
            "relative_weight": float(row.get("relative_weight", 0.0)),
            "entropy": float(row.get("entropy", 1.0)),
            "competition_entropy": float(row.get("competition_entropy", 1.0)),
            "evidence_gap": float(row.get("evidence_gap", 1.0)),
            "relative_margin": float(row.get("relative_margin", 0.0)),
            "contradiction_pressure": float(row.get("contradiction_pressure", 0.0)),
            "answer_type_consistency": float(row.get("answer_type_consistency", 0.0)),
            "chain_coverage": float(row.get("chain_coverage", 0.0)),
            "sufficient_chain": bool(row.get("sufficient_chain", False)),
            "accepted": bool(row.get("accepted", False)),
            "rejection_reasons": [str(value) for value in row.get("rejection_reasons", [])],
            "scoring_version": str(row.get("scoring_version", "terminal-belief-readout-v2.2")),
        })
    for branch_id in sorted(set(unresolved_branch_ids)):
        contexts.setdefault(branch_id, {
            "candidate_answer": "",
            "terminal_gap": 1.0,
            "absolute_support": 0.0,
            "relative_weight": 0.0,
            "entropy": 1.0,
            "competition_entropy": 1.0,
            "evidence_gap": 1.0,
            "relative_margin": 0.0,
            "contradiction_pressure": 0.0,
            "answer_type_consistency": 0.0,
            "chain_coverage": 0.0,
            "sufficient_chain": False,
            "accepted": False,
            "rejection_reasons": ["missing_terminal_candidate"],
            "scoring_version": "terminal-belief-readout-v2.2",
        })
    return contexts


def _attach_terminal_context(
    operations: list[GraphOperation], contexts: dict[str, dict[str, Any]],
) -> list[GraphOperation]:
    attached = []
    for operation in operations:
        value = deepcopy(operation)
        value.payload = dict(value.payload)
        value.payload["_terminal_context"] = dict(contexts.get(
            value.branch_id,
            {
                "terminal_gap": 1.0,
                "absolute_support": 0.0,
                "relative_weight": 0.0,
                "entropy": 1.0,
                "evidence_gap": 1.0,
                "chain_coverage": 0.0,
                "rejection_reasons": ["missing_terminal_candidate"],
                "scoring_version": "terminal-belief-readout-v2.2",
            },
        ))
        attached.append(value)
    return attached


def _terminal_frontier_context(
    graph: DynamicReasoningHypergraphV2,
    terminal,
    terminal_readout,
    branch_id: str,
) -> dict[str, Any]:
    direct, unresolved = terminal.direct_operations(
        graph, graph.active_branches(), f"op_v2_{graph.step + 1:04d}_outcome_readout",
    )
    _, diagnostics = terminal_readout.evaluate(
        graph, direct, [branch.branch_id for branch in unresolved],
    )
    contexts = _terminal_contexts(
        diagnostics, [branch.branch_id for branch in unresolved],
    )
    if branch_id in contexts:
        return contexts[branch_id]
    if graph.terminal_beliefs:
        profile = max(
            graph.terminal_beliefs.values(),
            key=lambda value: (value.accepted, value.relative_weight, value.absolute_support),
        )
        return {
            "candidate_answer": profile.candidate_answer,
            "terminal_gap": profile.terminal_gap,
            "absolute_support": profile.absolute_support,
            "relative_weight": profile.relative_weight,
            "entropy": profile.entropy,
            "competition_entropy": profile.competition_entropy,
            "evidence_gap": profile.evidence_gap,
            "relative_margin": profile.relative_margin,
            "contradiction_pressure": profile.contradiction_pressure,
            "answer_type_consistency": profile.answer_type_consistency,
            "chain_coverage": profile.chain_coverage,
            "sufficient_chain": profile.sufficient_chain,
            "accepted": profile.accepted,
            "rejection_reasons": list(profile.rejection_reasons),
            "scoring_version": profile.scoring_version,
        }
    if contexts:
        return min(contexts.values(), key=lambda value: (
            float(value.get("terminal_gap", 1.0)),
            -float(value.get("absolute_support", 0.0)),
        ))
    return {
        "terminal_gap": 1.0,
        "absolute_support": 0.0,
        "relative_weight": 0.0,
        "entropy": 1.0,
        "competition_entropy": 1.0,
        "evidence_gap": 1.0,
        "relative_margin": 0.0,
        "contradiction_pressure": 0.0,
        "answer_type_consistency": 0.0,
        "chain_coverage": 0.0,
        "sufficient_chain": False,
        "accepted": False,
        "rejection_reasons": ["missing_terminal_candidate"],
        "scoring_version": "terminal-belief-readout-v2.2",
    }


def _committable(
    chosen: ClaimNode,
    candidates: list[ClaimNode],
    graph: DynamicReasoningHypergraphV2,
    config: DynamicV2ResearchConfig,
) -> bool:
    """Apply the frozen support/margin/entropy gate before binding a slot."""
    if chosen.score.absolute_support < config.commit_support_threshold:
        return False
    if chosen.score.evidence_gap > config.terminal_max_evidence_gap:
        return False
    if chosen.score.raw.contradiction_risk >= config.terminal_max_contradiction:
        return False
    if chosen.score.raw.type_match < config.terminal_min_type_consistency:
        return False
    if graph.claim_semantics[chosen.node_id].join_depth > 0:
        return True
    if chosen.score.set_entropy > config.commit_entropy_threshold:
        return False
    by_answer = {}
    for candidate in candidates:
        key = normalize_text(candidate.value)
        by_answer[key] = max(by_answer.get(key, 0.0), candidate.score.relative_weight)
    weights = sorted(by_answer.values(), reverse=True)
    margin = weights[0] - weights[1] if len(weights) > 1 else 1.0
    return margin >= config.commit_margin_threshold


def _close_terminal_dependency_gap(operation: GraphOperation) -> GraphOperation:
    """Make every emitted non-terminal frontier a prerequisite of the root.

    Provider plans occasionally emit useful bridge subgoals while declaring the
    terminal root independent.  That lets terminal evidence bypass unfinished
    reasoning.  Only dependency-free roots are repaired, and only with sink
    nodes already present in the provider plan; no semantic edge or answer label
    is invented.
    """
    rows = operation.payload.get("subgoals", [])
    if not isinstance(rows, list):
        return operation
    typed_rows = [row for row in rows if isinstance(row, dict)]
    root = next((row for row in typed_rows if row.get("node_id") == "subgoal_root"), None)
    if root is None or root.get("dependencies"):
        return operation
    nonterminal = {
        str(row.get("node_id")): row
        for row in typed_rows
        if row.get("node_id") != "subgoal_root" and not bool(row.get("terminal", False))
    }
    if not nonterminal:
        return operation
    consumed = {
        str(dependency)
        for row in nonterminal.values()
        for dependency in row.get("dependencies", [])
        if str(dependency) in nonterminal
    }
    sinks = sorted(set(nonterminal) - consumed)
    if not sinks:
        return operation
    value = GraphOperation(
        operation.operation_id, operation.operation_type, operation.target_id,
        list(operation.source_ids), operation.branch_id,
        {**operation.payload, "subgoals": [dict(row) for row in rows]},
        operation.reason, operation.proposed_by,
        dict(operation.estimated_cost), dict(operation.utility_components),
    )
    normalized_root = next(
        row for row in value.payload["subgoals"] if row.get("node_id") == "subgoal_root"
    )
    normalized_root["dependencies"] = sinks
    normalized_root["dependency_repair"] = "terminal_sink_closure_v1"
    return value


def _join_can_answer_subgoal(
    graph: DynamicReasoningHypergraphV2,
    candidate: JoinCandidate,
    subgoal: SubgoalNode,
    branch: BranchState,
) -> bool:
    """Hard-filter JOINs that cannot project the requested slot output."""
    if candidate.projection_premise_id:
        return True
    if candidate.join_kind in {"numeric_argmax", "numeric_argmin", "set_intersection"}:
        return True
    if len(candidate.open_endpoints) != 2:
        return False
    anchors = {
        normalize_text(str(value))
        for row in graph.query_graph.get("constraints", [])
        if str(row.get("subgoal_id")) == subgoal.node_id
        for value in row.get("known_entities", [])
        if normalize_text(str(value))
    }
    for dependency_id in subgoal.dependencies:
        assigned = graph.nodes.get(str(branch.assignments.get(dependency_id, "")))
        if isinstance(assigned, ClaimNode):
            anchors.update(normalize_text(value) for value in (assigned.subject, assigned.value) if value)
    bound = [normalize_text(value) in anchors for value in candidate.open_endpoints]
    if sum(bound) != 1:
        return False
    output = normalize_text(candidate.open_endpoints[bound.index(False)])
    output_types: set[str] = set()
    for node_id in candidate.premise_ids:
        claim = graph.node(node_id, ClaimNode)
        semantics = graph.claim_semantics[node_id]
        if normalize_text(claim.subject) == output:
            output_types.add(semantics.subject_type)
        if normalize_text(claim.value) == output:
            output_types.add(semantics.value_type)
    return any(
        _strict_output_type_compatible(value, subgoal.answer_type)
        for value in output_types
    )


def _premise_closure(
    graph: DynamicReasoningHypergraphV2, premise_ids: tuple[str, ...],
) -> set[str]:
    closure = set(premise_ids)
    queue = list(premise_ids)
    while queue:
        node = graph.nodes.get(queue.pop())
        if not isinstance(node, ClaimNode):
            continue
        for dependency_id in node.dependency_claim_ids:
            if dependency_id not in closure:
                closure.add(dependency_id)
                queue.append(dependency_id)
    return closure


def _nary_relevant(
    graph: DynamicReasoningHypergraphV2,
    candidate: JoinCandidate,
    dependency_ids: set[str],
) -> bool:
    """Reject redundant n-ary frontiers before they can consume a model call."""
    if len(candidate.premise_ids) < 3:
        return True
    if candidate.join_kind == "set_intersection":
        return bool(candidate.deterministic_validation.get("set_intersection_members"))
    closure = _premise_closure(graph, candidate.premise_ids)
    if len(closure & dependency_ids) >= 2:
        return True
    orientations = {str(row.get("orientation", "")) for row in candidate.constraints}
    if (
        candidate.join_kind == "conjunctive_relational_path"
        and orientations == {"value_subject"}
        and len(candidate.open_endpoints) == 2
    ):
        return True
    if candidate.join_kind == "shared_role_conjunction":
        relations = {
            graph.claim_semantics[node_id].normalized_relation
            for node_id in candidate.premise_ids
        }
        return len(relations) == len(candidate.premise_ids)
    return False


def _extraction_state_fingerprint(
    graph: DynamicReasoningHypergraphV2,
    target_subgoal: str,
    branch_id: str,
    evidence: list[EvidenceNode],
    dependency_ids: list[str],
) -> str:
    """Fingerprint all state that can materially change extraction output."""
    regional_claim_ids = {
        claim.node_id for claim in graph.claims(target_subgoal, branch_id)
    }
    return stable_hash({
        "target_subgoal": target_subgoal,
        "branch_id": branch_id,
        "evidence": sorted((
            node.node_id,
            node.passage_id,
            stable_hash(node.source_span),
        ) for node in evidence),
        "dependencies": sorted((
            node_id,
            graph.belief_states.get(node_id).version
            if graph.belief_states.get(node_id) is not None else 0,
        ) for node_id in dependency_ids),
        "relevant_supersessions": sorted(
            row.supersession_id for row in graph.supersession_history
            if row.target_claim_id in regional_claim_ids
            or row.target_claim_id in set(dependency_ids)
        ),
    })


def _pending_recovery_extraction(
    graph: DynamicReasoningHypergraphV2,
    target_subgoal: str,
    branch_id: str,
    evidence: list[EvidenceNode],
    claims: list[ClaimNode],
    dependency_ids: list[str],
    attempted_extractions: set[tuple[str, str, str, str]],
) -> tuple[Any, str] | None:
    """Return fresh, unconsumed recovery evidence in one graph region."""
    attempts = [
        row for row in graph.retrieval_attempt_history
        if row.target_subgoal == target_subgoal and row.branch_id == branch_id
    ]
    if not evidence or not attempts:
        return None
    latest = attempts[-1]
    if (
        latest.recovery_policy != "proof_gap_recovery_v1"
        or latest.new_evidence_count <= 0
        or not latest.recovery_target_obligation_ids
    ):
        return None
    last_extracted_evidence_count = max((
        int(claim.provenance.metadata.get("extraction_evidence_count", 0))
        for claim in claims
    ), default=0)
    if len(evidence) <= last_extracted_evidence_count:
        return None
    fingerprint = _extraction_state_fingerprint(
        graph, target_subgoal, branch_id, evidence, dependency_ids,
    )
    key = (target_subgoal, branch_id, fingerprint, "coverage")
    if key in attempted_extractions:
        return None
    return latest, fingerprint


def _query_token_overlap(left: str, right: str) -> float:
    import re

    left_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(left)))
    right_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def _retrieval_retry_gate(
    graph: DynamicReasoningHypergraphV2,
    subgoal: SubgoalNode,
    branch: BranchState,
    query: str,
    dependency_ids: list[str],
    claims: list[ClaimNode],
    region_attempts: list[Any],
    attempted_extractions: set[tuple[str, str, str, str]],
    max_query_overlap: float,
    config: DynamicV2ResearchConfig | None = None,
) -> dict[str, Any]:
    """Gold-free hard gate for a second retrieval in one graph region."""
    if not region_attempts:
        return {"allowed": True, "reason_code": "initial_region_retrieval"}
    overlaps = [
        _query_token_overlap(query, row.query) for row in region_attempts
    ]
    maximum_overlap = max(overlaps, default=0.0)
    common = {
        "query_token_overlap": maximum_overlap,
        "previous_attempt_id": region_attempts[-1].attempt_id,
        "previous_new_evidence_count": region_attempts[-1].new_evidence_count,
    }
    if maximum_overlap > max_query_overlap:
        return {
            "allowed": False,
            "reason_code": "query_not_materially_novel",
            **common,
        }
    last = region_attempts[-1]
    if last.new_evidence_count == 0:
        return {
            "allowed": True,
            "reason_code": "zero_yield_novel_query_recovery",
            **common,
        }
    last_passages = set(last.passage_ids)
    yielded_claims = [
        claim for claim in claims
        if any(
            isinstance(graph.nodes.get(evidence_id), EvidenceNode)
            and graph.nodes[evidence_id].passage_id in last_passages
            for evidence_id in claim.evidence_refs
        )
    ]
    if not yielded_claims:
        evidence = graph.evidence(subgoal.node_id, branch.branch_id)
        fingerprint = _extraction_state_fingerprint(
            graph, subgoal.node_id, branch.branch_id, evidence, dependency_ids,
        )
        coverage_done = (
            subgoal.node_id, branch.branch_id, fingerprint, "coverage",
        ) in attempted_extractions
        direct_done = (
            subgoal.node_id, branch.branch_id, fingerprint, "direct_answer",
        ) in attempted_extractions
        if coverage_done and direct_done:
            return {
                "allowed": True,
                "reason_code": "bounded_empty_extraction_recovery",
                "coverage_attempted": True,
                "direct_answer_attempted": True,
                **common,
            }
        return {
            "allowed": False,
            "reason_code": "extraction_recovery_not_exhausted",
            "coverage_attempted": coverage_done,
            "direct_answer_attempted": direct_done,
            **common,
        }
    target_usable = []
    for claim in yielded_claims:
        projects = _claim_answers_subgoal(graph, claim)
        if not projects:
            continue
        if config is not None and config.proof_usable_target_gate:
            target_usable.append(proof_usable_target_claim(
                graph, claim, subgoal, config, projects_target=True,
            ).usable)
        else:
            target_usable.append(True)
    if any(target_usable):
        return {
            "allowed": False,
            "reason_code": "target_local_answer_claim_already_grounded",
            "yielded_claim_count": len(yielded_claims),
            **common,
        }
    return {
        "allowed": True,
        "reason_code": "missing_binding_after_grounded_frontier",
        "yielded_claim_count": len(yielded_claims),
        "dependency_count": len(dependency_ids),
        **common,
    }


def _missing_binding_query(
    graph: DynamicReasoningHypergraphV2,
    subgoal,
    branch,
    question: str,
    dependency_ids: list[str],
    claims: list[ClaimNode],
    evidence: list[Any],
    attempted_queries: set[str] | None = None,
) -> str:
    """Choose a novel query from unresolved typed graph state, never oracle fields."""
    existing = {normalize_text(row.retrieval_query) for row in evidence}
    existing.update(attempted_queries or set())
    dependencies = [
        graph.node(node_id, ClaimNode) for node_id in dependency_ids
        if isinstance(graph.nodes.get(node_id), ClaimNode)
    ]
    dependency_values = list(dict.fromkeys(
        value for claim in dependencies for value in (claim.value, claim.subject) if value
    ))[:4]
    frontier = sorted(
        (
            claim for claim in claims
            if claim.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
        ),
        key=lambda row: (
            -graph.belief_states.get(row.node_id).evidence_gap
            if graph.belief_states.get(row.node_id) is not None else 0.0,
            row.node_id,
        ),
    )
    frontier_values = list(dict.fromkeys(
        value for claim in frontier[:3] for value in (claim.subject, claim.value) if value
    ))[:5]
    answer_type = str(getattr(subgoal, "answer_type", "entity") or "entity")
    candidates = []
    if (
        bool(getattr(subgoal, "terminal", False))
        and normalize_text(question) != normalize_text(graph.question)
        and normalize_text(graph.question) not in existing
    ):
        # The coarse planner may produce a useful executable inner query while
        # omitting an outer relation.  Give the immutable user objective one
        # independent retrieval turn instead of only concatenating it to the
        # inner query, which can dilute lexical retrieval for that outer edge.
        candidates.append(graph.question)
    if dependency_values:
        candidates.append(
            f"{question} Resolve the missing {answer_type} binding from "
            f"{' ; '.join(dependency_values)}."
        )
    if frontier_values:
        candidates.append(
            f"{question} Find evidence connecting {' ; '.join(frontier_values)} "
            f"to an answer of type {answer_type}."
        )
    candidates.append(
        f"{question} Find a missing typed relation for the unresolved {answer_type} binding."
    )
    for candidate in candidates:
        if normalize_text(candidate) not in existing:
            return candidate
    return ""


def _novel_retrieval_hits_for_region(
    graph: DynamicReasoningHypergraphV2,
    hits: list[RetrievalHit],
    target_subgoal: str,
    branch_id: str,
) -> list[RetrievalHit]:
    """Keep only passage-level novelty inside one reasoning region.

    A changed query is an audit event, not new evidence.  The same passage may
    still be activated independently for another subgoal/branch, where it has a
    different graph role.
    """
    seen = {
        node.passage_id for node in graph.evidence(target_subgoal, branch_id)
    }
    novel = []
    for hit in hits:
        passage_id = hit.passage.passage_id
        if passage_id in seen:
            continue
        seen.add(passage_id)
        novel.append(hit)
    return novel


def _template_similarity(left: str, right: str) -> float:
    import re

    stop = {
        "a", "an", "the", "name", "of", "was", "is", "did", "does", "what", "which",
        "who", "when", "where", "how", "that", "this",
    }
    def tokens(value):
        normalized = re.sub(r"\$[A-Za-z][A-Za-z0-9_]*", " variable ", value.casefold())
        return {
            _light_stem(token)
            for token in re.findall(r"[a-z0-9]+", normalized)
            if token not in stop
        }
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    return max(
        intersection / len(left_tokens | right_tokens),
        intersection / min(len(left_tokens), len(right_tokens)),
    )


def _light_stem(token: str) -> str:
    for suffix in ("ization", "ation", "tion", "ingly", "edly", "ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[:-len(suffix)]
    return token


def _join_attempt_key(
    graph: DynamicReasoningHypergraphV2, candidate: JoinCandidate,
) -> str:
    """Bind a failed JOIN to the premise belief state that was evaluated."""
    return stable_hash({
        "signature": candidate.signature,
        "premise_versions": {
            node_id: (
                graph.belief_states[node_id].version
                if node_id in graph.belief_states else 0
            )
            for node_id in candidate.premise_ids
        },
    })
