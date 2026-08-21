from __future__ import annotations

import time
from dataclasses import dataclass
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
from ..utils import normalize_text
from .allocator import AdaptiveComputationAllocator, ComputationPacket
from .config import DynamicV2ResearchConfig
from .controller import V2GraphController
from .editor import EventTriggeredGraphEditorV2
from .extraction import TypedClaimExtractor
from .graph import DynamicReasoningHypergraphV2, TerminationKind
from .join import JoinCandidate, MultiHopJoinEngine
from .revision import BeliefRevisionDetector
from .termination import MetaDecision, MetaStopPolicy
from .verifier import MultiSampleIndependentVerifier


class DynamicHypergraphV2Reasoner:
    """Graph-state-driven training-free reasoning and computation allocation."""

    def __init__(self, llm: BaseLLM, retriever: BaseRetriever, config: DynamicV2ResearchConfig) -> None:
        self.llm = llm
        self.retriever = retriever
        self.config = config

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
        failed_extractions: set[tuple[str, str]] = set()
        attempted_joins: set[str] = set()
        editor_events: set[tuple[str, str, str]] = set()
        decision = MetaDecision(TerminationKind.ABSTAIN, "not_started", 0.0)
        try:
            planner = DynamicPlanner(self.llm, budget, self.config)
            try:
                initial = self._normalize_initial_plan(planner.initial_expand(example.question))
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
            allocator = AdaptiveComputationAllocator(self.config)
            stop_policy = MetaStopPolicy(self.config)

            for _ in range(self.config.max_policy_iterations):
                operations = self._ready_operations(
                    graph, terminal, revision_detector, join_engine,
                    failed_extractions, attempted_joins, editor_events,
                )
                packets = allocator.allocate(graph, operations, budget)
                decision = stop_policy.decide(graph, packets, budget)
                reasoning_trace.append({
                    "event": "meta_decision",
                    "step": graph.step,
                    "outcome": decision.outcome.value,
                    "reason": decision.reason,
                    "best_predicted_evc": decision.best_predicted_evc,
                    "allocation_candidates": [row.trace() for row in packets],
                })
                if decision.outcome != TerminationKind.CONTINUE:
                    break
                packet = packets[0]
                usage_before = (
                    budget.usage.llm_calls,
                    budget.usage.total_tokens,
                    budget.usage.retrieval_calls,
                )
                failure_reason = ""
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
                actual_cost = {
                    "llm_calls": float(budget.usage.llm_calls - usage_before[0]),
                    "tokens": float(budget.usage.total_tokens - usage_before[1]),
                    "retrieval_calls": float(budget.usage.retrieval_calls - usage_before[2]),
                }
                graph = controller.reconcile_allocation(
                    graph, packet, actual_cost, progressed,
                    failure_reason or ("operation_produced_no_commit" if not progressed else ""),
                )
                reasoning_trace.append({
                    "event": "allocation_reconciled",
                    "allocation": packet.trace(),
                    "actual_cost": actual_cost,
                    "progressed": progressed,
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
                        attempted_joins.add(str(packet.operation.payload.get("join_signature", "")))
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
        attempted_joins,
        editor_events,
    ):
        operations: list[GraphOperation] = []
        for trigger in revision_detector.detect(graph):
            claim = graph.node(trigger.claim_id, ClaimNode)
            branch_id = claim.branch_id
            operations.append(revision_detector.operation(
                graph, trigger, branch_id,
                f"op_v2_{graph.step + 1:04d}_revise_{_safe(trigger.claim_id)}",
            ))
        direct, _ = terminal.direct_operations(
            graph, graph.active_branches(), f"op_v2_{graph.step + 1:04d}_answer",
        )
        operations.extend(direct)
        if direct:
            return _unique_operations(operations)
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
                if not evidence:
                    query = question
                    if subgoal.node_id == "subgoal_root" and normalize_text(query) != normalize_text(graph.question):
                        query = f"{query} Original objective: {graph.question}"
                    operations.append(self._placeholder(
                        graph, OperationType.RETRIEVE, subgoal, branch,
                        {"query": query, "dependency_claim_ids": dependencies},
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
                if not claims and extraction_key not in failed_extractions:
                    operations.append(self._placeholder(
                        graph, OperationType.BRANCH, subgoal, branch,
                        {
                            "mode": "extract_typed", "question": question,
                            "dependency_claim_ids": dependencies,
                            "extraction_evidence_count": len(evidence),
                        },
                        sources=[node.node_id for node in evidence] + dependencies,
                    ))
                    continue
                direct_claims = [
                    claim for claim in claims
                    if claim.status in {CandidateStatus.SCORED, CandidateStatus.RETAINED, CandidateStatus.REVISED}
                    and (
                        bool(claim.provenance.metadata.get("answers_subgoal", False))
                        or graph.claim_semantics[claim.node_id].extraction_mode == "typed_relational_join"
                    )
                ]
                raw_direct_ids = {
                    claim.node_id for claim in direct_claims
                    if graph.claim_semantics[claim.node_id].join_depth == 0
                }
                if dependencies and raw_direct_ids:
                    dependency_ids = set(dependencies)
                    sufficient_chains = [
                        claim for claim in direct_claims
                        if graph.claim_semantics[claim.node_id].join_depth > 0
                        and _premise_closure(graph, (claim.node_id,)) & dependency_ids
                        and _premise_closure(graph, (claim.node_id,)) & raw_direct_ids
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
                joins = [
                    row for row in join_engine.discover(graph, branch.branch_id, subgoal.node_id)
                    if row.signature not in attempted_joins
                ]
                if len(attempted_joins) >= self.config.max_join_attempts_per_question:
                    joins = []
                if dependencies:
                    dependency_ids = set(dependencies)
                    # A sequential decomposition is still an explicit multi-hop
                    # composition: materialize the dependency claim + current
                    # relation before committing the subgoal value. Intermediate
                    # bridge joins are allowed when their transitive premise closure
                    # contains a dependency; a later JOIN must still connect that
                    # state to a direct answer claim. Entity/type equality remains
                    # the only discovery rule and Qwen validates every derivation.
                    direct_ids = raw_direct_ids
                    direct_endpoints = {
                        normalize_text(endpoint)
                        for claim in direct_claims
                        for endpoint in (claim.subject, claim.value)
                    }
                    chain_joins = [
                        row for row in joins
                        if _premise_closure(graph, row.premise_ids) & dependency_ids
                    ]
                    joins = sorted(chain_joins, key=lambda row: (
                        -int(bool(row.projection_premise_id)),
                        -int(bool(_premise_closure(graph, row.premise_ids) & direct_ids)),
                        -sum(
                            normalize_text(endpoint) in direct_endpoints
                            for endpoint in row.open_endpoints
                        ),
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
                            "join_depth": join.join_depth,
                            "orientation": join.orientation,
                            "open_endpoints": list(join.open_endpoints),
                            "projection_premise_id": join.projection_premise_id,
                        },
                        sources=list(join.premise_ids),
                    ))
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
                if has_new_evidence_batch and extraction_key not in failed_extractions:
                    operations.append(self._placeholder(
                        graph, OperationType.BRANCH, subgoal, branch,
                        {
                            "mode": "extract_typed", "question": question,
                            "dependency_claim_ids": dependencies,
                            "extraction_evidence_count": len(evidence),
                        },
                        sources=[node.node_id for node in evidence] + dependencies,
                    ))
                    continue
                retrieval_rounds = len({node.retrieval_query for node in evidence})
                if retrieval_rounds < self.config.max_retrieval_rounds_per_subgoal:
                    operations.append(self._placeholder(
                        graph, OperationType.RETRIEVE, subgoal, branch,
                        {
                            "query": f"{question} Find a missing relation that connects the existing typed claims.",
                            "dependency_claim_ids": dependencies,
                        },
                        sources=[claim.node_id for claim in claims],
                    ))
                    continue
                if self.config.enable_adaptive_planning:
                    event_name = "high_uncertainty_no_join" if claims else "missing_terminal_path"
                    event = (branch.branch_id, subgoal.node_id, event_name)
                    if event not in editor_events:
                        operations.append(self._placeholder(
                            graph, OperationType.EXPAND, subgoal, branch,
                            {"event": event_name}, sources=[claim.node_id for claim in claims],
                        ))
        return _unique_operations(operations)

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
            existing = {(node.passage_id, node.retrieval_query) for node in graph.evidence()}
            rows = []
            for hit in hits:
                if (hit.passage.passage_id, query) in existing:
                    continue
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
            actual = GraphOperation(
                operation.operation_id, OperationType.RETRIEVE, operation.target_id,
                operation.source_ids, operation.branch_id,
                {"query": query, "evidence": rows},
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
            return graph, True
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
                graph = self._apply(
                    controller, graph, AdaptiveComputationAllocator.attach(actual, packet),
                    reasoning_trace, packet,
                )
                return graph, True
            actual = extractor.propose(
                graph, operation.target_id, operation.branch_id,
                str(operation.payload["question"]),
                list(operation.payload.get("dependency_claim_ids", [])),
                operation.operation_id, request["max_tokens"], request["candidate_cap"],
            )
            if actual is None:
                reasoning_trace.append({
                    "event": "typed_extraction_rejected", "target_id": operation.target_id,
                    "branch_id": operation.branch_id, "diagnostics": extractor.last_diagnostics,
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
                tuple(str(value) for value in operation.payload.get("open_endpoints", []))[:2],
                str(operation.payload.get("projection_premise_id", "")),
            )
            attempted_joins.add(candidate.signature)
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
    def _normalize_initial_plan(operation: GraphOperation) -> GraphOperation:
        """Collapse a near-duplicate final subgoal/root pair into an alias.

        This prevents applying the same relation twice when a planner emits both a
        final decomposition step and a lightly rephrased root.  It is lexical and
        structural, independent of relation names and question IDs.
        """
        if operation.operation_type != OperationType.EXPAND:
            return operation
        rows = operation.payload.get("subgoals", [])
        if not isinstance(rows, list):
            return operation
        by_id = {str(row.get("node_id")): row for row in rows if isinstance(row, dict)}
        root = by_id.get("subgoal_root")
        if not root:
            return operation
        dependencies = [str(value) for value in root.get("dependencies", [])]
        if len(dependencies) != 1 or dependencies[0] not in by_id:
            return operation
        source = by_id[dependencies[0]]
        similarity = _template_similarity(
            str(root.get("question_template", "")), str(source.get("question_template", "")),
        )
        if similarity < 0.55:
            return operation
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
        normalized_root["variable_bindings"] = {}
        normalized_root["dependencies"] = [dependencies[0]]
        return value

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
        answer = next((
            node for node in sorted(graph.answers(), key=lambda value: (-value.confidence, value.node_id))
            if node.status == AnswerStatus.ACCEPTED
        ), None)
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


def _unique_operations(values: list[GraphOperation]) -> list[GraphOperation]:
    rows = {}
    for value in values:
        rows.setdefault(value.operation_id, value)
    return list(rows.values())


def _committable(
    chosen: ClaimNode,
    candidates: list[ClaimNode],
    graph: DynamicReasoningHypergraphV2,
    config: DynamicV2ResearchConfig,
) -> bool:
    """Apply the frozen support/margin/entropy gate before binding a slot."""
    if chosen.score.absolute_support < config.commit_support_threshold:
        return False
    if graph.claim_semantics[chosen.node_id].join_depth > 0:
        return chosen.score.raw.contradiction_risk < config.contradiction_threshold
    if chosen.score.set_entropy > config.commit_entropy_threshold:
        return False
    by_answer = {}
    for candidate in candidates:
        key = normalize_text(candidate.value)
        by_answer[key] = max(by_answer.get(key, 0.0), candidate.score.relative_weight)
    weights = sorted(by_answer.values(), reverse=True)
    margin = weights[0] - weights[1] if len(weights) > 1 else 1.0
    return margin >= config.commit_margin_threshold


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


def _template_similarity(left: str, right: str) -> float:
    import re

    stop = {
        "a", "an", "the", "name", "of", "was", "is", "did", "does", "what", "which",
        "who", "when", "where", "how", "that", "this",
    }
    def tokens(value):
        normalized = re.sub(r"\$[A-Za-z][A-Za-z0-9_]*", " variable ", value.casefold())
        return {token for token in re.findall(r"[a-z0-9]+", normalized) if token not in stop}
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
