from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from ..budget import Budget, BudgetExceeded
from ..llm import BaseLLM, InfrastructureError, ProviderRefusalError, StructuredOutputError
from ..models import (
    Claim,
    ClaimStatus,
    Passage,
    Prediction,
    ReasoningPlan,
    ReasoningSlot,
    RetrievalHit,
    RunStatus,
    SlotStatus,
    Usage,
    VariableBinding,
)
from ..retrieval import BaseRetriever
from ..utils import normalize_text
from .candidates import DynamicCandidateGenerator, SoftCandidateVerifier
from .config import DynamicResearchConfig
from .controller import GraphController
from .editor import EventTriggeredGraphEditor
from .graph import (
    AnswerNode,
    BranchState,
    BranchStatus,
    CandidateStatus,
    ClaimNode,
    DynamicReasoningHypergraph,
    EvidenceNode,
    GraphInvariantError,
    GraphBudgetExceeded,
    GraphLimits,
    GraphOperation,
    OperationType,
    SubgoalNode,
)
from .planner import DynamicPlanner, direct_fallback_operation
from .policy import candidate_prune_value, decide_candidate_set
from .scheduler import OperationScheduler, OperationSignals
from .scoring import CandidateSetSummary
from .terminal import GraphGroundedTerminalReasoner


@dataclass(frozen=True)
class DynamicInferenceInput:
    qid: str
    question: str
    passages: list[Passage]

    @classmethod
    def from_view(cls, value: dict[str, Any]) -> "DynamicInferenceInput":
        # The method entry receives QAExample.inference_view(), not QAExample, so
        # gold answer/decomposition/support labels are structurally unavailable.
        return cls(
            qid=str(value["qid"]),
            question=str(value["question"]),
            passages=[Passage(**row) for row in value.get("passages", [])],
        )


class DynamicHypergraphReasoner:
    def __init__(self, llm: BaseLLM, retriever: BaseRetriever, config: DynamicResearchConfig) -> None:
        self.llm = llm
        self.retriever = retriever
        self.config = config

    def solve(self, value: DynamicInferenceInput | dict[str, Any]) -> tuple[Prediction, list[dict], list[dict]]:
        example = value if isinstance(value, DynamicInferenceInput) else DynamicInferenceInput.from_view(value)
        started = time.perf_counter()
        usage = Usage()
        budget = Budget(self.config.max_llm_calls, self.config.max_total_tokens, self.config.final_reserve_tokens, usage)
        graph = self._new_graph(example.question)
        controller = GraphController()
        retrieval_trace: list[dict] = []
        reasoning_trace: list[dict] = []
        all_hits: list[RetrievalHit] = []
        summaries: dict[tuple[str, str], CandidateSetSummary] = {}
        candidate_failures: set[tuple[str, str]] = set()
        editor_events: set[tuple[str, str, str]] = set()
        terminal_attempted = False
        failed_decision: tuple[str, str] | None = None
        policy_iterations = 0
        try:
            planner = DynamicPlanner(self.llm, budget, self.config)
            try:
                initial = planner.initial_expand(example.question)
                graph = self._apply(controller, graph, initial, reasoning_trace, [], None)
            except (GraphInvariantError, ProviderRefusalError, StructuredOutputError) as exc:
                if isinstance(exc, ProviderRefusalError):
                    self._record_refusal(reasoning_trace, budget, exc, "initial_plan", "subgoal_root", "branch_root")
                elif isinstance(exc, StructuredOutputError):
                    self._record_structured_failure(
                        reasoning_trace, budget, exc, "initial_plan", "subgoal_root", "branch_root",
                    )
                graph = self._apply(
                    controller, graph, direct_fallback_operation(example.question),
                    reasoning_trace, [], None,
                )
            generator = DynamicCandidateGenerator(self.llm, budget, self.config)
            verifier = SoftCandidateVerifier(self.llm, budget, self.config)
            editor = EventTriggeredGraphEditor(self.llm, budget, self.config)
            scheduler = OperationScheduler(self.config)
            terminal = GraphGroundedTerminalReasoner(self.llm, budget, self.config)

            while (
                len(graph.operation_history) < self.config.max_graph_operations
                and policy_iterations < self.config.max_policy_iterations
            ):
                policy_iterations += 1
                # Alias roots and event-inserted final relations already have a
                # complete graph-grounded answer path. Do not retrieve and solve
                # the same relation a second time.
                direct, _ = terminal.direct_operations(
                    graph, graph.active_branches(), f"op_{graph.step + 1:04d}_answer_ready",
                )
                if direct:
                    for answer_operation in direct:
                        if len(graph.operation_history) >= graph.limits.max_graph_operations:
                            break
                        graph = self._apply(
                            controller, graph, answer_operation, reasoning_trace, [], None,
                        )
                    break
                operations, signals = self._ready_operations(
                    graph, summaries, candidate_failures, editor_events, budget,
                )
                if not operations:
                    if terminal_attempted:
                        break
                    terminal_attempted = True
                    try:
                        graph, produced = self._terminalize(
                            graph, controller, terminal, reasoning_trace, budget,
                        )
                    except (ProviderRefusalError, StructuredOutputError) as exc:
                        self._record_model_failure(
                            reasoning_trace, budget, exc, "terminal", "subgoal_root", "*",
                        )
                        produced = False
                    if produced:
                        break
                    # No graph-grounded answer. Give the event-triggered editor one
                    # final chance to add a missing subgoal, never an answer.
                    active = graph.active_branches()
                    if not active or not self.config.enable_adaptive_planning:
                        break
                    branch = active[0]
                    event = (branch.branch_id, "subgoal_root", "missing_terminal_path")
                    if event in editor_events:
                        break
                    editor_events.add(event)
                    try:
                        proposals = editor.propose(
                            graph, "missing_terminal_path", branch,
                            f"op_{graph.step + 1:04d}_editor_terminal", "subgoal_root",
                        )
                    except (ProviderRefusalError, StructuredOutputError) as exc:
                        self._record_model_failure(
                            reasoning_trace, budget, exc, "expand", "subgoal_root", branch.branch_id,
                        )
                        break
                    if not self.config.enable_revision:
                        proposals = [value for value in proposals if value.operation_type != OperationType.REVISE]
                    if not proposals:
                        break
                    for proposal in proposals:
                        graph = self._apply(controller, graph, proposal, reasoning_trace, [], None)
                    terminal_attempted = False
                    continue

                scheduler_active = self.config.enable_operation_scheduler and len(operations) > 1
                if scheduler_active:
                    selected = scheduler.select(operations, signals)
                    assert selected is not None
                    operation = selected.operation
                    ranking = scheduler.last_ranking
                else:
                    operation = sorted(
                        operations,
                        key=lambda item: (_operation_order(item.operation_type), item.operation_id),
                    )[0]
                    ranking = scheduler.rank([operation], {operation.operation_id: signals[operation.operation_id]})
                scheduler_context = {
                    "ready_operation_count": len(operations),
                    "scheduler_active": scheduler_active,
                    "candidate_operations": [
                        {
                            "operation_id": row.operation.operation_id,
                            "operation": row.operation.operation_type.value,
                            "target_id": row.operation.target_id,
                            "branch_id": row.operation.branch_id,
                            "raw_signals": row.raw_signals.__dict__,
                            "normalized_signals": row.normalized_signals.__dict__,
                            "utility": row.utility,
                        }
                        for row in ranking
                    ],
                    "selected_operation": operation.operation_id,
                }
                try:
                    graph, progressed = self._execute(
                        example, graph, operation, controller, generator, verifier, editor,
                        retrieval_trace, reasoning_trace, all_hits, summaries,
                        candidate_failures, editor_events, scheduler_context, budget,
                    )
                except (ProviderRefusalError, StructuredOutputError) as exc:
                    self._record_model_failure(
                        reasoning_trace, budget, exc, operation.operation_type.value.lower(),
                        operation.target_id, operation.branch_id,
                    )
                    progressed = False
                    if operation.operation_type == OperationType.VERIFY:
                        # Independent deterministic components are a conservative
                        # auditable fallback for one refused soft-verifier call.
                        actual, summary = verifier.deterministic_propose(
                            graph, operation.target_id, operation.branch_id,
                            f"{operation.operation_id}_model_failure_fallback",
                        )
                        if actual is not None:
                            actual.reason = "model_failure_deterministic_verification_fallback"
                            graph = self._apply(
                                controller, graph, actual, reasoning_trace, [], scheduler_context,
                            )
                            summaries[(operation.target_id, operation.branch_id)] = summary
                            progressed = True
                if not progressed and operation.operation_type not in {OperationType.EXPAND, OperationType.REVISE}:
                    candidate_failures.add((operation.target_id, operation.branch_id))
                decision_key = (graph.state_hash(), operation.operation_id)
                if not progressed and failed_decision == decision_key:
                    # A proposal that cannot mutate the same graph twice is a
                    # controller stall, not useful reasoning. Terminalize instead
                    # of spinning without consuming the graph-operation budget.
                    break
                failed_decision = decision_key if not progressed else None

            if not graph.answers() and not terminal_attempted:
                try:
                    graph, _ = self._terminalize(graph, controller, terminal, reasoning_trace, budget)
                except (ProviderRefusalError, StructuredOutputError) as exc:
                    self._record_model_failure(
                        reasoning_trace, budget, exc, "terminal", "subgoal_root", "*",
                    )
            prediction = self._prediction(example, graph, all_hits, budget, None)
            if (
                policy_iterations >= self.config.max_policy_iterations
                and prediction.status == RunStatus.ABSTAIN
            ):
                prediction.stop_reason = "dynamic_policy_iteration_budget_exhausted"
        except StructuredOutputError as exc:
            self._record_structured_failure(
                reasoning_trace, budget, exc, "unhandled", "", "",
            )
            prediction = self._prediction(example, graph, all_hits, budget, exc)
            if prediction.status == RunStatus.ABSTAIN:
                prediction.stop_reason = "dynamic_structured_output_failure_without_grounded_answer"
        except InfrastructureError as exc:
            budget.record_infrastructure_failure(exc)
            prediction = self._prediction(
                example, graph, all_hits, budget, exc,
                status=RunStatus.INFRASTRUCTURE_FAILURE,
                stop_reason="dynamic_infrastructure_failure",
            )
        except ProviderRefusalError as exc:
            # Defensive boundary: operation-level refusals are handled above, but
            # a future call site must still abstain rather than misreport infra.
            self._record_refusal(reasoning_trace, budget, exc, "unhandled", "", "")
            prediction = self._prediction(example, graph, all_hits, budget, exc)
            if prediction.status == RunStatus.ABSTAIN:
                prediction.stop_reason = "dynamic_provider_refusal_without_grounded_answer"
        except BudgetExceeded:
            # No further provider call. A direct terminal claim can still be turned
            # into an AnswerNode deterministically.
            terminal = GraphGroundedTerminalReasoner(self.llm, budget, self.config)
            direct, _ = terminal.direct_operations(
                graph, graph.active_branches(), f"op_{graph.step + 1:04d}_budget_answer",
            )
            for operation in direct:
                if len(graph.operation_history) >= graph.limits.max_graph_operations:
                    break
                graph = self._apply(controller, graph, operation, reasoning_trace, [], None)
            prediction = self._prediction(example, graph, all_hits, budget, None)
            if prediction.status == RunStatus.ABSTAIN:
                prediction.stop_reason = "dynamic_budget_exhausted_without_grounded_answer"
        except GraphBudgetExceeded as exc:
            terminal = GraphGroundedTerminalReasoner(self.llm, budget, self.config)
            direct, _ = terminal.direct_operations(
                graph, graph.active_branches(), f"op_{graph.step + 1:04d}_graph_budget_answer",
            )
            for operation in direct:
                if len(graph.operation_history) >= graph.limits.max_graph_operations:
                    break
                graph = self._apply(controller, graph, operation, reasoning_trace, [], None)
            prediction = self._prediction(example, graph, all_hits, budget, exc)
            if prediction.status == RunStatus.ABSTAIN:
                prediction.stop_reason = "dynamic_graph_budget_exhausted"
        except (GraphInvariantError, TypeError, ValueError, KeyError) as exc:
            prediction = self._prediction(
                example, graph, all_hits, budget, exc,
                status=RunStatus.INFRASTRUCTURE_FAILURE,
                stop_reason="dynamic_invalid_graph_or_payload",
            )
        usage.wall_seconds = time.perf_counter() - started
        prediction.usage = usage
        return prediction, retrieval_trace, reasoning_trace

    @staticmethod
    def _record_refusal(trace, budget, error, stage, target_id, branch_id):
        budget.record_provider_failure(error)
        trace.append({
            "event": "provider_refusal",
            "stage": stage,
            "target_id": target_id,
            "branch_id": branch_id,
            "provider_attempts": int(getattr(error, "provider_attempts", 0)),
            "error_type": type(error).__name__,
        })

    @staticmethod
    def _record_structured_failure(trace, budget, error, stage, target_id, branch_id):
        try:
            budget.record_generation(error.generation)
        except BudgetExceeded:
            pass
        trace.append({
            "event": "structured_output_failure",
            "stage": stage,
            "target_id": target_id,
            "branch_id": branch_id,
            "finish_reason": str(getattr(error.generation, "finish_reason", "")),
            "error_type": type(error).__name__,
        })

    @classmethod
    def _record_model_failure(cls, trace, budget, error, stage, target_id, branch_id):
        if isinstance(error, StructuredOutputError):
            cls._record_structured_failure(trace, budget, error, stage, target_id, branch_id)
        else:
            cls._record_refusal(trace, budget, error, stage, target_id, branch_id)

    def _new_graph(self, question: str) -> DynamicReasoningHypergraph:
        graph = DynamicReasoningHypergraph(question, GraphLimits(
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
        return graph

    def _ready_operations(self, graph, summaries, candidate_failures, editor_events, budget):
        operations: list[GraphOperation] = []
        signals: dict[str, OperationSignals] = {}
        for branch in sorted(graph.active_branches(), key=lambda value: value.branch_id):
            for subgoal in sorted(graph.subgoals(), key=lambda value: value.node_id):
                if subgoal.node_id in branch.completed_subgoals:
                    continue
                if not all(dependency in branch.assignments for dependency in subgoal.dependencies):
                    continue
                key = (subgoal.node_id, branch.branch_id)
                question, dependency_claim_ids = self._instantiate(graph, subgoal, branch)
                evidence = graph.evidence(subgoal.node_id, branch.branch_id)
                candidates = [
                    claim for claim in graph.claims(subgoal.node_id, branch.branch_id)
                    if claim.status not in {CandidateStatus.ARCHIVED, CandidateStatus.INVALID}
                ]
                retrieval_rounds = len({node.retrieval_query for node in evidence})
                if not evidence:
                    empty_event = (branch.branch_id, subgoal.node_id, "no_retrieval_evidence")
                    if key in candidate_failures:
                        if self.config.enable_adaptive_planning and empty_event not in editor_events:
                            operation = self._placeholder(graph, OperationType.EXPAND, subgoal, branch, {
                                "event": "no_retrieval_evidence",
                            })
                        else:
                            continue
                    else:
                        retrieval_query = question
                        if subgoal.node_id == "subgoal_root" and normalize_text(question) != normalize_text(graph.question):
                            # Progressive rewriting supplies the resolved bridge;
                            # retaining the original objective prevents relation terms
                            # from being lost when the rewrite is deliberately short.
                            retrieval_query = f"{question} Original objective: {graph.question}"
                        operation = self._placeholder(graph, OperationType.RETRIEVE, subgoal, branch, {
                            "query": retrieval_query, "dependency_claim_ids": dependency_claim_ids,
                        })
                elif not candidates and key not in candidate_failures:
                    operation = self._placeholder(graph, OperationType.BRANCH, subgoal, branch, {
                        "mode": "generate_candidates", "question": question,
                        "dependency_claim_ids": dependency_claim_ids,
                    })
                elif any(candidate.status == CandidateStatus.PROPOSED for candidate in candidates):
                    operation = self._placeholder(graph, OperationType.VERIFY, subgoal, branch, {
                        "question": question,
                    })
                elif candidates:
                    summary = summaries.get(key) or _summary_from_candidates(candidates)
                    prunable = [
                        candidate.node_id for candidate in candidates
                        if candidate.status != CandidateStatus.COMMITTED
                        and candidate_prune_value(candidate) < self.config.prune_value_threshold
                    ]
                    if prunable and len(candidates) - len(prunable) >= 1:
                        operation = self._placeholder(graph, OperationType.PRUNE, subgoal, branch, {
                            "candidate_ids": prunable,
                        })
                        operations.append(operation)
                        signals[operation.operation_id] = self._operation_signals(graph, operation, subgoal, branch)
                        continue
                    decision = decide_candidate_set(
                        candidates, summary, self.config,
                        budget_fallback=(
                            not budget.can_call(
                                self.config.candidate_set_max_tokens,
                                estimated_prompt_tokens=200,
                            )
                        ),
                    )
                    if decision.action == "commit" or not self.config.enable_candidate_preservation:
                        chosen = decision.candidate_ids[0] if decision.candidate_ids else max(
                            candidates, key=lambda value: value.score.absolute_support,
                        ).node_id
                        operation = self._placeholder(graph, OperationType.COMMIT, subgoal, branch, {
                            "candidate_id": chosen,
                        })
                    elif decision.action == "branch" and self.config.enable_candidate_preservation:
                        available_slots = graph.limits.max_active_branches - (len(graph.active_branches()) - 1)
                        candidate_ids = decision.candidate_ids[:available_slots]
                        if len(candidate_ids) >= 2:
                            operation = self._placeholder(graph, OperationType.BRANCH, subgoal, branch, {
                                "mode": "assignments", "candidate_ids": candidate_ids,
                            })
                        else:
                            operation = self._placeholder(graph, OperationType.COMMIT, subgoal, branch, {
                                "candidate_id": decision.candidate_ids[0],
                            })
                    elif retrieval_rounds < self.config.max_retrieval_rounds_per_subgoal and graph.retrieval_calls < graph.limits.max_retrieval_calls:
                        alternatives = ", ".join(candidate.value for candidate in candidates[:3])
                        disambiguation = f"{question} Distinguish the evidence for these alternatives: {alternatives}."
                        operation = self._placeholder(graph, OperationType.RETRIEVE, subgoal, branch, {
                            "query": disambiguation, "dependency_claim_ids": dependency_claim_ids,
                        })
                    elif self.config.enable_adaptive_planning:
                        event = (branch.branch_id, subgoal.node_id, "high_uncertainty")
                        if event not in editor_events:
                            operation = self._placeholder(graph, OperationType.EXPAND, subgoal, branch, {
                                "event": "high_uncertainty",
                            })
                        else:
                            chosen = max(candidates, key=lambda value: value.score.absolute_support).node_id
                            operation = self._placeholder(graph, OperationType.COMMIT, subgoal, branch, {
                                "candidate_id": chosen,
                            })
                    else:
                        chosen = max(candidates, key=lambda value: value.score.absolute_support).node_id
                        operation = self._placeholder(graph, OperationType.COMMIT, subgoal, branch, {
                            "candidate_id": chosen,
                        })
                else:
                    event = (branch.branch_id, subgoal.node_id, "no_viable_candidate")
                    if self.config.enable_adaptive_planning and event not in editor_events:
                        operation = self._placeholder(graph, OperationType.EXPAND, subgoal, branch, {
                            "event": "no_viable_candidate",
                        })
                    else:
                        continue
                operations.append(operation)
                signals[operation.operation_id] = self._operation_signals(graph, operation, subgoal, branch)
        return operations, signals

    def _placeholder(self, graph, kind, subgoal, branch, payload):
        return GraphOperation(
            f"op_{graph.step + 1:04d}_{kind.value.lower()}_{subgoal.node_id}_{branch.branch_id}",
            kind, subgoal.node_id, [], branch.branch_id, payload,
            "operation_policy", "deterministic_operation_controller",
        )

    def _operation_signals(self, graph, operation, subgoal, branch):
        children = sum(subgoal.node_id in node.dependencies for node in graph.subgoals())
        distance = _distance_to_root(graph, subgoal.node_id)
        cost = {
            OperationType.RETRIEVE: 0.25,
            OperationType.COMMIT: 0.05,
            OperationType.PRUNE: 0.02,
            OperationType.MERGE: 0.02,
            OperationType.BRANCH: 0.8,
            OperationType.VERIFY: 0.8,
            OperationType.EXPAND: 1.0,
            OperationType.REVISE: 0.4,
        }[operation.operation_type]
        growth = {
            OperationType.BRANCH: 1.0,
            OperationType.EXPAND: 0.8,
            OperationType.RETRIEVE: 0.4,
        }.get(operation.operation_type, 0.1)
        return OperationSignals(
            uncertainty_reduction=subgoal.uncertainty,
            dependency_unlock=min(1.0, children / 3.0),
            answer_impact=1.0 / max(1, distance),
            evidence_novelty=1.0 if operation.operation_type == OperationType.RETRIEVE else 0.3,
            recovery_value=1.0 if operation.operation_type in {OperationType.REVISE, OperationType.EXPAND} else 0.0,
            expected_cost=cost,
            graph_growth_risk=growth,
        )

    def _execute(
        self, example, graph, operation, controller, generator, verifier, editor,
        retrieval_trace, reasoning_trace, all_hits, summaries, candidate_failures,
        editor_events, scheduler_context, budget,
    ):
        subgoal = graph.node(operation.target_id, SubgoalNode)
        branch = graph.branches[operation.branch_id]
        if operation.operation_type == OperationType.RETRIEVE:
            query = str(operation.payload["query"])
            if graph.retrieval_calls >= graph.limits.max_retrieval_calls:
                return graph, False
            budget.record_retrieval()
            hits = self.retriever.search(query, self.config.top_k)
            all_seen = {hit.passage.passage_id for hit in all_hits}
            all_hits.extend(hit for hit in hits if hit.passage.passage_id not in all_seen)
            rows = []
            existing = {(node.passage_id, node.retrieval_query) for node in graph.evidence()}
            for hit in hits:
                if (hit.passage.passage_id, query) in existing:
                    continue
                rows.append({
                    "node_id": f"evidence_{graph.step + 1}_{len(rows) + 1}_{_safe(hit.passage.passage_id)}",
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
                operation.operation_id, OperationType.RETRIEVE, subgoal.node_id,
                [], branch.branch_id, {"query": query, "evidence": rows},
                "retrieve_for_current_uncertain_state", "retriever_adapter",
                {"retrieval_calls": 1.0},
            )
            graph = self._apply(controller, graph, actual, reasoning_trace, rows, scheduler_context)
            retrieval_trace.append({
                "step": graph.step, "operation_id": actual.operation_id,
                "subgoal_id": subgoal.node_id, "branch_id": branch.branch_id,
                "query": query, "hits": [hit.to_dict() for hit in hits],
            })
            return graph, bool(rows)
        if operation.operation_type == OperationType.BRANCH and operation.payload.get("mode") == "generate_candidates":
            actual = generator.propose(
                graph, subgoal.node_id, branch.branch_id,
                str(operation.payload["question"]), list(operation.payload["dependency_claim_ids"]),
                operation.operation_id,
            )
            if actual is None:
                reasoning_trace.append({
                    "event": "candidate_generation_rejected",
                    "stage": "candidate_generation",
                    "target_id": subgoal.node_id,
                    "branch_id": branch.branch_id,
                    "diagnostics": dict(generator.last_diagnostics),
                })
                candidate_failures.add((subgoal.node_id, branch.branch_id))
                return graph, False
            graph = self._apply(controller, graph, actual, reasoning_trace, [], scheduler_context)
            return graph, True
        if operation.operation_type == OperationType.VERIFY:
            if self.config.enable_soft_verification:
                actual, summary = verifier.propose(
                    graph, subgoal.node_id, branch.branch_id,
                    str(operation.payload["question"]), operation.operation_id,
                )
            else:
                actual, summary = verifier.deterministic_propose(
                    graph, subgoal.node_id, branch.branch_id, operation.operation_id,
                )
            if actual is None:
                return graph, False
            graph = self._apply(controller, graph, actual, reasoning_trace, [], scheduler_context)
            summaries[(subgoal.node_id, branch.branch_id)] = summary
            return graph, True
        if operation.operation_type == OperationType.BRANCH:
            graph = self._apply(controller, graph, operation, reasoning_trace, [], scheduler_context)
            return graph, True
        if operation.operation_type in {OperationType.COMMIT, OperationType.PRUNE, OperationType.MERGE, OperationType.REVISE}:
            graph = self._apply(controller, graph, operation, reasoning_trace, [], scheduler_context)
            return graph, True
        if operation.operation_type == OperationType.EXPAND:
            event_name = str(operation.payload.get("event", "uncertainty"))
            event_branch = "*" if operation.payload.get("event_scope") == "graph" else branch.branch_id
            event = (event_branch, subgoal.node_id, event_name)
            editor_events.add(event)
            proposals = editor.propose(
                graph, event_name, branch,
                f"op_{graph.step + 1:04d}_editor_{_safe(subgoal.node_id)}", subgoal.node_id,
            )
            if not self.config.enable_revision:
                proposals = [value for value in proposals if value.operation_type != OperationType.REVISE]
            if not proposals:
                return graph, False
            for proposal in proposals:
                graph = self._apply(controller, graph, proposal, reasoning_trace, [], scheduler_context)
            return graph, True
        return graph, False

    def _terminalize(self, graph, controller, terminal, reasoning_trace, budget):
        active = graph.active_branches()
        direct, unresolved = terminal.direct_operations(
            graph, active, f"op_{graph.step + 1:04d}_answer_direct",
        )
        operations = direct
        if unresolved and budget.can_call(
            self.config.terminal_derivation_max_tokens,
            estimated_prompt_tokens=300, final=True,
        ):
            operations += terminal.derive_operations(
                graph, unresolved, f"op_{graph.step + 1:04d}_answer_derive",
            )
        for operation in operations:
            if len(graph.operation_history) >= graph.limits.max_graph_operations:
                break
            graph = self._apply(controller, graph, operation, reasoning_trace, [], {
                "ready_operation_count": len(operations),
                "scheduler_active": False,
                "candidate_operations": [],
                "selected_operation": operation.operation_id,
            })
        return graph, bool(graph.answers())

    def _apply(self, controller, graph, operation, trace, retrieval_rows, scheduler_context):
        before_calls = len(graph.operation_history)
        updated = controller.apply(graph, operation)
        audit = updated.operation_history[-1]
        trace.append({
            "step_id": audit.step,
            "operation": audit.operation_type.value,
            "operation_id": audit.operation_id,
            "graph_before_hash": audit.graph_before_hash,
            "graph_after_hash": audit.graph_after_hash,
            "created_nodes": audit.created_nodes,
            "updated_nodes": audit.updated_nodes,
            "pruned_nodes": audit.pruned_nodes,
            "created_hyperedges": audit.created_hyperedges,
            "branch_id": audit.branch_id,
            "candidate_scores": {
                claim.node_id: claim.score.__dict__ | {"raw": claim.score.raw.__dict__}
                for claim in updated.claims()
            },
            "uncertainty": {
                node.node_id: node.uncertainty for node in updated.subgoals()
            },
            "scheduler": scheduler_context,
            "retrieval_results": retrieval_rows,
            "revision_reason": operation.reason if operation.operation_type == OperationType.REVISE else None,
            "prune_reason": operation.reason if operation.operation_type == OperationType.PRUNE else None,
            "commit_reason": operation.reason if operation.operation_type == OperationType.COMMIT else None,
            "estimated_cost": operation.estimated_cost,
            "llm_call_count": float(operation.estimated_cost.get("llm_calls", 0.0)),
            "token_cost": float(operation.estimated_cost.get("tokens", 0.0)),
            "graph_snapshot": updated.to_dict(),
        })
        assert len(updated.operation_history) == before_calls + 1
        return updated

    @staticmethod
    def _instantiate(graph, subgoal, branch):
        question = subgoal.question_template
        dependency_claims = []
        for variable, source_subgoal in subgoal.variable_bindings.items():
            claim_id = branch.assignments.get(source_subgoal)
            if claim_id is None:
                raise GraphInvariantError(f"missing branch assignment for variable {variable}")
            claim = graph.node(claim_id, ClaimNode)
            question = question.replace(variable, claim.value)
            dependency_claims.append(claim_id)
        for dependency in subgoal.dependencies:
            claim_id = branch.assignments.get(dependency)
            if claim_id and claim_id not in dependency_claims:
                dependency_claims.append(claim_id)
        if re.search(r"\$[A-Za-z][A-Za-z0-9_]*", question):
            raise GraphInvariantError("instantiated dynamic subgoal has an unbound variable")
        return question, dependency_claims

    def _prediction(self, example, graph, hits, budget, error, status=None, stop_reason=None):
        answers = graph.answers()
        selected = max(
            answers,
            key=lambda node: (
                node.confidence * graph.branches.get(node.branch_id, BranchState("", None, {}, [], 1, BranchStatus.ARCHIVED, 0)).score,
                node.node_id,
            ),
            default=None,
        )
        if status is None:
            status = RunStatus.ANSWER if selected is not None else RunStatus.ABSTAIN
        if stop_reason is None:
            stop_reason = "dynamic_graph_grounded_answer" if selected is not None else "dynamic_no_grounded_answer"
        legacy_claims = _legacy_claims(graph)
        best = max(graph.claims(), key=lambda value: value.score.absolute_support, default=None)
        return Prediction(
            qid=example.qid,
            question=example.question,
            status=status,
            answer=selected.candidate_answer if selected is not None and status == RunStatus.ANSWER else None,
            confidence=selected.confidence if selected is not None else 0.0,
            stop_reason=stop_reason,
            best_unverified_candidate=best.value if best is not None and selected is None else None,
            rejection_reasons=[] if selected is not None else ["no_graph_grounded_answer_node"],
            claims=legacy_claims,
            plan=_legacy_plan(graph),
            retrieved=hits,
            usage=budget.usage,
            error=None if error is None else f"{type(error).__name__}: {error}",
        )


def _legacy_claims(graph: DynamicReasoningHypergraph) -> list[Claim]:
    assigned = {claim_id for branch in graph.branches.values() for claim_id in branch.assignments.values()}
    rows = []
    for node in graph.claims():
        evidence = [graph.node(value, EvidenceNode) for value in node.evidence_refs]
        rows.append(Claim(
            claim_id=node.node_id,
            subject=node.subject,
            relation=node.relation,
            object=node.value,
            answer_type=node.answer_type,
            target_slot=node.target_subgoal,
            source_document_ids=list(dict.fromkeys(value.document_id for value in evidence)),
            source_spans=[str(value) for value in node.provenance.metadata.get("source_spans", [])],
            depends_on_claim_ids=node.dependency_claim_ids,
            retrieval_score=max((value.retrieval_score for value in evidence), default=0.0),
            entailment_score=node.score.raw.entailment,
            type_score=node.score.raw.type_match,
            calibrated_confidence=node.score.absolute_support,
            status=(
                ClaimStatus.VERIFIED if node.node_id in assigned
                else ClaimStatus.REJECTED if node.status in {CandidateStatus.ARCHIVED, CandidateStatus.INVALID}
                else ClaimStatus.PROPOSED
            ),
            contradiction_ids=node.contradiction_links,
            created_step=node.created_at_step,
        ))
    return rows


def _legacy_plan(graph: DynamicReasoningHypergraph) -> ReasoningPlan:
    completed = {value for branch in graph.branches.values() for value in branch.completed_subgoals}
    slots = []
    for node in sorted(graph.subgoals(), key=lambda value: (value.created_at_step, value.node_id)):
        bindings = [VariableBinding(variable=variable, source_slot=source) for variable, source in node.variable_bindings.items()]
        slots.append(ReasoningSlot(
            slot_id=node.node_id,
            subquestion_template=node.question_template,
            answer_type=node.answer_type,
            dependencies=node.dependencies,
            variable_bindings=bindings,
            output_variable=f"${node.node_id}_answer",
            terminal=node.terminal,
            status=SlotStatus.COMPLETE if node.node_id in completed else SlotStatus.PENDING,
            confidence=node.confidence,
            bound_question=node.instantiated_question,
        ))
    return ReasoningPlan(graph.question, slots, "dynamic_hypergraph", "dynamic_hypergraph_tdca")


def _summary_from_candidates(candidates: list[ClaimNode]) -> CandidateSetSummary:
    ranked = sorted(candidates, key=lambda value: (-value.score.relative_weight, value.node_id))
    margin = ranked[0].score.relative_weight - ranked[1].score.relative_weight if len(ranked) > 1 else 1.0
    return CandidateSetSummary(
        entropy=max((value.score.set_entropy for value in candidates), default=1.0),
        top_margin=max(0.0, margin),
        top_candidate_id=ranked[0].node_id if ranked else None,
        candidate_count=len(ranked),
    )


def _distance_to_root(graph: DynamicReasoningHypergraph, node_id: str) -> int:
    if node_id == "subgoal_root":
        return 1
    reverse = {node.node_id: [] for node in graph.subgoals()}
    for node in graph.subgoals():
        for dependency in node.dependencies:
            reverse.setdefault(dependency, []).append(node.node_id)
    frontier = [(node_id, 1)]
    visited = set()
    while frontier:
        current, distance = frontier.pop(0)
        if current == "subgoal_root":
            return distance
        if current in visited:
            continue
        visited.add(current)
        frontier.extend((value, distance + 1) for value in reverse.get(current, []))
    return graph.limits.max_graph_depth


def _operation_order(value: OperationType) -> int:
    order = {
        OperationType.REVISE: 0,
        OperationType.VERIFY: 1,
        OperationType.COMMIT: 2,
        OperationType.RETRIEVE: 3,
        OperationType.BRANCH: 4,
        OperationType.MERGE: 5,
        OperationType.PRUNE: 6,
        OperationType.EXPAND: 7,
    }
    return order[value]


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)[:48]
