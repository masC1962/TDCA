from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from ..budget import Budget
from ..dynamic.graph import (
    AnswerStatus,
    CandidateStatus,
    ClaimNode,
    EvidenceNode,
    GraphOperation,
    OperationType,
    SubgoalNode,
    SubgoalStatus,
)
from ..utils import stable_hash
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2, OperationFeedbackStats
from .obligations import (
    graph_local_operation_value,
    operation_conditioned_closure_value,
    operation_obligation_targets,
)


@dataclass(frozen=True)
class EVCSignals:
    graph_heat: float = 0.0
    uncertainty_reduction: float = 0.0
    answer_impact: float = 0.0
    dependency_unlock: float = 0.0
    evidence_novelty: float = 0.0
    recovery_value: float = 0.0
    observed_value: float = 0.5
    expected_cost: float = 0.0
    graph_growth_risk: float = 0.0
    failure_cooldown: float = 0.0
    terminal_gap: float = 0.0
    terminal_proximity: float = 0.0
    expected_call_cost: float = 0.0
    expected_token_cost: float = 0.0
    expected_retrieval_cost: float = 0.0
    retrieval_saturation: float = 0.0
    proof_gap_reducibility: float = 0.0
    feasibility_unlock: float = 0.0
    obligation_closure: float = 0.0
    terminal_reachability: float = 0.0
    missing_premise_reduction: float = 0.0
    candidate_reachability: float = 0.0
    evidence_path: float = 0.0
    dead_end_risk: float = 0.0
    absolute_graph_risk: float = 0.0
    obligation_importance: float = 0.0
    operation_closure_probability: float = 0.0
    expected_obligation_delta: float = 0.0
    obligation_terminal_return: float = 0.0
    operation_redundancy: float = 0.0


@dataclass(frozen=True)
class ComputationPacket:
    allocation_id: str
    operation: GraphOperation
    target_region: tuple[str, ...]
    predicted_evc: float
    raw: EVCSignals
    normalized: EVCSignals
    requested_budget: dict[str, int]
    remaining_global_budget: dict[str, int]
    allocator_mode: str = "adaptive_evc"
    operation_family: str = ""
    region_key: str = ""
    pre_state_summary: dict[str, float] = field(default_factory=dict)
    feedback_prior: dict[str, float] = field(default_factory=dict)
    fidelity_level: str = "medium"
    fidelity_fraction: float = 0.65
    predicted_immediate_utility: float = 0.0
    predicted_delayed_proof_return: float = 0.0
    predicted_normalized_cost: float = 0.0
    predicted_gross_opportunity: float = 0.0
    target_obligation_ids: tuple[str, ...] = ()
    obligation_estimate: dict = field(default_factory=dict)
    predicted_marginal_evc: float = 0.0
    predicted_provider_calls: int = 0
    critical_obligation_reserve: dict[str, int] = field(default_factory=dict)
    reserve_feasible: bool = True

    def trace(self) -> dict:
        raw = dict(self.raw.__dict__)
        normalized = dict(self.normalized.__dict__)
        if not self.obligation_estimate:
            for name in (
                "obligation_importance", "operation_closure_probability",
                "expected_obligation_delta", "obligation_terminal_return",
                "operation_redundancy",
            ):
                raw.pop(name, None)
                normalized.pop(name, None)
        payload = {
            "allocation_id": self.allocation_id,
            "operation_id": self.operation.operation_id,
            "operation_family": self.operation_family,
            "region_key": self.region_key,
            "target_region": list(self.target_region),
            "predicted_evc": self.predicted_evc,
            "evc_components_raw": raw,
            "evc_components_normalized": normalized,
            "requested_budget": dict(self.requested_budget),
            "remaining_global_budget": dict(self.remaining_global_budget),
            "allocator_mode": self.allocator_mode,
            "pre_state_summary": dict(self.pre_state_summary),
            "feedback_prior": dict(self.feedback_prior),
            "fidelity_level": self.fidelity_level,
            "fidelity_fraction": self.fidelity_fraction,
            "predicted_immediate_utility": self.predicted_immediate_utility,
            "predicted_delayed_proof_return": self.predicted_delayed_proof_return,
            "predicted_normalized_cost": self.predicted_normalized_cost,
            "predicted_gross_opportunity": self.predicted_gross_opportunity,
            "target_obligation_ids": list(self.target_obligation_ids),
            "obligation_estimate": deepcopy(self.obligation_estimate),
            "predicted_marginal_evc": self.predicted_marginal_evc,
            "predicted_provider_calls": self.predicted_provider_calls,
            "critical_obligation_reserve": dict(self.critical_obligation_reserve),
            "reserve_feasible": self.reserve_feasible,
        }
        if not self.obligation_estimate:
            for name in (
                "obligation_estimate", "predicted_marginal_evc",
                "predicted_provider_calls", "critical_obligation_reserve",
                "reserve_feasible",
            ):
                payload.pop(name, None)
        return payload


@dataclass
class AdaptiveComputationAllocator:
    config: DynamicV2ResearchConfig
    last_packets: list[ComputationPacket] = field(default_factory=list)
    allocation_serial: int = 0

    def allocate(
        self,
        graph: DynamicReasoningHypergraphV2,
        operations: list[GraphOperation],
        budget: Budget,
    ) -> list[ComputationPacket]:
        if not operations:
            self.last_packets = []
            return []
        mode = self.config.allocator_mode
        remaining = {
            "llm_calls": max(0, budget.max_llm_calls - budget.usage.llm_calls),
            "tokens": max(0, budget.max_total_tokens - budget.usage.total_tokens),
            "retrieval_calls": max(0, graph.limits.max_retrieval_calls - graph.retrieval_calls),
            "graph_operations": max(0, graph.limits.max_graph_operations - len(graph.operation_history)),
        }
        expanded: list[tuple[int, GraphOperation, str, float, EVCSignals]] = []
        base_rows = [self._signals(graph, operation, budget) for operation in operations]
        normalized_base_rows = (
            self._normalize_choice_conditioned_signals(base_rows, operations)
            if self.config.choice_conditioned_evc
            else self._normalize_base_signals(base_rows)
        )
        if self.config.absolute_resource_cost:
            # Cost is an absolute resource fraction.  It must not change when a
            # dominated or irrelevant action is inserted into the ready set.
            absolute_names = {
                "expected_cost", "expected_call_cost", "expected_token_cost",
                "expected_retrieval_cost", "absolute_graph_risk",
                "obligation_closure", "terminal_reachability",
                "missing_premise_reduction", "candidate_reachability",
                "evidence_path", "dead_end_risk",
                "obligation_importance", "operation_closure_probability",
                "expected_obligation_delta", "obligation_terminal_return",
                "operation_redundancy",
            }
            normalized_base_rows = [
                EVCSignals(**{
                    name: (
                        float(getattr(base_rows[index], name))
                        if name in absolute_names
                        else float(getattr(row, name))
                    )
                    for name in EVCSignals.__dataclass_fields__
                })
                for index, row in enumerate(normalized_base_rows)
            ]
        for index, operation in enumerate(operations):
            base = base_rows[index]
            levels = self._fidelity_options(operation)
            for name, fraction in levels:
                call_scale, token_scale = self._fidelity_resource_scales(
                    operation, fraction,
                )
                expanded.append((
                    index, operation, name, fraction,
                    self._fidelity_signals(
                        base, fraction,
                        bounded_opportunity=self.config.horizon_aware_evc,
                        absolute_cost=self.config.absolute_resource_cost,
                        call_scale=call_scale,
                        token_scale=token_scale,
                    ),
                ))
        names = tuple(EVCSignals.__dataclass_fields__)
        if self.config.multi_resource_evc:
            normalized_expanded = []
            for index, operation in enumerate(operations):
                for _, fraction in self._fidelity_options(operation):
                    call_scale, token_scale = self._fidelity_resource_scales(
                        operation, fraction,
                    )
                    normalized_expanded.append(
                        self._fidelity_signals(
                            normalized_base_rows[index], fraction, clamp=True,
                            bounded_opportunity=self.config.horizon_aware_evc,
                            absolute_cost=self.config.absolute_resource_cost,
                            call_scale=call_scale,
                            token_scale=token_scale,
                        )
                    )
        else:
            raw_rows = [row[4] for row in expanded]
            columns = {
                name: _minmax([float(getattr(row, name)) for row in raw_rows])
                for name in names
            }
            normalized_expanded = [
                EVCSignals(**{name: columns[name][index] for name in names})
                for index in range(len(expanded))
            ]
        packets: list[tuple[int, ComputationPacket]] = []
        serial_start = self.allocation_serial
        previous_fidelity: dict[str, tuple[float, float]] = {}
        for packet_index, (index, operation, fidelity_name, fidelity_fraction, raw_row) in enumerate(expanded):
            normalized = normalized_expanded[packet_index]
            if mode == "adaptive_evc":
                if self.config.horizon_aware_evc:
                    immediate, delayed, normalized_cost = self._horizon_scores(
                        normalized, operation_family(operation),
                    )
                    evc = max(0.0, (
                        self.config.evc_immediate_horizon_weight * immediate
                        + self.config.evc_delayed_horizon_weight * delayed
                        - normalized_cost
                    ))
                    gross = _unit(
                        self.config.evc_immediate_horizon_weight * immediate
                        + self.config.evc_delayed_horizon_weight * delayed
                    )
                else:
                    evc = self._adaptive_evc(normalized)
                    immediate, delayed, normalized_cost = evc, 0.0, normalized.expected_cost
                    gross = _unit(evc + normalized_cost)
            elif mode == "uniform":
                evc = 0.5
                immediate, delayed, normalized_cost = 0.5, 0.5, normalized.expected_cost
                gross = 0.5
            else:
                evc = max(0.1, 1.0 - 0.1 * _fixed_priority(operation.operation_type))
                immediate, delayed, normalized_cost = evc, 0.0, normalized.expected_cost
                gross = _unit(evc + normalized_cost)
            region = tuple(dict.fromkeys([
                operation.target_id,
                *(
                    [operation.branch_id]
                    if self.config.operation_conditioned_obligation_closure else []
                ),
                *operation.source_ids,
            ]))
            family = operation_family(operation)
            rkey = operation_region_key(operation)
            prior = feedback_prior(
                graph, family, rkey,
                operation_coarse_region_key(operation),
                use_hierarchical=self.config.hierarchical_within_question_feedback,
            )
            requested = self._budget_packet(
                operation, max(
                    raw_row.graph_heat,
                    0.75 * raw_row.terminal_gap,
                    raw_row.terminal_proximity,
                ), remaining, prior,
                fidelity_fraction=fidelity_fraction,
            )
            estimate = (
                operation_conditioned_closure_value(graph, operation)
                if self.config.operation_conditioned_obligation_closure else {}
            )
            previous_gross, previous_cost = previous_fidelity.get(
                operation.operation_id, (0.0, 0.0),
            )
            marginal_evc = float(gross - previous_gross - (
                normalized_cost - previous_cost
            ))
            previous_fidelity[operation.operation_id] = (gross, normalized_cost)
            reserve = self._critical_obligation_reserve(graph, operation)
            reserve_feasible = (
                remaining["llm_calls"] - requested.get("llm_calls", 0)
                >= reserve["llm_calls"]
                and remaining["tokens"]
                - requested.get("max_tokens", 0) * max(
                    1, requested.get("verification_samples", 1),
                )
                >= reserve["tokens"]
            )
            if (
                fidelity_name == "high"
                and self.config.marginal_fidelity_evc_gate
                and (marginal_evc <= 0.0 or not reserve_feasible)
            ):
                continue
            packets.append((index, ComputationPacket(
                allocation_id=f"allocation_{serial_start + packet_index + 1:06d}",
                operation=operation,
                target_region=region,
                predicted_evc=evc,
                raw=raw_row,
                normalized=normalized,
                requested_budget=requested,
                remaining_global_budget=dict(remaining),
                allocator_mode=mode,
                operation_family=family,
                region_key=rkey,
                pre_state_summary=summarize_operation_region(graph, operation),
                feedback_prior=prior,
                fidelity_level=fidelity_name,
                fidelity_fraction=fidelity_fraction,
                predicted_immediate_utility=_unit(immediate),
                predicted_delayed_proof_return=_unit(delayed),
                predicted_normalized_cost=_unit(normalized_cost),
                predicted_gross_opportunity=_unit(gross),
                target_obligation_ids=tuple(operation_obligation_targets(
                    graph, operation,
                    strict=self.config.operation_conditioned_obligation_closure,
                )),
                obligation_estimate=estimate,
                predicted_marginal_evc=marginal_evc,
                predicted_provider_calls=int(requested.get("llm_calls", 0)),
                critical_obligation_reserve=reserve,
                reserve_feasible=reserve_feasible,
            )))
        self.allocation_serial += len(expanded)
        if mode == "adaptive_evc":
            packets.sort(key=lambda row: (
                -row[1].predicted_evc,
                row[1].operation.operation_type.value,
                row[1].operation.operation_id,
            ))
        elif mode == "fixed_order":
            packets.sort(key=lambda row: (
                _fixed_priority(row[1].operation.operation_type),
                row[1].operation.operation_id,
            ))
        else:
            packets.sort(key=lambda row: row[0])
        self.last_packets = [packet for _, packet in packets]
        return self.last_packets

    def _fidelity_options(self, operation: GraphOperation) -> list[tuple[str, float]]:
        mode = str(operation.payload.get("mode", ""))
        if operation.operation_type == OperationType.BRANCH and mode == "extract_typed":
            # The smoke trace showed that low/medium JSON caps truncate atomic
            # claim coverage. A full schema is the minimum viable extraction;
            # adaptation occurs in passage/span activation and candidate count.
            return [("high", 1.0)]
        if operation.operation_type == OperationType.MERGE and mode == "validate_join":
            # JOIN validation has a compact but irreducible audit schema.
            return [("high", 1.0)]
        adjustable = operation.operation_type in {
            OperationType.RETRIEVE, OperationType.BRANCH, OperationType.VERIFY,
            OperationType.MERGE, OperationType.EXPAND,
        }
        if self.config.allocator_mode != "adaptive_evc" or not adjustable:
            return [("medium", self.config.allocation_mid_token_fraction)]
        rows = [
            ("low", self.config.allocation_min_token_fraction),
            ("medium", self.config.allocation_mid_token_fraction),
            ("high", 1.0),
        ]
        return rows[: self.config.allocation_fidelity_levels]

    @staticmethod
    def _fidelity_signals(
        base: EVCSignals, fraction: float, *, clamp: bool = False,
        bounded_opportunity: bool = False,
        absolute_cost: bool = False,
        call_scale: float | None = None,
        token_scale: float | None = None,
    ) -> EVCSignals:
        # Score *marginal* progress rather than total progress. Diminishing gains
        # make a cheap pass preferable until graph heat/impact justifies deeper
        # compute; failed cheap passes can then expose a new high-value action.
        fraction = max(0.1, min(1.0, float(fraction)))
        gain = fraction ** 0.5
        efficiency = gain / fraction
        row = EVCSignals(
            graph_heat=base.graph_heat * fraction,
            uncertainty_reduction=base.uncertainty_reduction * efficiency,
            answer_impact=base.answer_impact * gain,
            dependency_unlock=base.dependency_unlock * efficiency,
            evidence_novelty=base.evidence_novelty * efficiency,
            recovery_value=base.recovery_value * efficiency,
            observed_value=base.observed_value * efficiency,
            expected_cost=base.expected_cost * fraction,
            graph_growth_risk=base.graph_growth_risk * fraction,
            failure_cooldown=base.failure_cooldown,
            terminal_gap=base.terminal_gap * gain,
            terminal_proximity=base.terminal_proximity * gain,
            expected_call_cost=base.expected_call_cost * (
                call_scale if call_scale is not None
                else 1.0 if absolute_cost else (0.5 + 0.5 * fraction)
            ),
            expected_token_cost=base.expected_token_cost * (
                token_scale if token_scale is not None else fraction
            ),
            expected_retrieval_cost=base.expected_retrieval_cost,
            retrieval_saturation=base.retrieval_saturation,
            proof_gap_reducibility=base.proof_gap_reducibility * (
                gain if bounded_opportunity else efficiency
            ),
            feasibility_unlock=base.feasibility_unlock * (
                gain if bounded_opportunity else efficiency
            ),
            obligation_closure=base.obligation_closure * gain,
            terminal_reachability=base.terminal_reachability,
            missing_premise_reduction=base.missing_premise_reduction * gain,
            candidate_reachability=base.candidate_reachability * gain,
            evidence_path=base.evidence_path * gain,
            dead_end_risk=base.dead_end_risk,
            absolute_graph_risk=base.absolute_graph_risk * fraction,
            obligation_importance=base.obligation_importance,
            operation_closure_probability=base.operation_closure_probability,
            expected_obligation_delta=base.expected_obligation_delta * gain,
            obligation_terminal_return=base.obligation_terminal_return,
            operation_redundancy=base.operation_redundancy,
        )
        if not clamp and not bounded_opportunity:
            return row
        if not clamp:
            values = dict(row.__dict__)
            values["proof_gap_reducibility"] = _unit(values["proof_gap_reducibility"])
            values["feasibility_unlock"] = _unit(values["feasibility_unlock"])
            return EVCSignals(**values)
        return EVCSignals(**{
            name: _unit(float(getattr(row, name)))
            for name in EVCSignals.__dataclass_fields__
        })

    @staticmethod
    def _normalize_base_signals(rows: list[EVCSignals]) -> list[EVCSignals]:
        """Normalize components across operations, before fidelity expansion.

        This removes the v2.2 artifact where low/medium/high copies of a single
        action created their own comparison range and therefore looked like
        distinct reasoning opportunities.  Constant columns retain their
        absolute [0,1] meaning rather than collapsing to an arbitrary midpoint.
        """
        if not rows:
            return []
        names = tuple(EVCSignals.__dataclass_fields__)
        columns = {
            name: _minmax([float(getattr(row, name)) for row in rows])
            for name in names
        }
        return [
            EVCSignals(**{name: columns[name][index] for name in names})
            for index in range(len(rows))
        ]

    @staticmethod
    def _normalize_choice_conditioned_signals(
        rows: list[EVCSignals], operations: list[GraphOperation],
    ) -> list[EVCSignals]:
        """Blend global and within-family scales for a real ready-set choice.

        A family-local comparison prevents an expensive operation family from
        winning merely because its raw signal range differs.  The global half
        retains cross-family magnitude and singleton absolute semantics.
        """
        if not rows:
            return []
        names = tuple(EVCSignals.__dataclass_fields__)
        global_columns = {
            name: _minmax([float(getattr(row, name)) for row in rows])
            for name in names
        }
        families = [operation_family(operation) for operation in operations]
        family_indices = {
            family: [index for index, value in enumerate(families) if value == family]
            for family in sorted(set(families))
        }
        family_columns: dict[str, dict[str, dict[int, float]]] = {}
        for family, indices in family_indices.items():
            family_columns[family] = {}
            for name in names:
                values = [float(getattr(rows[index], name)) for index in indices]
                normalized = _minmax(values)
                family_columns[family][name] = dict(zip(indices, normalized))
        return [
            EVCSignals(**{
                name: 0.5 * global_columns[name][index]
                + 0.5 * family_columns[families[index]][name][index]
                for name in names
            })
            for index in range(len(rows))
        ]

    def _adaptive_evc(self, normalized: EVCSignals) -> float:
        value = (
            self.config.evc_weight_heat * normalized.graph_heat
            + self.config.evc_weight_uncertainty_reduction * normalized.uncertainty_reduction
            + self.config.evc_weight_answer_impact * normalized.answer_impact
            + self.config.evc_weight_unlock * normalized.dependency_unlock
            + self.config.evc_weight_novelty * normalized.evidence_novelty
            + self.config.evc_weight_recovery * normalized.recovery_value
            + self.config.evc_weight_observed_value * normalized.observed_value
            - self.config.evc_weight_cost * normalized.expected_cost
            - self.config.evc_weight_growth_risk * normalized.graph_growth_risk
            - self.config.evc_weight_failure_cooldown * normalized.failure_cooldown
            + self.config.evc_weight_terminal_gap * normalized.terminal_gap
            + self.config.evc_weight_terminal_proximity * normalized.terminal_proximity
            + self.config.evc_weight_proof_gap_reducibility
            * normalized.proof_gap_reducibility
            + self.config.evc_weight_feasibility_unlock
            * normalized.feasibility_unlock
        )
        if self.config.multi_resource_evc:
            # Explicit resources replace the old scalar expected-cost proxy.
            value += self.config.evc_weight_cost * normalized.expected_cost
            value -= (
                self.config.evc_weight_call_cost * normalized.expected_call_cost
                + self.config.evc_weight_token_cost * normalized.expected_token_cost
                + self.config.evc_weight_retrieval_cost * normalized.expected_retrieval_cost
                + self.config.evc_weight_retrieval_saturation * normalized.retrieval_saturation
            )
        return max(0.0, value)

    def _horizon_scores(
        self, normalized: EVCSignals, family: str = "",
    ) -> tuple[float, float, float]:
        """Return independent immediate, delayed and cost channels in [0,1]."""

        immediate = _weighted_mean({
            "graph_heat": (normalized.graph_heat, self.config.evc_weight_heat),
            "uncertainty_reduction": (
                normalized.uncertainty_reduction,
                self.config.evc_weight_uncertainty_reduction,
            ),
            "evidence_novelty": (
                normalized.evidence_novelty, self.config.evc_weight_novelty,
            ),
            "recovery_value": (
                normalized.recovery_value, self.config.evc_weight_recovery,
            ),
            "observed_value": (
                normalized.observed_value, self.config.evc_weight_observed_value,
            ),
            "terminal_proximity": (
                normalized.terminal_proximity,
                self.config.evc_weight_terminal_proximity,
            ),
        })
        structural_delayed = _weighted_mean({
            "answer_impact": (
                normalized.answer_impact, self.config.evc_weight_answer_impact,
            ),
            "dependency_unlock": (
                normalized.dependency_unlock, self.config.evc_weight_unlock,
            ),
            "terminal_gap": (
                normalized.terminal_gap, self.config.evc_weight_terminal_gap,
            ),
            "proof_gap_reducibility": (
                normalized.proof_gap_reducibility,
                self.config.evc_weight_proof_gap_reducibility,
            ),
            "feasibility_unlock": (
                normalized.feasibility_unlock,
                self.config.evc_weight_feasibility_unlock,
            ),
        })
        # A terminal ANSWER commit realizes value now and has no future proof
        # horizon.  Other operation families have different deterministic
        # capacities to seed causal descendants; this prior is structural, is
        # never updated across questions, and is blended with graph-state signals.
        family_delayed_capacity = {
            "commit:answer": self.config.delayed_capacity_commit_answer,
            "commit:default": self.config.delayed_capacity_commit_claim,
            "retrieve:default": self.config.delayed_capacity_retrieve,
            "verify:default": self.config.delayed_capacity_verify,
            "merge:validate_join": self.config.delayed_capacity_merge,
            "merge:derive_join": self.config.delayed_capacity_merge,
            "branch:extract_typed": self.config.delayed_capacity_extract,
            "branch:assignments": self.config.delayed_capacity_branch,
            "expand:default": self.config.delayed_capacity_expand,
            "revise:default": self.config.delayed_capacity_revise,
            "prune:default": self.config.delayed_capacity_prune,
        }.get(family, self.config.delayed_capacity_extract)
        structural_weight = float(self.config.delayed_structural_signal_weight)
        if self.config.graph_local_delayed_value:
            if self.config.operation_conditioned_obligation_closure:
                # v2.4.3.1: importance and tractability are independent raw
                # channels.  Multiplication prevents a severe but infeasible
                # obligation from receiving high delayed value.
                delayed = (
                    normalized.obligation_importance
                    * normalized.operation_closure_probability
                    * normalized.expected_obligation_delta
                    * normalized.obligation_terminal_return
                    - normalized.operation_redundancy
                )
            else:
                delayed = 0.0 if family == "commit:answer" else _weighted_mean({
                    "obligation_closure": (
                        normalized.obligation_closure,
                        self.config.graph_local_weight_obligation_closure,
                    ),
                    "terminal_reachability": (
                        normalized.terminal_reachability,
                        self.config.graph_local_weight_terminal_reachability,
                    ),
                    "missing_premise_reduction": (
                        normalized.missing_premise_reduction,
                        self.config.graph_local_weight_missing_premise_reduction,
                    ),
                    "candidate_reachability": (
                        normalized.candidate_reachability,
                        self.config.graph_local_weight_candidate_reachability,
                    ),
                    "evidence_path": (
                        normalized.evidence_path,
                        self.config.graph_local_weight_evidence_path,
                    ),
                }) * (1.0 - 0.5 * normalized.dead_end_risk)
        else:
            delayed = (
                0.0 if family == "commit:answer"
                else (
                    (1.0 - structural_weight) * family_delayed_capacity
                    + structural_weight * structural_delayed
                )
            )
        if self.config.absolute_resource_cost:
            cost = (
                self.config.absolute_cost_weight_call * normalized.expected_call_cost
                + self.config.absolute_cost_weight_token * normalized.expected_token_cost
                + self.config.absolute_cost_weight_retrieval * normalized.expected_retrieval_cost
                + self.config.absolute_cost_weight_graph_risk * normalized.absolute_graph_risk
            )
        elif self.config.multi_resource_evc:
            cost = _weighted_mean({
                "call": (
                    normalized.expected_call_cost, self.config.evc_weight_call_cost,
                ),
                "token": (
                    normalized.expected_token_cost, self.config.evc_weight_token_cost,
                ),
                "retrieval": (
                    normalized.expected_retrieval_cost,
                    self.config.evc_weight_retrieval_cost,
                ),
                "growth": (
                    normalized.graph_growth_risk,
                    self.config.evc_weight_growth_risk,
                ),
                "saturation": (
                    normalized.retrieval_saturation,
                    self.config.evc_weight_retrieval_saturation,
                ),
                "cooldown": (
                    normalized.failure_cooldown,
                    self.config.evc_weight_failure_cooldown,
                ),
            })
        else:
            cost = _weighted_mean({
                "expected": (normalized.expected_cost, self.config.evc_weight_cost),
                "growth": (
                    normalized.graph_growth_risk,
                    self.config.evc_weight_growth_risk,
                ),
                "cooldown": (
                    normalized.failure_cooldown,
                    self.config.evc_weight_failure_cooldown,
                ),
            })
        return _unit(immediate), _unit(delayed), _unit(cost)

    @staticmethod
    def attach(operation: GraphOperation, packet: ComputationPacket) -> GraphOperation:
        value = deepcopy(operation)
        value.payload = dict(value.payload)
        value.payload["_allocation"] = packet.trace()
        return value

    def _signals(
        self, graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
        budget: Budget | None = None,
    ) -> EVCSignals:
        assignment_branch = (
            operation.operation_type == OperationType.BRANCH
            and str(operation.payload.get("mode", "")) == "assignments"
        )
        # Diffusion has already aggregated neighborhood state into the target
        # and explicit sources. Using every alternative in the subgoal here lets
        # one hot distractor dominate all candidate operations and repeatedly
        # starve downstream regions.
        node_ids = list(dict.fromkeys([operation.target_id, *operation.source_ids]))
        states = [graph.belief_states[node_id] for node_id in node_ids if node_id in graph.belief_states]
        heat = max((state.computation_heat for state in states), default=0.0)
        uncertainty = max((state.uncertainty for state in states), default=1.0)
        impact = max((state.downstream_answer_impact for state in states), default=0.0)
        unlock = max((state.dependency_unlock_value for state in states), default=0.0)
        novelty = (
            0.25 if assignment_branch
            else 1.0 if operation.operation_type == OperationType.RETRIEVE
            else 0.4
        )
        recovery = 1.0 if operation.operation_type == OperationType.REVISE else 0.0
        base_cost = 0.05 if assignment_branch else {
            OperationType.COMMIT: 0.02,
            OperationType.PRUNE: 0.02,
            OperationType.MERGE: 0.55,
            OperationType.REVISE: 0.10,
            OperationType.RETRIEVE: 0.35,
            OperationType.VERIFY: 0.70,
            OperationType.BRANCH: 0.75,
            OperationType.EXPAND: 1.00,
        }[operation.operation_type]
        growth = min(
            1.0,
            len(operation.payload.get("candidate_ids", []))
            / max(1, graph.limits.max_active_branches),
        ) if assignment_branch else {
            OperationType.EXPAND: 1.0,
            OperationType.BRANCH: 0.8,
            OperationType.RETRIEVE: 0.4,
            OperationType.MERGE: 0.35,
        }.get(operation.operation_type, 0.05)
        family = operation_family(operation)
        rkey = operation_region_key(operation)
        prior = feedback_prior(
            graph, family, rkey,
            operation_coarse_region_key(operation),
            use_hierarchical=self.config.hierarchical_within_question_feedback,
        )
        if self.config.graph_local_delayed_value:
            # v2.4.3 records the hierarchical prior for diagnostics but lets
            # only exact, same-region causal outcomes affect the policy.
            exact = graph.operation_feedback.get(feedback_key(family, rkey))
            decision_success = exact.posterior_success if exact is not None else 0.5
            decision_value = exact.posterior_value if exact is not None else 0.5
            empirical_cost = (
                exact.cumulative_cost / max(1, exact.observations)
                if exact is not None else 0.0
            )
            decision_cooldown = float(
                exact is not None and graph.step < exact.cooldown_until_step
            )
        else:
            decision_success = float(prior["posterior_success"])
            decision_value = float(prior["posterior_value"])
            empirical_cost = float(prior.get("mean_cost", 0.0))
            decision_cooldown = float(prior["cooldown_active"])
        terminal_context = operation.payload.get("_terminal_context", {})
        if not isinstance(terminal_context, dict):
            terminal_context = {}
        terminal_gap = _unit(float(terminal_context.get("terminal_gap", 1.0)))
        terminal_affinity = _terminal_operation_affinity(operation, terminal_context)
        terminal_proximity = (
            1.0 - terminal_gap
            if operation.operation_type == OperationType.COMMIT else 0.0
        )
        call_demand = {
            OperationType.EXPAND: 1.0,
            OperationType.BRANCH: 0.0 if assignment_branch else 1.0,
            OperationType.VERIFY: float(
                self.config.max_independent_verifications
                if self.config.exact_fidelity_resource_accounting else 1
            ),
            OperationType.MERGE: 1.0,
        }.get(operation.operation_type, 0.0)
        token_demand = base_cost if call_demand else 0.0
        retrieval_demand = 1.0 if operation.operation_type == OperationType.RETRIEVE else 0.0
        if budget is None:
            call_pressure = token_pressure = retrieval_pressure = 0.5
        else:
            call_pressure = _shadow_price(
                budget.max_llm_calls - budget.usage.llm_calls, budget.max_llm_calls,
            )
            token_pressure = _shadow_price(
                budget.max_total_tokens - budget.usage.total_tokens, budget.max_total_tokens,
            )
            retrieval_pressure = _shadow_price(
                graph.limits.max_retrieval_calls - graph.retrieval_calls,
                graph.limits.max_retrieval_calls,
            )
        region_attempts = [
            row for row in graph.retrieval_attempt_history
            if row.target_subgoal == operation.target_id and row.branch_id == operation.branch_id
        ]
        retrieval_saturation = 0.0
        if operation.operation_type == OperationType.RETRIEVE and region_attempts:
            last = region_attempts[-1]
            round_fraction = len(region_attempts) / max(
                1, self.config.max_retrieval_rounds_per_subgoal,
            )
            zero_yield = 1.0 if last.new_evidence_count == 0 else 0.0
            yield_fraction = 1.0 - min(
                1.0, last.new_evidence_count / max(1, last.allocated_top_k),
            )
            retrieval_saturation = _unit(max(round_fraction, zero_yield, yield_fraction))
        proof_gap_reducibility = (
            _unit(float(operation.payload.get("proof_gap_reducibility", 0.0)))
            if self.config.choice_conditioned_evc else 0.0
        )
        feasibility_unlock = (
            _unit(float(operation.payload.get("feasibility_unlock", 0.0)))
            if self.config.choice_conditioned_evc else 0.0
        )
        local = (
            graph_local_operation_value(graph, operation)
            if self.config.graph_local_delayed_value else {}
        )
        closure = (
            operation_conditioned_closure_value(graph, operation)
            if self.config.operation_conditioned_obligation_closure else {}
        )
        if self.config.absolute_resource_cost:
            max_token_demand = 0.0 if call_demand <= 0.0 else float({
                OperationType.EXPAND: self.config.graph_editor_max_tokens,
                OperationType.BRANCH: self.config.typed_extraction_max_tokens,
                OperationType.VERIFY: self.config.soft_verifier_max_tokens,
                OperationType.MERGE: self.config.join_validation_max_tokens,
            }.get(operation.operation_type, 0.0))
            if (
                self.config.exact_fidelity_resource_accounting
                and operation.operation_type == OperationType.VERIFY
            ):
                max_token_demand *= call_demand
            call_pressure = _absolute_resource_fraction(
                call_demand, self.config.max_llm_calls,
                (
                    self.config.max_llm_calls
                    if budget is None
                    else budget.max_llm_calls - budget.usage.llm_calls
                ),
                self.config.absolute_cost_scarcity_max_multiplier,
            )
            token_pressure = _absolute_resource_fraction(
                max_token_demand, self.config.max_total_tokens,
                (
                    self.config.max_total_tokens
                    if budget is None
                    else budget.max_total_tokens - budget.usage.total_tokens
                ),
                self.config.absolute_cost_scarcity_max_multiplier,
            )
            retrieval_pressure = _absolute_resource_fraction(
                retrieval_demand, graph.limits.max_retrieval_calls,
                graph.limits.max_retrieval_calls - graph.retrieval_calls,
                self.config.absolute_cost_scarcity_max_multiplier,
            )
        return EVCSignals(
            graph_heat=heat,
            uncertainty_reduction=uncertainty * _operation_reduction(operation.operation_type),
            answer_impact=impact,
            dependency_unlock=unlock,
            evidence_novelty=novelty * max(0.2, decision_success),
            recovery_value=recovery,
            observed_value=decision_value,
            expected_cost=max(base_cost, empirical_cost),
            graph_growth_risk=growth,
            failure_cooldown=decision_cooldown,
            terminal_gap=terminal_gap * terminal_affinity,
            terminal_proximity=terminal_proximity,
            expected_call_cost=(
                call_pressure if self.config.exact_fidelity_resource_accounting
                else call_demand * call_pressure
            ),
            expected_token_cost=(
                token_pressure if self.config.exact_fidelity_resource_accounting
                else token_demand * token_pressure
            ),
            expected_retrieval_cost=(
                retrieval_pressure if self.config.exact_fidelity_resource_accounting
                else retrieval_demand * retrieval_pressure
            ),
            retrieval_saturation=retrieval_saturation,
            proof_gap_reducibility=proof_gap_reducibility,
            feasibility_unlock=feasibility_unlock,
            obligation_closure=float(local.get("obligation_closure", 0.0)),
            terminal_reachability=float(local.get("terminal_reachability", 0.0)),
            missing_premise_reduction=float(local.get("missing_premise_reduction", 0.0)),
            candidate_reachability=float(local.get("candidate_reachability", 0.0)),
            evidence_path=float(local.get("evidence_path", 0.0)),
            dead_end_risk=float(local.get("dead_end_risk", 0.0)),
            absolute_graph_risk=_unit(growth),
            obligation_importance=float(closure.get("obligation_importance", 0.0)),
            operation_closure_probability=float(
                closure.get("operation_closure_probability", 0.0)
            ),
            expected_obligation_delta=float(
                closure.get("expected_obligation_delta", 0.0)
            ),
            obligation_terminal_return=float(
                closure.get("obligation_terminal_return", 0.0)
            ),
            operation_redundancy=float(closure.get("operation_redundancy", 0.0)),
        )

    def _fidelity_resource_scales(
        self, operation: GraphOperation, fraction: float,
    ) -> tuple[float | None, float | None]:
        if not self.config.exact_fidelity_resource_accounting:
            return None, None
        fraction = max(0.1, min(1.0, float(fraction)))
        provider_backed = operation.operation_type in {
            OperationType.EXPAND, OperationType.BRANCH,
            OperationType.VERIFY, OperationType.MERGE,
        } and not (
            operation.operation_type == OperationType.BRANCH
            and str(operation.payload.get("mode", "")) == "assignments"
        )
        if not provider_backed:
            return 1.0, fraction
        max_tokens = {
            OperationType.EXPAND: self.config.graph_editor_max_tokens,
            OperationType.BRANCH: self.config.typed_extraction_max_tokens,
            OperationType.VERIFY: self.config.soft_verifier_max_tokens,
            OperationType.MERGE: self.config.join_validation_max_tokens,
        }[operation.operation_type]
        if operation.operation_type == OperationType.VERIFY:
            maximum = max(1, int(self.config.max_independent_verifications))
            requested = max(1, int(round(maximum * fraction)))
            schema_floor = 260 + 90 * max(1, len(operation.source_ids))
            per_call = min(max_tokens, max(int(max_tokens * fraction), schema_floor))
            token_scale = per_call * requested / max(1, max_tokens * maximum)
            return requested / maximum, token_scale
        schema_floor = {
            OperationType.BRANCH: 260 + 90 * max(1, int(round(
                self.config.max_extracted_claims_per_round * max(fraction, 0.25)
            ))),
            OperationType.MERGE: 400,
            OperationType.EXPAND: 500,
        }.get(operation.operation_type, 0)
        per_call = min(max_tokens, max(int(max_tokens * fraction), schema_floor))
        # Token fidelity changes a single request; the provider call count does
        # not become fractional merely because its output cap is smaller.
        return 1.0, per_call / max(1, max_tokens)

    def _critical_obligation_reserve(
        self, graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
    ) -> dict[str, int]:
        if not self.config.critical_obligation_budget_reserve:
            return {"llm_calls": 0, "tokens": 0, "region_count": 0}
        targeted = set(operation_obligation_targets(
            graph, operation,
            strict=self.config.operation_conditioned_obligation_closure,
        ))
        requirements = {
            "missing_evidence": (1, 530),
            "missing_claim": (1, 260),
            "missing_binding": (1, 530),
            "missing_verification": (1, 260),
            "missing_join_premise": (1, 400),
            "terminal_disconnected_join": (1, 500),
            "contradiction": (0, 0),
        }
        by_region: dict[tuple[str, str], tuple[int, int]] = {}
        for obligation_id, row in graph.proof_obligations.items():
            if (
                obligation_id in targeted
                or row.status != "OPEN"
                or not row.terminal_reachable
            ):
                continue
            need = requirements.get(row.obligation_type, (0, 0))
            key = (row.target_subgoal, row.branch_id)
            previous = by_region.get(key, (0, 0))
            by_region[key] = (max(previous[0], need[0]), max(previous[1], need[1]))
        # Preserve one causal continuation after the selected operation closes
        # its immediate target (retrieve->extract, extract->verify, verify->join).
        continuation = {
            OperationType.RETRIEVE: (1, 530),
            OperationType.BRANCH: (1, 260),
            OperationType.VERIFY: (1, 400),
            OperationType.MERGE: (0, 0),
        }.get(operation.operation_type, (0, 0))
        return {
            "llm_calls": sum(value[0] for value in by_region.values()) + continuation[0],
            "tokens": sum(value[1] for value in by_region.values()) + continuation[1],
            "region_count": len(by_region),
        }

    def _budget_packet(
        self,
        operation: GraphOperation,
        heat: float,
        remaining: dict[str, int],
        prior: dict[str, float],
        fidelity_fraction: float | None = None,
    ) -> dict[str, int]:
        assignment_branch = (
            operation.operation_type == OperationType.BRANCH
            and str(operation.payload.get("mode", "")) == "assignments"
        )
        if self.config.graph_local_delayed_value:
            prior_observations = float(prior.get("exact_observations", 0.0))
            prior_value = float(prior.get("exact_posterior_value", 0.5))
            prior_cooldown = float(prior.get("exact_cooldown_active", 0.0))
        else:
            prior_observations = float(prior.get("observations", 0.0))
            prior_value = float(prior.get("posterior_value", 0.5))
            prior_cooldown = float(prior.get("cooldown_active", 0.0))
        max_tokens = 0 if assignment_branch else {
            OperationType.EXPAND: self.config.graph_editor_max_tokens,
            OperationType.BRANCH: self.config.typed_extraction_max_tokens,
            OperationType.VERIFY: self.config.soft_verifier_max_tokens,
            OperationType.MERGE: self.config.join_validation_max_tokens,
            OperationType.COMMIT: 0,
            OperationType.REVISE: 0,
            OperationType.RETRIEVE: 0,
            OperationType.PRUNE: 0,
        }[operation.operation_type]
        if fidelity_fraction is not None:
            fraction = max(0.1, min(1.0, float(fidelity_fraction)))
            verifications = max(1, int(round(
                self.config.max_independent_verifications * fraction
            )))
            packet_heat = fraction
        elif self.config.allocator_mode != "adaptive_evc":
            fraction = self.config.allocation_mid_token_fraction
            verifications = 1
            packet_heat = 0.5
        else:
            # With no outcomes, preserve the pre-feedback allocation exactly.
            # Once evidence exists, use a conservative posterior adjustment; a
            # neutral prior must never dilute a genuinely hot graph region.
            if prior_observations <= 0.0:
                packet_heat = max(0.0, min(1.0, heat))
            else:
                posterior = prior_value
                packet_heat = max(0.0, min(1.0, heat + 0.25 * (posterior - 0.5)))
            if prior_cooldown >= 1.0:
                packet_heat = min(packet_heat, self.config.allocation_min_token_fraction)
            if packet_heat >= self.config.allocation_high_heat_threshold:
                fraction = 1.0
                verifications = self.config.max_independent_verifications
            elif packet_heat >= self.config.allocation_mid_heat_threshold:
                fraction = self.config.allocation_mid_token_fraction
                verifications = 1
            else:
                fraction = self.config.allocation_min_token_fraction
                verifications = 1
        candidate_cap = max(
            1,
            int(round(self.config.max_extracted_claims_per_round * max(fraction, 0.25))),
        )
        requested_tokens = int(max_tokens * fraction) if max_tokens else 0
        schema_floor = {
            OperationType.BRANCH: 260 + 90 * candidate_cap,
            OperationType.VERIFY: 260 + 90 * max(1, len(operation.source_ids)),
            OperationType.MERGE: 400,
            OperationType.EXPAND: 500,
        }.get(operation.operation_type, 0) if not assignment_branch else 0
        requested_tokens = min(max_tokens, max(requested_tokens, schema_floor)) if max_tokens else 0
        tokens = min(remaining["tokens"], requested_tokens)
        top_k = min(
            self.config.max_adaptive_top_k,
            max(1, int(round(self.config.top_k * (0.5 + 0.5 * packet_heat)))),
        )
        branch_width = max(
            1, int(round(self.config.max_active_branches * max(fraction, 0.34))),
        )
        if assignment_branch:
            branch_width = min(
                len(operation.payload.get("candidate_ids", [])),
                max(2, branch_width),
            )
        packet = {
            "max_tokens": tokens,
            "retrieval_top_k": top_k,
            "candidate_cap": candidate_cap,
            "verification_samples": verifications,
            "branch_width": branch_width,
            "revision_allowance": 1 if (
                operation.operation_type == OperationType.REVISE
                or packet_heat >= self.config.allocation_mid_heat_threshold
            ) else 0,
        }
        if self.config.exact_fidelity_resource_accounting:
            provider_backed = operation.operation_type in {
                OperationType.EXPAND, OperationType.BRANCH,
                OperationType.VERIFY, OperationType.MERGE,
            } and not assignment_branch
            packet["llm_calls"] = (
                verifications if operation.operation_type == OperationType.VERIFY
                else int(provider_backed)
            )
            packet["token_upper_bound"] = (
                tokens * verifications
                if operation.operation_type == OperationType.VERIFY else tokens
            )
        return packet


def operation_family(operation: GraphOperation) -> str:
    mode = str(operation.payload.get("mode", ""))
    return f"{operation.operation_type.value.lower()}:{mode or 'default'}"


def operation_region_key(operation: GraphOperation) -> str:
    contextual_fields = {
        key: operation.payload.get(key)
        for key in (
            "query", "event", "extraction_evidence_count", "join_signature",
            "candidate_id", "candidate_ids", "dependency_claim_ids", "premise_ids",
            "proof_gap_reason", "recovery_target_claim_ids",
        )
        if key in operation.payload
    }
    return stable_hash({
        "target": operation.target_id,
        "branch": operation.branch_id,
        "family": operation_family(operation),
        "sources": sorted(str(value) for value in operation.source_ids),
        "context": contextual_fields,
    })[:16]


def operation_coarse_region_key(operation: GraphOperation) -> str:
    """Stable within-question region shared by contextual action variants."""
    return stable_hash({
        "target": operation.target_id,
        "branch": operation.branch_id,
        "family": operation_family(operation),
    })[:16]


def feedback_key(family: str, region_key: str = "*") -> str:
    return f"{family}|{region_key}"


def feedback_prior(
    graph: DynamicReasoningHypergraphV2, family: str, region_key: str,
    coarse_region_key: str | None = None,
    *,
    use_hierarchical: bool = False,
) -> dict[str, float]:
    exact = graph.operation_feedback.get(feedback_key(family, region_key))
    # Different subgoals can have radically different evidence and topology.
    # Family-wide statistics remain serialized for audit, but only causal
    # outcomes from the same target region control later allocation.
    stats = exact or OperationFeedbackStats()
    coarse = (
        graph.operation_feedback.get(feedback_key(family, coarse_region_key))
        if use_hierarchical and coarse_region_key else None
    )
    exact_observations = float(stats.observations)
    # The coarse ledger contains the exact observation too.  It is a back-off
    # prior for unseen contextual variants, never an extra vote when exact data
    # already exists.
    coarse_observations = (
        0.5 * float(coarse.observations)
        if coarse is not None and exact_observations <= 0.0 else 0.0
    )
    observations = exact_observations + coarse_observations
    if observations > 0.0:
        coarse_value = coarse.posterior_value if coarse is not None else 0.5
        coarse_success = coarse.posterior_success if coarse is not None else 0.5
        posterior_value = (
            exact_observations * stats.posterior_value
            + coarse_observations * coarse_value
        ) / observations
        posterior_success = (
            exact_observations * stats.posterior_success
            + coarse_observations * coarse_success
        ) / observations
        cumulative_cost = stats.cumulative_cost + (
            0.5 * coarse.cumulative_cost if coarse is not None else 0.0
        )
        mean_cost = cumulative_cost / observations
    else:
        posterior_value = posterior_success = 0.5
        mean_cost = 0.0
    cooldown_until = max(
        stats.cooldown_until_step,
        coarse.cooldown_until_step if coarse is not None else 0,
    )
    return {
        "observations": observations,
        "exact_observations": exact_observations,
        "coarse_observations": coarse_observations,
        "posterior_value": float(posterior_value),
        "posterior_success": float(posterior_success),
        "consecutive_failures": float(max(
            stats.consecutive_failures,
            coarse.consecutive_failures if coarse is not None else 0,
        )),
        "cooldown_until_step": float(cooldown_until),
        "cooldown_active": float(graph.step < cooldown_until),
        "mean_cost": max(0.0, min(1.0, mean_cost)),
        "exact_posterior_value": float(stats.posterior_value),
        "exact_posterior_success": float(stats.posterior_success),
        "exact_cooldown_active": float(graph.step < stats.cooldown_until_step),
        "exact_mean_cost": (
            float(stats.cumulative_cost) / max(1.0, float(stats.observations))
        ),
    }


def operation_region_node_ids(
    graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
) -> list[str]:
    node_ids = list(dict.fromkeys([operation.target_id, *operation.source_ids]))
    for node in graph.nodes.values():
        if isinstance(node, (ClaimNode, EvidenceNode)):
            if node.target_subgoal == operation.target_id and node.branch_id == operation.branch_id:
                node_ids.append(node.node_id)
        elif isinstance(node, SubgoalNode) and node.node_id == operation.target_id:
            node_ids.append(node.node_id)
    return list(dict.fromkeys(node_ids))


def summarize_operation_region(
    graph: DynamicReasoningHypergraphV2, operation: GraphOperation,
) -> dict[str, float]:
    node_ids = operation_region_node_ids(graph, operation)
    states = [graph.belief_states[node_id] for node_id in node_ids if node_id in graph.belief_states]
    claims = [graph.nodes[node_id] for node_id in node_ids if isinstance(graph.nodes.get(node_id), ClaimNode)]
    evidence = [graph.nodes[node_id] for node_id in node_ids if isinstance(graph.nodes.get(node_id), EvidenceNode)]
    active_claims = [
        node for node in claims
        if node.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
    ]
    joined = [
        node for node in claims
        if graph.claim_semantics.get(node.node_id)
        and graph.claim_semantics[node.node_id].join_depth > 0
    ]
    grounded_claims = [
        node for node in active_claims
        if node.evidence_refs and node.provenance.metadata.get("source_spans")
    ]
    proof_leaves = {
        leaf_id
        for row in graph.join_attempt_history if row.accepted
        for leaf_id in (row.proof_leaf_ids or row.premise_ids)
    }
    accepted_answers = [row for row in graph.answers() if row.status == AnswerStatus.ACCEPTED]
    completed = [row for row in graph.subgoals() if row.status == SubgoalStatus.RESOLVED]
    total_subgoals = max(1, len(graph.subgoals()))

    def avg(name: str, default: float) -> float:
        return sum(float(getattr(row, name)) for row in states) / len(states) if states else default

    chain_progress = min(1.0, len(completed) / total_subgoals + 0.25 * len(accepted_answers))
    terminal_context = operation.payload.get("_terminal_context", {})
    if not isinstance(terminal_context, dict):
        terminal_context = {}
    return {
        "node_count": float(len(node_ids)),
        "claim_count": float(len(claims)),
        "active_claim_count": float(len(active_claims)),
        "evidence_count": float(len(evidence)),
        "join_count": float(len(joined)),
        "grounded_claim_count": float(len(grounded_claims)),
        "proof_leaf_count": float(len(proof_leaves)),
        "accepted_answer_count": float(len(accepted_answers)),
        "completed_subgoal_count": float(len(completed)),
        "uncertainty": avg("uncertainty", 1.0),
        "absolute_support": avg("absolute_support", 0.0),
        "evidence_gap": avg("evidence_gap", 1.0),
        "entropy": avg("entropy", 1.0),
        "dependency_unlock": avg("dependency_unlock_value", 0.0),
        "contradiction_pressure": avg("contradiction_pressure", 0.0),
        "answer_chain_progress": chain_progress,
        "terminal_gap": _unit(float(terminal_context.get("terminal_gap", 1.0))),
        "terminal_absolute_support": _unit(float(
            terminal_context.get("absolute_support", 0.0)
        )),
        "terminal_relative_weight": _unit(float(
            terminal_context.get("relative_weight", 0.0)
        )),
        "terminal_entropy": _unit(float(terminal_context.get("entropy", 1.0))),
        "terminal_evidence_gap": _unit(float(
            terminal_context.get("evidence_gap", 1.0)
        )),
        "terminal_chain_coverage": _unit(float(
            terminal_context.get("chain_coverage", 0.0)
        )),
    }


def _terminal_operation_affinity(
    operation: GraphOperation, context: dict,
) -> float:
    reasons = {str(value) for value in context.get("rejection_reasons", [])}
    if not reasons:
        return 1.0 if operation.operation_type == OperationType.COMMIT else 0.25
    affinities = {
        OperationType.RETRIEVE: {
            "evidence_gap_above_maximum": 1.0,
            "missing_terminal_candidate": 0.85,
            "unresolved_competing_branches": 0.75,
            "insufficient_support_chain": 0.60,
        },
        OperationType.VERIFY: {
            "absolute_support_below_minimum": 1.0,
            "relative_margin_below_minimum": 0.90,
            "claim_set_entropy_above_maximum": 0.90,
            "answer_type_consistency_below_minimum": 0.85,
            "contradiction_pressure_above_maximum": 0.75,
        },
        OperationType.MERGE: {
            "insufficient_support_chain": 1.0,
            "chain_coverage_below_minimum": 1.0,
            "missing_terminal_candidate": 0.80,
        },
        OperationType.BRANCH: {
            "missing_terminal_candidate": 1.0,
            "absolute_support_below_minimum": 0.70,
            "unresolved_competing_branches": 0.70,
        },
        OperationType.EXPAND: {
            "missing_terminal_candidate": 0.90,
            "insufficient_support_chain": 0.85,
            "chain_coverage_below_minimum": 0.85,
        },
        OperationType.REVISE: {
            "contradiction_pressure_above_maximum": 1.0,
            "relative_margin_below_minimum": 0.50,
        },
        OperationType.COMMIT: {},
        OperationType.PRUNE: {
            "relative_margin_below_minimum": 0.60,
            "claim_set_entropy_above_maximum": 0.60,
        },
    }
    scores = affinities[operation.operation_type]
    return max((scores.get(reason, 0.20) for reason in reasons), default=0.20)


def _fixed_priority(value: OperationType) -> int:
    return {
        OperationType.REVISE: 0,
        OperationType.COMMIT: 1,
        OperationType.VERIFY: 2,
        OperationType.MERGE: 3,
        OperationType.RETRIEVE: 4,
        OperationType.BRANCH: 5,
        OperationType.EXPAND: 6,
        OperationType.PRUNE: 7,
    }[value]


def _operation_reduction(value: OperationType) -> float:
    return {
        OperationType.VERIFY: 0.9,
        OperationType.RETRIEVE: 0.8,
        OperationType.MERGE: 0.8,
        OperationType.REVISE: 0.7,
        OperationType.BRANCH: 0.6,
        OperationType.EXPAND: 0.5,
        OperationType.COMMIT: 0.2,
        OperationType.PRUNE: 0.2,
    }[value]


def _minmax(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [max(0.0, min(1.0, values[0]))] if values else []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        # Preserve absolute signal when the ready set has no variance. Returning
        # 0.5 erased cooldowns and made identical graph regions look neutral.
        value = max(0.0, min(1.0, values[0]))
        return [value for _ in values]
    return [(value - low) / (high - low) for value in values]


def _shadow_price(remaining: int | float, capacity: int | float) -> float:
    """Normalized deterministic scarcity price for a resource budget."""
    ratio = max(0.0, min(1.0, float(remaining) / max(1.0, float(capacity))))
    return 0.5 + 0.5 * (1.0 - ratio)


def _absolute_resource_fraction(
    demand: int | float,
    capacity: int | float,
    remaining: int | float,
    scarcity_max_multiplier: float,
) -> float:
    """Choice-invariant resource price, monotone in resource scarcity."""
    capacity = max(1.0, float(capacity))
    remaining_ratio = _unit(float(remaining) / capacity)
    multiplier = 1.0 + (
        max(1.0, float(scarcity_max_multiplier)) - 1.0
    ) * (1.0 - remaining_ratio)
    return _unit(float(demand) / capacity * multiplier)


def _weighted_mean(values: dict[str, tuple[float, float]]) -> float:
    weight = sum(max(0.0, float(row[1])) for row in values.values())
    if weight <= 1e-12:
        return 0.0
    return sum(
        _unit(value) * max(0.0, float(component_weight))
        for value, component_weight in values.values()
    ) / weight


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
