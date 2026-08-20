from __future__ import annotations

from dataclasses import dataclass, field

from .config import DynamicResearchConfig
from .graph import GraphOperation


@dataclass(frozen=True)
class OperationSignals:
    uncertainty_reduction: float = 0.0
    dependency_unlock: float = 0.0
    answer_impact: float = 0.0
    evidence_novelty: float = 0.0
    recovery_value: float = 0.0
    expected_cost: float = 0.0
    graph_growth_risk: float = 0.0


@dataclass(frozen=True)
class RankedOperation:
    operation: GraphOperation
    raw_signals: OperationSignals
    normalized_signals: OperationSignals
    utility: float


@dataclass
class OperationScheduler:
    config: DynamicResearchConfig
    last_ranking: list[RankedOperation] = field(default_factory=list)

    def rank(
        self, operations: list[GraphOperation], signals: dict[str, OperationSignals],
    ) -> list[RankedOperation]:
        if not operations:
            self.last_ranking = []
            return []
        names = tuple(OperationSignals.__dataclass_fields__)
        columns = {
            name: [float(getattr(signals.get(op.operation_id, OperationSignals()), name)) for op in operations]
            for name in names
        }
        normalized_columns = {name: _minmax(values) for name, values in columns.items()}
        ranked = []
        for index, operation in enumerate(operations):
            raw = signals.get(operation.operation_id, OperationSignals())
            normalized = OperationSignals(**{
                name: normalized_columns[name][index] for name in names
            })
            utility = (
                self.config.utility_weight_uncertainty * normalized.uncertainty_reduction
                + self.config.utility_weight_unlock * normalized.dependency_unlock
                + self.config.utility_weight_answer_impact * normalized.answer_impact
                + self.config.utility_weight_novelty * normalized.evidence_novelty
                + self.config.utility_weight_recovery * normalized.recovery_value
                - self.config.utility_weight_cost * normalized.expected_cost
                - self.config.utility_weight_growth_risk * normalized.graph_growth_risk
            )
            operation.utility_components = {
                "raw": raw.__dict__, "normalized": normalized.__dict__, "utility": utility,
            }
            ranked.append(RankedOperation(operation, raw, normalized, utility))
        self.last_ranking = sorted(
            ranked,
            key=lambda row: (-row.utility, row.operation.operation_type.value, row.operation.operation_id),
        )
        return self.last_ranking

    def select(
        self, operations: list[GraphOperation], signals: dict[str, OperationSignals],
    ) -> RankedOperation | None:
        ranked = self.rank(operations, signals)
        return ranked[0] if ranked else None


def _minmax(values: list[float]) -> list[float]:
    if len(values) <= 1:
        # A single ready operation is intentionally non-scheduled. Keeping its
        # components at zero avoids pretending a non-trivial choice occurred.
        return [0.0 for _ in values]
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]
