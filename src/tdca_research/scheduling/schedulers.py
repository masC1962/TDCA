from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..models import ReasoningPlan, ReasoningSlot


@dataclass(frozen=True)
class SlotSignals:
    expected_information_gain: float = 0.5
    dependency_unlock_value: float = 1.0
    evidence_gap: float = 1.0
    confidence_need: float = 1.0
    expected_cost: float = 1.0
    value: float = 0.0
    temperature: float = 0.0

    def expected_utility(self) -> float:
        numerator = (
            max(0.0, self.expected_information_gain)
            * max(0.0, self.dependency_unlock_value)
            * max(0.0, self.evidence_gap)
            * max(0.0, self.confidence_need)
        )
        return numerator / max(1e-6, self.expected_cost)


def diffuse_temperatures(
    plan: ReasoningPlan,
    temperatures: dict[str, float],
    alpha: float = 0.25,
    decay: float = 0.90,
) -> dict[str, float]:
    """One stable lazy random-walk diffusion step over the dependency DAG.

    T' = decay * ((1-alpha) I + alpha P^T) T, where P is row-stochastic.
    Non-negative input remains non-negative and total heat cannot exceed
    decay times the input heat (apart from floating error).
    """
    ids = [slot.slot_id for slot in plan.slots]
    index = {slot_id: position for position, slot_id in enumerate(ids)}
    graph = nx.DiGraph()
    graph.add_nodes_from(ids)
    for slot in plan.slots:
        for dependency in slot.dependencies:
            graph.add_edge(dependency, slot.slot_id)
    vector = np.asarray([max(0.0, temperatures.get(slot_id, 0.0)) for slot_id in ids], dtype=float)
    propagated = np.zeros_like(vector)
    for node in ids:
        outgoing = list(graph.successors(node))
        if outgoing:
            share = vector[index[node]] / len(outgoing)
            for target in outgoing:
                propagated[index[target]] += share
        else:
            propagated[index[node]] += vector[index[node]]
    updated = decay * ((1.0 - alpha) * vector + alpha * propagated)
    return {slot_id: float(updated[index[slot_id]]) for slot_id in ids}


class Scheduler:
    def __init__(self, mode: str, beam_width: int = 3) -> None:
        self.mode = mode
        self.beam_width = beam_width

    def rank(self, slots: list[ReasoningSlot], signals: dict[str, SlotSignals]) -> list[ReasoningSlot]:
        def key(slot: ReasoningSlot) -> tuple[float, str]:
            signal = signals.get(slot.slot_id, SlotSignals())
            if self.mode == "greedy":
                score = signal.value
            elif self.mode == "best_first":
                score = signal.value + signal.expected_information_gain
            elif self.mode in {"diffusion", "tdca"}:
                score = signal.temperature + signal.value
            elif self.mode in {"expected_utility", "structured_tdca"}:
                score = signal.expected_utility()
            elif self.mode == "beam":
                score = signal.expected_utility()
            else:
                raise ValueError(f"unknown scheduler mode {self.mode}")
            return (-score, slot.slot_id)

        ranked = sorted(slots, key=key)
        return ranked[: self.beam_width] if self.mode == "beam" else ranked

    def select(self, slots: list[ReasoningSlot], signals: dict[str, SlotSignals]) -> ReasoningSlot | None:
        ranked = self.rank(slots, signals)
        return ranked[0] if ranked else None

