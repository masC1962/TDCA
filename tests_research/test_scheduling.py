import pytest

from tdca_research.models import ReasoningPlan, ReasoningSlot
from tdca_research.scheduling import Scheduler, SlotSignals, diffuse_temperatures


def test_expected_utility_prefers_unlock_per_cost():
    slots = [ReasoningSlot("cheap", "x"), ReasoningSlot("expensive", "y", terminal=True)]
    signals = {
        "cheap": SlotSignals(expected_information_gain=0.8, dependency_unlock_value=2, evidence_gap=1, confidence_need=1, expected_cost=1),
        "expensive": SlotSignals(expected_information_gain=1, dependency_unlock_value=1, evidence_gap=1, confidence_need=1, expected_cost=4),
    }
    assert Scheduler("expected_utility").select(slots, signals).slot_id == "cheap"


def test_diffusion_is_nonnegative_stable_and_conservative():
    plan = ReasoningPlan("q", [
        ReasoningSlot("a", "a", output_variable="$a"),
        ReasoningSlot("b", "b", dependencies=["a"], terminal=True),
    ])
    result = diffuse_temperatures(plan, {"a": 1.0, "b": 0.0}, alpha=0.25, decay=0.9)
    assert all(value >= 0 for value in result.values())
    assert sum(result.values()) == pytest.approx(0.9)
    assert result["b"] > 0

