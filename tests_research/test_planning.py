import pytest

from tdca_research.models import Claim, ClaimStatus, ReasoningPlan, ReasoningSlot, VariableBinding
from tdca_research.planning import PlanValidationError, bind_slot_question, ready_slots, validate_plan


def plan():
    return validate_plan(ReasoningPlan("root", [
        ReasoningSlot("s1", "Who leads Alpha?", output_variable="$leader"),
        ReasoningSlot("s2", "Where was $leader born?", dependencies=["s1"], variable_bindings=[VariableBinding("$leader", "s1")], terminal=True),
    ]))


def test_successor_is_blocked_until_verified_binding_exists():
    value = plan()
    assert [slot.slot_id for slot in ready_slots(value, [])] == ["s1"]
    proposed = Claim("c1", "Alpha", "led by", "Ada", "person", "s1", status=ClaimStatus.PROPOSED)
    assert [slot.slot_id for slot in ready_slots(value, [proposed])] == ["s1"]
    proposed.status = ClaimStatus.VERIFIED
    value.slots[0].status = value.slots[0].status.COMPLETE
    assert [slot.slot_id for slot in ready_slots(value, [proposed])] == ["s2"]
    question, used = bind_slot_question(value.slots[1], value, [proposed])
    assert question == "Where was Ada born?"
    assert used == ["c1"]


def test_cycle_and_unknown_variable_are_rejected():
    with pytest.raises(PlanValidationError):
        validate_plan(ReasoningPlan("x", [ReasoningSlot("a", "Use $x", terminal=True)]))
    with pytest.raises(PlanValidationError):
        validate_plan(ReasoningPlan("x", [
            ReasoningSlot("a", "a", dependencies=["b"], output_variable="$a"),
            ReasoningSlot("b", "b", dependencies=["a"], output_variable="$b", terminal=True),
        ]))


def test_disconnected_work_and_non_sink_terminal_are_rejected():
    with pytest.raises(PlanValidationError, match="does not contribute"):
        validate_plan(ReasoningPlan("x", [
            ReasoningSlot("unused", "unused", output_variable="$unused"),
            ReasoningSlot("answer", "answer", output_variable="$answer", terminal=True),
        ]))


def test_oracle_answer_types_are_inferred_from_generic_question_form():
    planner = __import__("tdca_research.planning.planner", fromlist=["Planner"]).Planner
    assert planner._infer_answer_type("What year did X die?", "1572") == "year"
    assert planner._infer_answer_type("When was X abolished?", "30 November 1999") == "date"
    assert planner._infer_answer_type("Who did the Tigers beat?", "Chicago Cubs") == "entity"
    assert planner._infer_answer_type("Where was X born?", "Paris") == "location"


def test_planner_prunes_only_slots_outside_terminal_ancestor_closure():
    planner = __import__("tdca_research.planning.planner", fromlist=["Planner"]).Planner
    value = ReasoningPlan("q", [
        ReasoningSlot("needed", "needed", output_variable="$needed"),
        ReasoningSlot("side", "side", output_variable="$side"),
        ReasoningSlot("terminal", "use $needed", dependencies=["needed"],
                      variable_bindings=[VariableBinding("$needed", "needed")], terminal=True),
    ])
    pruned = planner._prune_noncontributing_slots(value)
    assert [slot.slot_id for slot in pruned.slots] == ["needed", "terminal"]
    validate_plan(pruned)
    with pytest.raises(PlanValidationError, match="sinks"):
        validate_plan(ReasoningPlan("x", [
            ReasoningSlot("early", "early", output_variable="$early", terminal=True),
            ReasoningSlot("late", "use $early", dependencies=["early"],
                          variable_bindings=[VariableBinding("$early", "early")], output_variable="$answer"),
        ]))
