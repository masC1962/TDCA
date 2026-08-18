from __future__ import annotations

import re

import networkx as nx

from ..models import Claim, ClaimStatus, ReasoningPlan, ReasoningSlot, SlotStatus


class PlanValidationError(ValueError):
    pass


_VARIABLE = re.compile(r"\$[A-Za-z][A-Za-z0-9_]*")


def validate_plan(plan: ReasoningPlan) -> ReasoningPlan:
    if not plan.slots or len(plan.slots) > 12:
        raise PlanValidationError("a plan must contain 1..12 slots")
    by_id = plan.by_id()
    if len(by_id) != len(plan.slots):
        raise PlanValidationError("slot ids must be unique")
    graph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    output_variables: set[str] = set()
    for slot in plan.slots:
        if not slot.slot_id or not slot.subquestion_template.strip():
            raise PlanValidationError("slot id and question are required")
        if not slot.output_variable.startswith("$"):
            raise PlanValidationError("output variables must start with $")
        if slot.output_variable in output_variables:
            raise PlanValidationError(f"duplicate output variable {slot.output_variable}")
        output_variables.add(slot.output_variable)
        for dependency in slot.dependencies:
            if dependency not in by_id:
                raise PlanValidationError(f"unknown dependency {dependency}")
            if dependency == slot.slot_id:
                raise PlanValidationError("self dependency")
            graph.add_edge(dependency, slot.slot_id)
        dependency_set = set(slot.dependencies)
        for binding in slot.variable_bindings:
            if binding.source_slot not in dependency_set:
                raise PlanValidationError("binding source must be a declared dependency")
            expected = by_id[binding.source_slot].output_variable
            if binding.variable != expected:
                raise PlanValidationError(f"binding {binding.variable} does not match source output {expected}")
            if binding.variable not in slot.subquestion_template:
                raise PlanValidationError(f"bound variable {binding.variable} absent from template")
        referenced = set(_VARIABLE.findall(slot.subquestion_template))
        allowed = {binding.variable for binding in slot.variable_bindings}
        unknown = referenced - allowed
        if unknown:
            raise PlanValidationError(f"unbound variables in {slot.slot_id}: {sorted(unknown)}")
    if not nx.is_directed_acyclic_graph(graph):
        raise PlanValidationError("reasoning plan must be acyclic")
    if not any(slot.terminal for slot in plan.slots):
        sinks = [node for node, degree in graph.out_degree() if degree == 0]
        if len(sinks) == 1:
            by_id[sinks[0]].terminal = True
        else:
            raise PlanValidationError("plan needs an explicit terminal slot")
    terminal_ids = {slot.slot_id for slot in plan.slots if slot.terminal}
    for terminal_id in terminal_ids:
        if graph.out_degree(terminal_id) != 0:
            raise PlanValidationError("terminal slots must be dependency sinks")
    # Every slot must contribute to some terminal result. Disconnected side work
    # otherwise consumes budget without being part of a valid answer chain.
    for slot in plan.slots:
        if slot.slot_id in terminal_ids:
            continue
        if not any(nx.has_path(graph, slot.slot_id, terminal_id) for terminal_id in terminal_ids):
            raise PlanValidationError(f"slot {slot.slot_id} does not contribute to a terminal slot")
    for slot in plan.slots:
        slot.status = SlotStatus.READY if not slot.dependencies else SlotStatus.PENDING
    return plan


def _verified_by_slot(claims: list[Claim]) -> dict[str, Claim]:
    result: dict[str, Claim] = {}
    for claim in claims:
        if claim.status != ClaimStatus.VERIFIED:
            continue
        current = result.get(claim.target_slot)
        if current is None or claim.calibrated_confidence > current.calibrated_confidence:
            result[claim.target_slot] = claim
    return result


def ready_slots(plan: ReasoningPlan, claims: list[Claim]) -> list[ReasoningSlot]:
    verified = _verified_by_slot(claims)
    ready: list[ReasoningSlot] = []
    for slot in plan.slots:
        if slot.status in {SlotStatus.COMPLETE, SlotStatus.RUNNING, SlotStatus.FAILED}:
            continue
        if all(dependency in verified for dependency in slot.dependencies):
            slot.status = SlotStatus.READY
            ready.append(slot)
        else:
            slot.status = SlotStatus.PENDING
    return ready


def bind_slot_question(slot: ReasoningSlot, plan: ReasoningPlan, claims: list[Claim]) -> tuple[str, list[str]]:
    verified = _verified_by_slot(claims)
    question = slot.subquestion_template
    used_claims: list[str] = []
    for binding in slot.variable_bindings:
        claim = verified.get(binding.source_slot)
        if claim is None:
            raise PlanValidationError(f"slot {slot.slot_id} cannot bind {binding.variable}")
        question = question.replace(binding.variable, claim.object)
        used_claims.append(claim.claim_id)
    if _VARIABLE.search(question):
        raise PlanValidationError(f"slot {slot.slot_id} still has an unbound variable")
    slot.bound_question = question
    return question, used_claims
