from __future__ import annotations

import re
from dataclasses import dataclass

from ..dynamic.graph import SubgoalNode
from ..utils import normalize_text, stable_hash


TYPE_PARENTS = {
    "city": "location", "country": "location", "state": "location",
    "province": "location", "region": "location", "continent": "location",
    "county": "location", "district": "location", "municipality": "location",
    "administrative_district": "location", "administrative_entity": "location",
    "administrative_territorial_entity": "location",
    "river": "location", "mountain": "location",
    "person": "entity", "organization": "entity", "company": "organization",
    "university": "organization", "team": "organization",
    "film": "creative_work", "book": "creative_work", "song": "creative_work",
    "creative_work": "entity", "location": "entity", "date": "literal",
    "year": "date", "number": "literal", "quantity": "number",
}


@dataclass(frozen=True)
class QueryVariable:
    variable_id: str
    subgoal_id: str
    expected_type: str
    role: str = "answer"


@dataclass(frozen=True)
class QueryConstraint:
    constraint_id: str
    subgoal_id: str
    description: str
    input_variables: tuple[str, ...]
    output_variable: str
    known_entities: tuple[str, ...]
    required_qualifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryGraph:
    root_question: str
    variables: tuple[QueryVariable, ...]
    constraints: tuple[QueryConstraint, ...]

    def to_payload(self) -> dict:
        return {
            "root_question": self.root_question,
            "variables": [row.__dict__ for row in self.variables],
            "constraints": [{
                **row.__dict__,
                "input_variables": list(row.input_variables),
                "known_entities": list(row.known_entities),
                "required_qualifiers": list(row.required_qualifiers),
            } for row in self.constraints],
        }


def compile_query_graph(
    question: str,
    subgoals: list[SubgoalNode],
    *,
    constraint_aware_entities: bool = False,
) -> QueryGraph:
    """Compile planner output into an explicit, label-free constraint agenda."""
    variables = []
    constraints = []
    known_by_subgoal: dict[str, tuple[str, ...]] = {}
    for subgoal in sorted(subgoals, key=lambda value: value.node_id):
        variable_id = f"?answer:{subgoal.node_id}"
        variables.append(QueryVariable(
            variable_id, subgoal.node_id, canonical_type(subgoal.answer_type),
        ))
        known = tuple(_question_entities(
            subgoal.instantiated_question or subgoal.question_template,
            preserve_connectors=constraint_aware_entities,
        ))
        known_by_subgoal[subgoal.node_id] = known
        inputs = tuple(f"?answer:{value}" for value in subgoal.dependencies)
        constraints.append(QueryConstraint(
            constraint_id=f"constraint_{stable_hash([subgoal.node_id, subgoal.question_template])[:16]}",
            subgoal_id=subgoal.node_id,
            description=subgoal.question_template,
            input_variables=inputs,
            output_variable=variable_id,
            known_entities=known,
            required_qualifiers=tuple(_constraint_qualifiers(
                subgoal.instantiated_question or subgoal.question_template
            )),
        ))
    return QueryGraph(question, tuple(variables), tuple(constraints))


def canonical_type(value: object) -> str:
    text = str(value or "entity").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "place": "location", "geo": "location", "geographic_location": "location",
        "human": "person", "people": "person", "org": "organization",
        "work": "creative_work", "work_of_art": "creative_work",
        "datetime": "date", "integer": "number", "float": "number",
    }
    return aliases.get(text, text or "entity")


def type_lineage(value: object) -> tuple[str, ...]:
    current = canonical_type(value)
    rows = [current]
    seen = {current}
    while current in TYPE_PARENTS and TYPE_PARENTS[current] not in seen:
        current = TYPE_PARENTS[current]
        rows.append(current)
        seen.add(current)
    if "entity" not in seen and "literal" not in seen:
        rows.append("entity")
    return tuple(rows)


def types_compatible(left: object, right: object) -> bool:
    left_type = canonical_type(left)
    right_type = canonical_type(right)
    if "entity" in {left_type, right_type}:
        return True
    return bool(set(type_lineage(left_type)) & set(type_lineage(right_type)))


def _question_entities(text: str, *, preserve_connectors: bool = False) -> list[str]:
    # Relation-light anchors only.  Wh-phrases are explicitly removed so they
    # cannot become fake entities in the activated graph.
    token = r"[A-Z][\w'’\-]*"
    if preserve_connectors:
        connector = r"(?:of|the|de|del|la|van|von|and)"
        rows = re.findall(
            rf"\b{token}(?:(?:\s+{token})|(?:\s+{connector}\s+{token})){{0,5}}\b",
            text,
        )
    else:
        rows = re.findall(rf"\b{token}(?:\s+{token}){{0,5}}\b", text)
    stop = {"what", "which", "who", "where", "when", "how", "the", "a", "an"}
    return [
        value for value in dict.fromkeys(rows)
        if normalize_text(value) not in stop
    ]


def _constraint_qualifiers(text: str) -> list[str]:
    """Extract only explicit, label-free constraint families from a subgoal.

    The values are audit descriptors for the independent verifier.  They do
    not identify an answer, dataset, entity, or hidden hop count.
    """
    normalized = normalize_text(text)
    patterns = {
        "temporal": (
            " before ", " after ", " during ", " when ", " earliest ",
            " latest ", " first ", " last ", " year ", " date ",
        ),
        "comparison": (
            " more ", " less ", " fewer ", " greater ", " largest ",
            " smallest ", " highest ", " lowest ", " older ", " younger ",
        ),
        "cardinality": (
            "how many", " number of ", " total ", " percentage ", " percent ",
        ),
        "set": (" both ", " common ", " intersection ", " either ", " all "),
        "negation": (" not ", " never ", " except ", " without "),
    }
    padded = f" {normalized} "
    return [
        family for family, cues in patterns.items()
        if any(cue in padded for cue in cues)
    ]
