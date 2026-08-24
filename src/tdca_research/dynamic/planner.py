from __future__ import annotations

import re
from typing import Any

from ..budget import Budget
from ..llm import BaseLLM
from ..utils import estimate_message_tokens
from .config import DynamicResearchConfig
from .graph import GraphOperation, OperationType


SYSTEM = """Create only a coarse initial reasoning state for a multi-hop question. Return JSON only.
Do not answer the question and do not predict a complete long chain. Identify at most two obvious initial
subgoals plus the root objective. Output {subgoals, root_dependencies, root_question_template,
root_variable_bindings, root_answer_type}. Each subgoal has local_id, question_template, answer_type,
dependencies, variable_bindings. Rewrite the root as the next executable question, replacing resolved bridge
entities with generic variables such as $bridge_1. Every binding maps a variable to a dependency local_id.
The root objective must request exactly the same final answer as the user's complete question; never promote
an early bridge lookup into the root. Preserve a final standalone wh-clause after sentence punctuation (for
example, a trailing "Who ...?") as the root answer focus. Subgoals resolve bridge variables only.
Use no hidden labels or outside knowledge."""


class DynamicPlanner:
    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config

    def initial_expand(self, question: str) -> GraphOperation:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        self.budget.require(
            self.config.initial_plan_max_tokens,
            estimated_prompt_tokens=estimate_message_tokens(messages),
        )
        data, generation = self.llm.generate_json(
            messages, "dynamic_initial_plan_v3_root_objective_invariant", self.config.initial_plan_max_tokens,
            self.config.temperature,
        )
        self.budget.record_generation(generation)
        rows = data.get("subgoals", []) if isinstance(data.get("subgoals"), list) else []
        local_to_node: dict[str, str] = {}
        normalized_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows[:2], start=1):
            if not isinstance(row, dict):
                continue
            local_id = _identifier(str(row.get("local_id", f"sg_{index}")), f"sg_{index}")
            if local_id in local_to_node:
                continue
            local_to_node[local_id] = f"subgoal_{len(local_to_node) + 1}"
        for row in rows[:2]:
            if not isinstance(row, dict):
                continue
            local_id = _identifier(str(row.get("local_id", "")), "")
            if local_id not in local_to_node:
                continue
            dependencies = [
                local_to_node[value] for value in map(str, row.get("dependencies", []))
                if value in local_to_node and local_to_node[value] != local_to_node[local_id]
            ]
            bindings = {
                str(variable): local_to_node[str(source)]
                for variable, source in (row.get("variable_bindings", {}) or {}).items()
                if str(source) in local_to_node and local_to_node[str(source)] in dependencies
                and re.fullmatch(r"\$[A-Za-z][A-Za-z0-9_]*", str(variable))
            }
            question_template = str(row.get("question_template", "")).strip()
            variables = set(re.findall(r"\$[A-Za-z][A-Za-z0-9_]*", question_template))
            if not question_template or not variables.issubset(bindings):
                continue
            normalized_rows.append({
                "node_id": local_to_node[local_id],
                "question_template": question_template,
                "instantiated_question": question_template,
                "dependencies": list(dict.fromkeys(dependencies)),
                "variable_bindings": bindings,
                "answer_type": str(row.get("answer_type", "entity")),
                "terminal": False,
                "confidence": 0.25,
                "uncertainty": 0.75,
            })
        # Model-local IDs are only provisional. If an invalid row was removed,
        # remove every row that depended on it rather than materializing a
        # dangling or runtime-unbound execution node.
        while True:
            materialized = {row["node_id"] for row in normalized_rows}
            valid_rows = [
                row for row in normalized_rows
                if set(row["dependencies"]).issubset(materialized)
                and set(row["variable_bindings"].values()).issubset(materialized)
            ]
            if len(valid_rows) == len(normalized_rows):
                break
            normalized_rows = valid_rows
        materialized = {row["node_id"] for row in normalized_rows}
        root_dependencies = [
            local_to_node[value] for value in map(str, data.get("root_dependencies", []))
            if value in local_to_node and local_to_node[value] in materialized
        ]
        root_bindings = {
            str(variable): local_to_node[str(source)]
            for variable, source in (data.get("root_variable_bindings", {}) or {}).items()
            if str(source) in local_to_node and local_to_node[str(source)] in root_dependencies
            and re.fullmatch(r"\$[A-Za-z][A-Za-z0-9_]*", str(variable))
        }
        root_template = str(data.get("root_question_template", "")).strip() or question
        root_variables = set(re.findall(r"\$[A-Za-z][A-Za-z0-9_]*", root_template))
        if not root_variables.issubset(root_bindings):
            # Never create a runtime-unbound execution node from malformed model
            # output. The original question remains a conservative fallback.
            root_template = question
            root_bindings = {}
        original_focus_type = _root_answer_type(
            question, str(data.get("root_answer_type", "entity")),
        )
        proposed_focus_type = _root_answer_type(
            root_template, str(data.get("root_answer_type", "entity")),
        )
        if not _same_answer_focus(original_focus_type, proposed_focus_type):
            # A provider may accidentally return the first bridge question as
            # the root. Preserve the dependency agenda but restore the user's
            # actual objective. This is a structural wh/type invariant, not a
            # semantic rewrite or a label-derived correction.
            root_template = question
            root_bindings = {}
        root = {
            "node_id": "subgoal_root",
            "question_template": root_template,
            "instantiated_question": root_template,
            "dependencies": list(dict.fromkeys(root_dependencies)),
            "variable_bindings": root_bindings,
            "answer_type": original_focus_type,
            "terminal": True,
            "confidence": 0.1,
            "uncertainty": 0.9,
        }
        # A coarse provider plan can accidentally use an inner bridge lookup as
        # the terminal root while keeping the same broad answer type (for
        # example, country -> country).  Recover the omitted outer objective
        # only when the recursively expanded bridge phrase is a literal,
        # order-preserving span of the user's question.  This is a structural
        # closure rule: it uses no corpus facts, labels, or semantic similarity.
        residual = _residual_root_closure(question, normalized_rows, root)
        if residual is not None and len(normalized_rows) < 2:
            promoted_id = f"subgoal_{len(normalized_rows) + 1}"
            promoted = {
                **root,
                "node_id": promoted_id,
                "terminal": False,
                "confidence": 0.25,
                "uncertainty": 0.75,
            }
            bridge = "$nested_bridge"
            normalized_rows.append(promoted)
            root = {
                "node_id": "subgoal_root",
                "question_template": residual.replace("$nested_bridge", bridge),
                "instantiated_question": residual.replace("$nested_bridge", bridge),
                "dependencies": [promoted_id],
                "variable_bindings": {bridge: promoted_id},
                "answer_type": original_focus_type,
                "terminal": True,
                "confidence": 0.1,
                "uncertainty": 0.9,
            }
        # Some providers repeat the final executable question as both the last
        # provisional subgoal and the root objective.  Keeping both creates a
        # fake extra hop and can leave an unresolved terminal node after the
        # complete proof has already been built.  Collapse only exact structural
        # duplicates (variable names are alpha-normalized), and inherit the
        # duplicate's real dependencies/bindings.  No semantic similarity or
        # question-specific rule is used.
        duplicate = next((
            row for row in reversed(normalized_rows)
            if _template_signature(row["question_template"])
            == _template_signature(root["question_template"])
            and not any(
                row["node_id"] in other["dependencies"]
                for other in normalized_rows if other is not row
            )
        ), None)
        if duplicate is not None:
            root["question_template"] = duplicate["question_template"]
            root["instantiated_question"] = duplicate["instantiated_question"]
            root["dependencies"] = list(duplicate["dependencies"])
            root["variable_bindings"] = dict(duplicate["variable_bindings"])
            normalized_rows = [row for row in normalized_rows if row is not duplicate]
        return GraphOperation(
            operation_id="op_0001_expand_initial",
            operation_type=OperationType.EXPAND,
            target_id="subgoal_root",
            source_ids=[],
            branch_id="branch_root",
            payload={"subgoals": normalized_rows + [root]},
            reason="coarse_initial_state",
            proposed_by="dynamic_planner_v1",
            estimated_cost={"llm_calls": 1.0, "tokens": float(generation.prompt_tokens + generation.completion_tokens)},
        )


def direct_fallback_operation(question: str) -> GraphOperation:
    return GraphOperation(
        "op_0001_expand_initial", OperationType.EXPAND, "subgoal_root", [], "branch_root",
        {"subgoals": [{
            "node_id": "subgoal_root", "question_template": question,
            "instantiated_question": question, "dependencies": [],
            "variable_bindings": {}, "answer_type": "entity", "terminal": True,
            "confidence": 0.05, "uncertainty": 0.95,
        }]},
        "generic_coarse_plan_fallback", "deterministic_fallback",
    )


def _identifier(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    return normalized or fallback


def _template_signature(value: str) -> str:
    alpha_normalized = re.sub(r"\$[A-Za-z][A-Za-z0-9_]*", "$variable", value)
    return " ".join(alpha_normalized.casefold().split())


def _root_answer_type(question: str, proposed: str) -> str:
    normalized = _answer_focus_text(question).strip().lower()
    if normalized.startswith("who"):
        return "person"
    if normalized.startswith("when"):
        return "date"
    if normalized.startswith("where"):
        return "location"
    if normalized.startswith("how many") or normalized.startswith("how much"):
        return "number"
    if normalized.startswith("which country") or normalized.startswith("what country"):
        return "country"
    return proposed or "entity"


def _answer_focus_text(question: str) -> str:
    """Use a standalone trailing wh-clause as the final user objective."""
    segments = [
        value.strip() for value in re.split(r"(?<=[.!?])\s+", str(question))
        if value.strip()
    ]
    if len(segments) > 1 and re.match(
        r"^(?:who|when|where|how\s+(?:many|much)|what|which)\b",
        segments[-1], re.IGNORECASE,
    ):
        return segments[-1]
    return str(question)


def _same_answer_focus(left: str, right: str) -> bool:
    aliases = {
        "city": "location", "country": "location", "state": "location",
        "human": "person", "people": "person", "year": "date", "time": "date",
        "count": "number", "quantity": "number",
    }
    canonical_left = aliases.get(str(left).strip().lower(), str(left).strip().lower())
    canonical_right = aliases.get(str(right).strip().lower(), str(right).strip().lower())
    return canonical_left == canonical_right or "entity" in {canonical_left, canonical_right}


def _residual_root_closure(
    question: str, rows: list[dict[str, Any]], root: dict[str, Any],
) -> str | None:
    """Restore an outer relation dropped from a nested coarse plan.

    The root's objective is recursively expanded through its literal variable
    bindings, then located in the original question while ignoring articles.
    A proper embedded span proves that the provider stopped at an inner lookup;
    replacing only that span yields the still-unanswered outer objective.
    """
    dependencies = list(root.get("dependencies", []))
    bindings = dict(root.get("variable_bindings", {}))
    if not dependencies or not bindings:
        return None
    by_id = {str(row.get("node_id")): row for row in rows}
    phrase = _objective_phrase(str(root.get("question_template", "")))
    if not phrase:
        return None
    for variable, source_id in bindings.items():
        source = by_id.get(str(source_id))
        if source is None:
            return None
        source_phrase = _objective_phrase(str(source.get("question_template", "")))
        if not source_phrase or str(variable) not in phrase:
            return None
        phrase = phrase.replace(str(variable), source_phrase)
    span = _ordered_content_span(question, phrase)
    if span is None:
        return None
    start, end = span
    residual = f"{question[:start]}$nested_bridge{question[end:]}"
    content = [
        token.casefold() for token in re.findall(r"[A-Za-z0-9]+", residual)
        if token.casefold() not in {"the", "a", "an"}
    ]
    if len(content) < 5 or not _same_answer_focus(
        _root_answer_type(question, "entity"),
        _root_answer_type(residual, "entity"),
    ):
        return None
    return residual


def _objective_phrase(template: str) -> str:
    value = str(template).strip().rstrip("?.! ")
    value = re.sub(
        r"^(?:what|which|who)\s+(?:(?:is|are|was|were|does|do|did)\s+)?(?:the\s+)?",
        "", value, flags=re.IGNORECASE,
    )
    return value.strip()


def _ordered_content_span(text: str, phrase: str) -> tuple[int, int] | None:
    """Return a literal token-order span, treating articles as transparent."""
    stop = {"the", "a", "an"}
    text_tokens = [
        (match.group(0).casefold(), match.start(), match.end())
        for match in re.finditer(r"[A-Za-z0-9]+", text)
        if match.group(0).casefold() not in stop
    ]
    phrase_tokens = [
        match.group(0).casefold() for match in re.finditer(r"[A-Za-z0-9]+", phrase)
        if match.group(0).casefold() not in stop
    ]
    if not phrase_tokens or len(phrase_tokens) >= len(text_tokens):
        return None
    width = len(phrase_tokens)
    for index in range(len(text_tokens) - width + 1):
        if [row[0] for row in text_tokens[index:index + width]] == phrase_tokens:
            return text_tokens[index][1], text_tokens[index + width - 1][2]
    return None
