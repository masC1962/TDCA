from __future__ import annotations

from typing import Any, List


def _format_evidence_items(evidence_items: List[Any], max_items: int = 6) -> str:
    lines = []
    for i, item in enumerate(evidence_items[:max_items], start=1):
        title = ""
        score = ""
        text = ""
        if item is None:
            continue
        title = str(getattr(item, "metadata", {}) or {}).replace("{", "").replace("}", "")
        md = getattr(item, "metadata", {}) or {}
        if isinstance(md, dict):
            title = str(md.get("title", "")).strip()
        score = f"{getattr(item, 'score', 0.0):.3f}"
        text = str(getattr(item, "text", "")).strip()
        text = " ".join(text.split())
        if len(text) > 500:
            text = text[:500] + " ..."
        prefix = f"[doc_{i}]"
        if title:
            prefix += f" title={title!r}"
        prefix += f" score={score}"
        lines.append(f"{prefix}\n{text}")
    return "\n\n".join(lines) if lines else "(no evidence)"


def _format_memory_items(memory_items: List[Any], max_items: int = 6) -> str:
    lines = []
    for i, item in enumerate(memory_items[:max_items], start=1):
        text = str(getattr(item, "text", "")).strip()
        text = " ".join(text.split())
        if len(text) > 300:
            text = text[:300] + " ..."
        score = f"{getattr(item, 'score', 0.0):.3f}"
        lines.append(f"[mem_{i}] score={score}\n{text}")
    return "\n\n".join(lines) if lines else "(no memory)"


def build_expansion_prompt(
    question: str,
    current_state: str,
    evidence_items: List[Any],
    memory_items: List[Any],
    branching_factor: int,
) -> str:
    evidence_block = _format_evidence_items(evidence_items)
    memory_block = _format_memory_items(memory_items)

    return f"""You are the expansion module of TDCA (Thermal Diffusion Compute Allocation).

Your job:
1. Read the ORIGINAL QUESTION and CURRENT STATE.
2. Use the evidence and memory to decide the next reasoning moves.
3. Return ONLY one JSON object. No markdown. No explanation. No extra text.

ORIGINAL QUESTION:
{question}

CURRENT STATE:
{current_state}

EVIDENCE:
{evidence_block}

MEMORY:
{memory_block}

You must output exactly this schema:
{{
  "sub_questions": [
    {{"text": "sub-question string", "kind": "bridge|retrieval|comparison|verification", "priority": 0.0}}
  ],
  "candidate_answer": "short answer or empty string",
  "stop": false,
  "confidence": 0.0
}}

Rules:
- Return valid JSON only.
- "sub_questions" length must be <= {branching_factor}.
- "priority" must be a float in [0, 1].
- "confidence" must be a float in [0, 1].
- "candidate_answer" must be a SHORT answer only. No explanation. Use "" if unsure.
- Use "verification" only when the answer is nearly grounded already.
- Prefer concrete multi-hop decomposition over meta statements.
- If the question contains comparison, alternatives ("or"), both/same, or a descriptive bridge ("whose"), do NOT return only a single verification step.
- For structurally complex questions, prefer explicit slot-filling sub-questions (one per entity / one intermediate relation / one target attribute).
- For structurally complex questions, output at least two concrete non-verification sub-questions unless the answer is already directly grounded.
- Do NOT output generic steps like "analyze the question", "look at evidence", "find missing entity", "verify strongest evidence".
- Do NOT repeat CURRENT STATE.
- If the answer is already strongly grounded in evidence, set "stop": true and provide "candidate_answer".
- If not grounded enough, set "stop": false and propose targeted sub-questions.

Additional constraints:
- If the question is comparative, create one sub-question per entity plus at most one comparison/verification sub-question.
- If the question is nested multi-hop, first solve the intermediate entity/relation, then the target attribute.
- Keep sub-questions short, concrete, and answerable from the current evidence or the next retrieval step.

Return ONLY the JSON object.
""".strip()



def build_root_plan_prompt(
    question: str,
    evidence_items: List[Any],
    memory_items: List[Any],
) -> str:
    evidence_block = _format_evidence_items(evidence_items, max_items=6)
    memory_block = _format_memory_items(memory_items, max_items=4)
    return f"""You are the root planning module of TDCA.

Your job is to convert the ORIGINAL QUESTION into a small set of persistent reasoning slots/subgoals that must be completed before the final answer is reliable.

ORIGINAL QUESTION:
{question}

EVIDENCE:
{evidence_block}

MEMORY:
{memory_block}

Return ONLY one JSON object with this exact schema:
{{
  "kind": "single_hop|comparison|bridge|nested_bridge|alternative_choice|descriptive_identification|multi_fact",
  "requires_structured_reasoning": false,
  "compose": "direct|compare_yesno|pick_one|attribute_after_bridge|combine_facts",
  "slots": [
    {{"name": "slot_name", "question": "short concrete sub-question", "kind": "bridge|retrieval|comparison", "slot_type": "person|location|country|date|year|boolean|title|position|organization|quantity|unit|generic", "priority": 0.0}}
  ]
}}

Rules:
- Return valid JSON only.
- Use at most 4 slots.
- If the question truly can be answered directly from one short fact, use kind=single_hop, requires_structured_reasoning=false, compose=direct, and slots=[].
- Otherwise set requires_structured_reasoning=true and provide the minimal set of concrete slot questions needed to solve the question.
- Do NOT output verification slots.
- For comparison questions, usually create one slot per entity/side.
- For bridge questions, first solve the intermediate entity/relation, then the target attribute.
- For alternative choice questions containing “A or B”, create candidate-discriminating slots rather than generic verification.
- Slot questions must be short, answerable, and useful as persistent intermediate answers.
- Each slot must include a slot_type that matches the expected answer object.
- Each slot must include a slot_role. Use bridge_entity for intermediate entities, target_attribute for the final attribute after a bridge, left_value/right_value for comparison operands, candidate_a/candidate_b for alternative choices, final_boolean for boolean conclusions, generic otherwise.
- Use depends_on when a slot can only be answered after another slot is solved.
- Set terminal=true only for slots that directly contribute to the final answer composition.
- Use country for nationality/from-country yes-no checks, location for city/neighborhood/place, date or year for birth-time comparison, person for people, title for works/series/games, organization for owners/companies/institutions, unit for military units, position for government/job titles, boolean for yes/no, generic only as a fallback.
- Prefer slot questions that can be directly answered by evidence titles or short passages.

Return ONLY the JSON object.
""".strip()

def build_final_answer_prompt(
    question: str,
    best_state: str,
    evidence_items: List[Any],
    convergence_context: str = "",
) -> str:
    evidence_block = _format_evidence_items(evidence_items, max_items=8)
    convergence_block = convergence_context.strip() or "(no additional high-temperature nodes)"
    return f"""You are the final answer module of TDCA.

Task:
Given the ORIGINAL QUESTION, the best current reasoning state, high-temperature reasoning nodes, intermediate answer candidates, and combined evidence, make the final root-level decision.

QUESTION:
{question}

BEST STATE:
{best_state}

HIGH-TEMPERATURE SUPPORTING NODES:
{convergence_block}

EVIDENCE:
{evidence_block}

Output rules:
- Return ONLY one line.
- Format must be exactly:
Final Answer: <short answer>
- No explanation.
- No chain of thought.
- No citations unless the answer itself requires them.
- Answer the ORIGINAL QUESTION, not a sub-question.
- Use BEST STATE, HIGH-TEMPERATURE SUPPORTING NODES, intermediate answer candidates, and EVIDENCE jointly.
- Intermediate answer candidates are evidence/candidate signals, not automatic final answers.
- Return an intermediate bridge entity only if the ORIGINAL QUESTION itself asks for that entity.
- If several terminal nodes each resolve part of the question, combine them into one root-level final answer.
- For comparison questions, compare the normalized partial answers before deciding Yes or No.
- If the candidates disagree, prefer the shortest answer span directly supported by evidence and matching the ORIGINAL QUESTION's answer type.
- If the terminal subgoals are not sufficiently resolved, return an empty answer rather than guessing from a bridge or intermediate node.
- If the answer is yes/no, output exactly Yes or No.
- If the question asks for a role, function, occupation, or job in a work, output the role/function, not the person who performed it.
- If the question asks for a kind/type/category, output the category, not a named instance, species title, page title, or intermediate entity.
- If the answer is a person, place, year, title, or organization, output only that short span.
- Prefer the canonical/full answer span found in evidence over a shortened alias or surname.
- Prefer the shortest correct span supported by the evidence.

Return ONLY:
Final Answer: <short answer>
""".strip()


def build_answer_judge_prompt(
    question: str,
    evidence_items: List[Any],
    convergence_context: str,
    candidates: List[dict],
) -> str:
    evidence_block = _format_evidence_items(evidence_items, max_items=8)
    convergence_block = convergence_context.strip() or "(no additional high-temperature nodes)"
    candidate_lines = []
    for cand in candidates:
        label = str(cand.get("label", "")).strip()
        answer = str(cand.get("answer", "")).strip()
        source = str(cand.get("source", "")).strip()
        role = str(cand.get("slot_role", "")).strip() or "generic"
        root_aligned = bool(cand.get("root_aligned", False))
        base_score = float(cand.get("base_score", 0.0) or 0.0)
        rerank_score = float(cand.get("rerank_score", base_score) or 0.0)
        coverage = float(cand.get("coverage_ratio", 0.0) or 0.0)
        evidence_score = float(cand.get("span_support", 0.0) or 0.0)
        candidate_lines.append(
            f"{label}. answer={answer!r} source={source} slot_role={role} "
            f"root_aligned={root_aligned} tdca_score={base_score:.3f} "
            f"rerank_score={rerank_score:.3f} coverage={coverage:.3f} evidence={evidence_score:.3f}"
        )
    candidate_block = "\n".join(candidate_lines) if candidate_lines else "(no candidates)"
    return f"""You are the final answer judge for TDCA.

Choose which candidate best answers the ORIGINAL QUESTION at the root level.

ORIGINAL QUESTION:
{question}

HIGH-TEMPERATURE SUPPORTING NODES:
{convergence_block}

EVIDENCE:
{evidence_block}

CANDIDATES:
{candidate_block}

Return ONLY one valid JSON object:
{{
  "choice": "A",
  "confidence": 0.0,
  "reject_all": false,
  "reason": "short reason"
}}

Rules:
- Choose only from the listed candidate labels.
- Judge the ORIGINAL QUESTION, not a sub-question.
- Prefer candidates that directly answer the question's requested type and granularity.
- Penalize intermediate bridge entities, verification answers, and yes/no answers when the original question asks for an entity, person, place, number, date, title, or organization.
- Penalize person names when the original question asks for a role, function, occupation, type, or kind.
- Penalize named instances/page titles when the original question asks for a category or type.
- For yes/no original questions, choose exactly Yes or No if supported.
- For bridge questions, prefer the final target attribute over the intermediate entity.
- For comparison questions, prefer the candidate that results from comparing both sides.
- For title questions, do not choose a person or production role unless the question asks for that person.
- For "near what" questions, prefer the nearby junction/road/place relation over the host mall or city.
- For "another name" questions, choose an alias/name variant, not merely the expanded full name if a separate alias is evidenced.
- Use evidence and supporting nodes jointly; tdca_score is a prior, not a command.
- If no candidate answers the original question, set reject_all=true and choice="".
- Keep reason under 20 words.

Return JSON only.
""".strip()


def build_intermediate_answer_prompt(
    question: str,
    current_state: str,
    evidence_items: List[Any],
    memory_items: List[Any],
    expected_answer_type: str,
) -> str:
    evidence_block = _format_evidence_items(evidence_items, max_items=6)
    memory_block = _format_memory_items(memory_items, max_items=4)
    return f"""You are the intermediate answer module of TDCA.

Your job:
Given the ORIGINAL QUESTION, one CURRENT REASONING STATE, evidence, and memory, produce a short evidence-grounded answer for the CURRENT STATE if possible. This answer will be stored as a reusable reasoning memory for later TDCA steps.

ORIGINAL QUESTION:
{question}

CURRENT REASONING STATE:
{current_state}

EXPECTED ANSWER TYPE FOR CURRENT STATE:
{expected_answer_type}

EVIDENCE:
{evidence_block}

MEMORY:
{memory_block}

Return ONLY one JSON object with this exact schema:
{{
  "intermediate_answer": "short answer or empty string",
  "confidence": 0.0,
  "supports_root": false,
  "next_query": ""
}}

Rules:
- Return valid JSON only.
- "intermediate_answer" must answer the CURRENT REASONING STATE, not a different sub-question.
- Use a short exact span from evidence whenever possible.
- If evidence is insufficient or ambiguous, use an empty string.
- Do not explain.
- Do not include citations unless they are part of the answer.
- "confidence" must be a float in [0, 1].
- "supports_root" is true only if this intermediate answer directly helps answer the ORIGINAL QUESTION.
- "next_query" may be a short follow-up question if another step is needed; otherwise use "".

Return ONLY the JSON object.
""".strip()


def build_scoring_prompt(
    question: str,
    state_text: str,
    evidence_items: List[Any],
    memory_items: List[Any],
) -> str:
    evidence_block = _format_evidence_items(evidence_items, max_items=4)
    memory_block = _format_memory_items(memory_items, max_items=4)

    return f"""You are the local scoring module of TDCA.

You must judge how useful the CURRENT STATE is for solving the QUESTION, given the available evidence and memory.

QUESTION:
{question}

CURRENT STATE:
{state_text}

EVIDENCE:
{evidence_block}

MEMORY:
{memory_block}

Return ONLY one JSON object with this exact schema:
{{
  "task_progress": 0.0,
  "evidence_support": 0.0,
  "memory_usefulness": 0.0,
  "answerability": 0.0,
  "uncertainty": 0.0
}}

Scoring rules:
- Each field must be a float in [0, 1].
- task_progress: how much this state advances the original question.
- evidence_support: how directly the evidence supports this state.
- memory_usefulness: how helpful the retrieved memory is for this state.
- answerability: how likely this state can now be answered correctly.
- uncertainty: how unreliable or ambiguous this state still is.

Important:
- Do not output prose.
- Do not output markdown.
- Do not output comments.
- Return ONLY the JSON object.
""".strip()
