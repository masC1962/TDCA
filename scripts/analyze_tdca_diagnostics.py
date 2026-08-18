#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SAMPLE_ID_KEYS = ["id", "_id", "question_id", "sample_id", "_tdca_source_index"]
JSONL_CANDIDATES = ["summary.jsonl", "results.jsonl", "predictions.jsonl"]
CSV_CANDIDATES = ["summary.csv"]
SCORE_COMPONENT_KEYS = [
    "root_alignment",
    "slot_coverage",
    "evidence_support",
    "dependency_satisfaction",
    "last_hop_support",
    "answer_type_match",
    "chain_compactness",
]
TCC_COMPONENT_KEYS = [
    "path_completeness",
    "dependency_closure",
    "last_hop_entailment",
    "terminality",
    "root_consistency",
    "evidence_grounding",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def as_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [text]
    return [value]


def as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def first_nonempty(row: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def answer_text(row: Dict[str, Any]) -> str:
    return first_nonempty(row, ["pred", "prediction", "final_answer", "answer", "output", "predicted_answer"])


def normalized_answer_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def gold_text(row: Dict[str, Any]) -> str:
    value = row.get("gold_answers")
    if isinstance(value, list) and value:
        return str(value[0])
    return first_nonempty(row, ["gold", "gold_answer", "answer_text", "target", "label"])


def exact_match_value(row: Dict[str, Any]) -> int:
    for key in ["exact_match", "em", "EM"]:
        if key in row:
            return 1 if safe_float(row.get(key), 0.0) >= 0.5 else 0
    return 0


def f1_value(row: Dict[str, Any]) -> float:
    return safe_float(row.get("answer_f1", row.get("f1", 0.0)))


def soft_em_value(row: Dict[str, Any]) -> float:
    return safe_float(row.get("soft_em", 0.0))


def title_hit_value(row: Dict[str, Any]) -> float:
    return safe_float(row.get("title_hit", 0.0))


def sample_key(row: Dict[str, Any]) -> str:
    for key in SAMPLE_ID_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    question = first_nonempty(row, ["question", "query"], "")
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
    return f"question_sha1:{digest}"


def detect_hop_bucket(row: Dict[str, Any]) -> str:
    for key in ["hop_count", "num_hops", "hops"]:
        value = row.get(key)
        if value not in {None, ""}:
            try:
                return f"{int(float(value))}hop"
            except (TypeError, ValueError):
                pass
    for key in ["question_hops", "decomposition"]:
        value = row.get(key)
        if isinstance(value, list) and value:
            return f"{len(value)}hop"
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list) and parsed:
                    return f"{len(parsed)}hop"
            except json.JSONDecodeError:
                pass
    sample_id = first_nonempty(row, ["sample_id", "id", "_id"], "")
    match = re.search(r"(\d+)hop", sample_id, flags=re.I)
    if match:
        return f"{int(match.group(1))}hop"
    return "unknown"


def find_jsonl_file(directory: Path) -> Optional[Path]:
    for name in JSONL_CANDIDATES:
        path = directory / name
        if path.exists():
            return path
    jsonls = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
    return jsonls[0] if jsonls else None


def find_csv_file(directory: Path) -> Optional[Path]:
    for name in CSV_CANDIDATES:
        path = directory / name
        if path.exists():
            return path
    return None


def merge_rows(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not primary:
        return secondary
    by_key = {sample_key(row): row for row in secondary}
    merged: List[Dict[str, Any]] = []
    for row in primary:
        extra = by_key.get(sample_key(row), {})
        out = dict(extra)
        out.update(row)
        merged.append(out)
    return merged


def resolve_run_dir(run_dir: str, output_dir: Path) -> Optional[Path]:
    if not run_dir:
        return None
    raw = Path(run_dir)
    if raw.exists():
        return raw
    text = run_dir.replace("\\", "/")
    marker = "batch_outputs/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        candidate = Path.cwd() / "batch_outputs" / suffix
        if candidate.exists():
            return candidate
    candidate = output_dir / "runs" / raw.name
    if candidate.exists():
        return candidate
    return None


def max_history_score(result: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any], bool]:
    best_score: Optional[float] = None
    best_parts: Dict[str, Any] = {}
    triggered = False
    for event in as_list(result.get("answer_history")):
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind", ""))
        source = str(event.get("source", ""))
        if "score_based_final_chain" in kind or source == "final_chain":
            triggered = triggered or "score_based_final_chain" in kind
        score = event.get("final_chain_score_v2", event.get("final_chain_score"))
        if score is not None and score != "":
            score_f = safe_float(score)
            if best_score is None or score_f > best_score:
                best_score = score_f
                parts = event.get("final_chain_score_v2_components", event.get("final_chain_score_parts"))
                best_parts = parts if isinstance(parts, dict) else {}
    return best_score, best_parts, triggered


def enrich_with_result(row: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    run_dir = resolve_run_dir(str(row.get("run_dir", "")), output_dir)
    if run_dir is None:
        return row
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return row
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return row
    out = dict(row)
    diagnostics = result.get("final_diagnostics")
    if isinstance(diagnostics, dict):
        for key, value in diagnostics.items():
            out.setdefault(key, value)
    best_node = result.get("best_node") or {}
    best_meta = best_node.get("metadata") if isinstance(best_node, dict) else {}
    if isinstance(best_meta, dict):
        out.setdefault("best_node_composition_kind", best_meta.get("composition_kind", ""))
        out.setdefault("final_chain_score", best_meta.get("final_chain_score"))
        out.setdefault("final_chain_score_old", best_meta.get("final_chain_score_old"))
        out.setdefault("final_chain_score_v2", best_meta.get("final_chain_score_v2", best_meta.get("final_chain_score")))
        out.setdefault("score_admission_precondition_passed", best_meta.get("score_admission_precondition_passed"))
        out.setdefault("score_admission_precondition_fail_reasons", best_meta.get("score_admission_precondition_fail_reasons"))
        out.setdefault("last_hop_verification", best_meta.get("last_hop_verification"))
        out.setdefault("bridge_entity_check", best_meta.get("bridge_entity_check"))
        out.setdefault("expected_answer_type", best_meta.get("expected_answer_type"))
        out.setdefault("candidate_answer_type", best_meta.get("candidate_answer_type"))
        out.setdefault("terminal_chain_closure_enabled", best_meta.get("terminal_chain_closure_enabled"))
        out.setdefault("terminal_chain_closure_score", best_meta.get("terminal_chain_closure_score"))
        out.setdefault("terminal_chain_closure_info", best_meta.get("terminal_chain_closure_info"))
        out.setdefault("terminal_chain_closure_gate_passed", best_meta.get("terminal_chain_closure_gate_passed"))
        out.setdefault("terminal_chain_closure_reject_reasons", best_meta.get("terminal_chain_closure_reject_reasons"))
        parts = best_meta.get("final_chain_score_v2_components", best_meta.get("final_chain_score_parts"))
        if isinstance(parts, dict):
            out.setdefault("final_chain_score_parts", parts)
            out.setdefault("final_chain_score_v2_components", parts)
    hist_score, hist_parts, hist_triggered = max_history_score(result)
    if out.get("final_chain_score") in {None, ""} and hist_score is not None:
        out["final_chain_score"] = hist_score
    if not isinstance(out.get("final_chain_score_parts"), dict) and hist_parts:
        out["final_chain_score_parts"] = hist_parts
    if hist_triggered:
        out["score_based_admission_triggered"] = True
    if "anytime_fallback_triggered" not in out:
        out["anytime_fallback_triggered"] = result.get("anytime_fallback_triggered", False)
    return out


def load_output_dir(directory: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    directory = directory.resolve()
    jsonl_path = find_jsonl_file(directory)
    csv_path = find_csv_file(directory)
    json_rows = load_jsonl(jsonl_path) if jsonl_path else []
    csv_rows = load_csv(csv_path) if csv_path else []
    rows = merge_rows(json_rows, csv_rows)
    rows = [enrich_with_result(row, directory) for row in rows]
    return rows, {
        "dir": str(directory),
        "jsonl_path": str(jsonl_path) if jsonl_path else None,
        "csv_path": str(csv_path) if csv_path else None,
        "row_count": len(rows),
    }


def distribution(values: Iterable[Any]) -> Dict[str, int]:
    return {str(k): v for k, v in Counter(values).most_common()}


def numeric_average(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [safe_float(row.get(key), default=math.nan) for row in rows]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def final_chain_score(row: Dict[str, Any]) -> Optional[float]:
    value = row.get("final_chain_score_v2", row.get("final_chain_score"))
    if value is None or value == "":
        return None
    return safe_float(value)


def final_chain_score_old(row: Dict[str, Any]) -> Optional[float]:
    value = row.get("final_chain_score_old")
    if value is None or value == "":
        return None
    return safe_float(value)


def score_parts(row: Dict[str, Any]) -> Dict[str, float]:
    parts = (
        row.get("final_chain_score_v2_components")
        or row.get("final_chain_score_parts")
        or row.get("new_score_components")
        or {}
    )
    parts = as_dict(parts) if isinstance(parts, str) else parts
    if not isinstance(parts, dict):
        return {}
    return {key: safe_float(parts.get(key)) for key in SCORE_COMPONENT_KEYS if key in parts}


def precondition_fail_reasons(row: Dict[str, Any]) -> List[str]:
    return [str(reason) for reason in as_list(row.get("score_admission_precondition_fail_reasons")) if str(reason)]


def last_hop_support(row: Dict[str, Any]) -> Optional[float]:
    info = as_dict(row.get("last_hop_verification"))
    value = info.get("last_hop_support")
    if value is None or value == "":
        parts = score_parts(row)
        value = parts.get("last_hop_support")
    if value is None or value == "":
        return None
    return safe_float(value)


def bridge_rejected(row: Dict[str, Any]) -> bool:
    if "precondition_failed_bridge_entity" in precondition_fail_reasons(row):
        return True
    info = as_dict(row.get("bridge_entity_check"))
    return safe_bool(info.get("is_bridge_entity"))


def answer_type_mismatch_rejected(row: Dict[str, Any]) -> bool:
    if "precondition_failed_answer_type" in precondition_fail_reasons(row):
        return True
    expected = str(row.get("expected_answer_type", "") or "")
    candidate = str(row.get("candidate_answer_type", "") or "")
    return bool(expected and candidate and expected != "unknown" and candidate != "unknown" and expected != candidate)


def average_components(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in score_parts(row).items():
            values[key].append(value)
    return {
        key: (sum(values.get(key, [])) / len(values.get(key, [])) if values.get(key) else None)
        for key in SCORE_COMPONENT_KEYS
    }


def tcc_info(row: Dict[str, Any]) -> Dict[str, Any]:
    return as_dict(row.get("terminal_chain_closure_info"))


def final_candidate_tcc_audit(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = row.get("final_candidate_tcc_audit")
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
            value = parsed
        except json.JSONDecodeError:
            value = []
    return [item for item in as_list(value) if isinstance(item, dict)]


def selected_candidate_tcc(row: Dict[str, Any]) -> Dict[str, Any]:
    return as_dict(row.get("selected_candidate_tcc"))


def tcc_rerank_policy_decision(row: Dict[str, Any]) -> Dict[str, Any]:
    return as_dict(row.get("tcc_rerank_policy_decision"))


def tcc_rerank_applied(row: Dict[str, Any]) -> bool:
    return safe_bool(row.get("tcc_rerank_applied"))


def tcc_promotion_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = row.get("tcc_promotion_candidates")
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    return [item for item in as_list(value) if isinstance(item, dict)]


def tcc_promotion_admitted(row: Dict[str, Any]) -> bool:
    return bool(str(row.get("tcc_promotion_selected_answer", "") or "").strip())


def tcc_promotion_selected_source(row: Dict[str, Any]) -> str:
    return str(row.get("tcc_promotion_selected_source", "") or "").strip().lower()


def root_composed_promotion_enabled(row: Dict[str, Any]) -> bool:
    return safe_bool(row.get("root_composed_promotion_enabled")) or str(row.get("tcc_promotion_policy", "") or "").strip().lower() == "root_composed_only"


def inferred_hop_count(row: Dict[str, Any]) -> Optional[int]:
    value = row.get("inferred_hop_count")
    if value in {None, ""}:
        value = tcc_rerank_policy_decision(row).get("inferred_hop_count")
    if value not in {None, ""}:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            pass
    bucket = detect_hop_bucket(row)
    match = re.match(r"(\d+)hop", bucket)
    return int(match.group(1)) if match else None


def tcc_score(row: Dict[str, Any]) -> Optional[float]:
    value = row.get("terminal_chain_closure_score")
    if value is None or value == "":
        return None
    return safe_float(value)


def tcc_fail_reasons(row: Dict[str, Any]) -> List[str]:
    reasons = [str(reason) for reason in as_list(row.get("terminal_chain_closure_reject_reasons")) if str(reason)]
    info = tcc_info(row)
    for reason in as_list(info.get("closure_fail_reasons")):
        if str(reason):
            reasons.append(str(reason))
    return list(dict.fromkeys(reasons))


def selected_tcc_fail_reasons(row: Dict[str, Any]) -> List[str]:
    selected = selected_candidate_tcc(row)
    reasons = [str(reason) for reason in as_list(selected.get("tcc_reject_reasons")) if str(reason)]
    closure = as_dict(selected.get("closure_info"))
    for reason in as_list(closure.get("closure_fail_reasons")):
        if str(reason):
            reasons.append(str(reason))
    return list(dict.fromkeys(reasons))


def tcc_gate_passed(row: Dict[str, Any]) -> bool:
    return safe_bool(row.get("terminal_chain_closure_gate_passed"))


def average_tcc_components(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        info = tcc_info(row)
        for key in TCC_COMPONENT_KEYS:
            if key in info:
                values[key].append(safe_float(info.get(key)))
    return {
        key: (sum(values.get(key, [])) / len(values.get(key, [])) if values.get(key) else None)
        for key in TCC_COMPONENT_KEYS
    }


def score_distribution(scores: List[float]) -> Dict[str, int]:
    bins = {
        "<0.60": 0,
        "0.60-0.69": 0,
        "0.70-0.79": 0,
        "0.80-0.89": 0,
        ">=0.90": 0,
    }
    for score in scores:
        if score < 0.60:
            bins["<0.60"] += 1
        elif score < 0.70:
            bins["0.60-0.69"] += 1
        elif score < 0.80:
            bins["0.70-0.79"] += 1
        elif score < 0.90:
            bins["0.80-0.89"] += 1
        else:
            bins[">=0.90"] += 1
    return bins


def is_score_based_triggered(row: Dict[str, Any]) -> bool:
    if safe_bool(row.get("score_based_admission_triggered")):
        return True
    kind = str(row.get("best_node_composition_kind", ""))
    return kind == "score_based_final_chain" or "score_based_final_chain" in str(row.get("stop_reason", ""))


def analyze_single(rows: List[Dict[str, Any]], source_info: Dict[str, Any]) -> Dict[str, Any]:
    total = len(rows)
    nonempty_rows = [row for row in rows if answer_text(row)]
    empty_rows = [row for row in rows if not answer_text(row)]
    exact_count = sum(exact_match_value(row) for row in rows)
    final_empty_reason_counter = Counter(str(row.get("final_empty_reason", "") or "") for row in empty_rows)
    terminal_reject_counter: Counter[str] = Counter()
    precondition_fail_counter: Counter[str] = Counter()
    tcc_fail_counter: Counter[str] = Counter()
    for row in rows:
        for reason in as_list(row.get("terminal_reject_reasons")):
            terminal_reject_counter[str(reason)] += 1
        for reason in precondition_fail_reasons(row):
            precondition_fail_counter[reason] += 1
        for reason in tcc_fail_reasons(row):
            tcc_fail_counter[reason] += 1

    hop_buckets: Dict[str, Dict[str, Any]] = {}
    by_hop: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_hop[detect_hop_bucket(row)].append(row)
    for bucket, bucket_rows in sorted(by_hop.items()):
        n = len(bucket_rows)
        hop_buckets[bucket] = {
            "total": n,
            "nonempty": sum(1 for row in bucket_rows if answer_text(row)),
            "EM": sum(exact_match_value(row) for row in bucket_rows),
            "average_f1": sum(f1_value(row) for row in bucket_rows) / n if n else 0.0,
            "average_goal_completion": sum(safe_float(row.get("goal_completion")) for row in bucket_rows) / n if n else 0.0,
            "average_final_candidate_count": sum(safe_float(row.get("final_candidate_count")) for row in bucket_rows) / n if n else 0.0,
        }

    scores = [score for row in rows for score in [final_chain_score(row)] if score is not None]
    old_scores = [score for row in rows for score in [final_chain_score_old(row)] if score is not None]
    last_hop_scores = [score for row in rows for score in [last_hop_support(row)] if score is not None]
    component_values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in score_parts(row).items():
            component_values[key].append(value)
    score_based_rows = [row for row in rows if is_score_based_triggered(row)]
    fallback_rows = [row for row in rows if safe_bool(row.get("anytime_fallback_triggered"))]
    tcc_rows = [row for row in rows if safe_bool(row.get("terminal_chain_closure_enabled"))]
    tcc_admitted_rows = [row for row in tcc_rows if tcc_gate_passed(row) and answer_text(row)]
    tcc_rejected_rows = [row for row in tcc_rows if not tcc_gate_passed(row) and (tcc_fail_reasons(row) or tcc_score(row))]
    audit_rows = [row for row in rows if safe_bool(row.get("tcc_final_audit_enabled"))]
    audit_records = [record for row in audit_rows for record in final_candidate_tcc_audit(row)]
    audit_downgraded_records = [
        record for record in audit_records
        if safe_float(record.get("tcc_penalty")) > 0.0 or not safe_bool(record.get("tcc_passed"))
    ]
    audit_changed_rows = [row for row in audit_rows if safe_bool(row.get("tcc_final_audit_changed_answer"))]
    rerank_mode_rows = [
        row for row in audit_rows
        if str(row.get("tcc_final_audit_mode", "") or "").strip().lower() == "rerank"
    ]
    rerank_policy_rows = [
        row for row in rerank_mode_rows
        if "tcc_rerank_applied" in row
        or "tcc_rerank_policy" in row
        or "tcc_rerank_policy_decision" in row
    ]
    rerank_applied_rows = [row for row in rerank_policy_rows if tcc_rerank_applied(row)]
    rerank_skipped_rows = [row for row in rerank_policy_rows if not tcc_rerank_applied(row)]
    short_hop_protected_rows = [
        row for row in rerank_skipped_rows
        if str(row.get("tcc_rerank_skip_reason", "") or "") == "skipped_short_hop_protected"
        or str(tcc_rerank_policy_decision(row).get("selected_reason", "") or "") == "skipped_short_hop_protected"
    ]
    longhop_rerank_rows = [
        row for row in rerank_applied_rows
        if str(tcc_rerank_policy_decision(row).get("selected_reason", "") or "") == "longhop"
    ]
    weak_candidate_rerank_rows = [
        row for row in rerank_applied_rows
        if str(tcc_rerank_policy_decision(row).get("selected_reason", "") or "") == "weak_candidate"
    ]
    two_hop_rerank_rows = [
        row for row in rerank_applied_rows
        if (inferred_hop_count(row) or 0) <= 2
    ]
    long_hop_rerank_rows = [
        row for row in rerank_applied_rows
        if (inferred_hop_count(row) or 0) >= 3
    ]
    promotion_triggered_rows = [row for row in rows if safe_bool(row.get("tcc_verified_promotion_triggered"))]
    promotion_admitted_rows = [row for row in promotion_triggered_rows if tcc_promotion_admitted(row)]
    promotion_candidate_records = [record for row in promotion_triggered_rows for record in tcc_promotion_candidates(row)]
    tmc_rows = [row for row in rows if safe_bool(row.get("terminal_memory_consolidation_enabled"))]
    tmc_triggered_rows = [
        row for row in tmc_rows
        if (
            safe_bool(row.get("tmc_triggered"))
            if "tmc_triggered" in row
            else safe_bool(row.get("terminal_memory_consolidation_enabled"))
        )
    ]
    tmc_entered_rows = [row for row in tmc_rows if safe_bool(row.get("tmc_entered_final_candidate"))]
    tmc_selected_rows = [row for row in tmc_rows if safe_bool(row.get("tmc_candidate_selected"))]
    tmc_entry_fail_counter = Counter(
        str(row.get("tmc_final_candidate_entry_fail_reason", "") or "")
        for row in tmc_rows
        if not safe_bool(row.get("tmc_entered_final_candidate"))
        and str(row.get("tmc_final_candidate_entry_fail_reason", "") or "")
    )
    imc_rows = [row for row in rows if safe_bool(row.get("iterative_memory_construction_enabled"))]
    promotion_fail_counter: Counter[str] = Counter()
    for record in promotion_candidate_records:
        if safe_bool(record.get("passed")):
            continue
        for reason in as_list(record.get("fail_reasons")):
            if str(reason):
                promotion_fail_counter[str(reason)] += 1
    promotion_source_stats: Dict[str, Dict[str, Any]] = {}
    for source in sorted({tcc_promotion_selected_source(row) or "unknown" for row in promotion_admitted_rows}):
        source_rows = [row for row in promotion_admitted_rows if (tcc_promotion_selected_source(row) or "unknown") == source]
        correct = sum(exact_match_value(row) for row in source_rows)
        promotion_source_stats[source] = {
            "admitted": len(source_rows),
            "correct": correct,
            "precision": correct / len(source_rows) if source_rows else None,
        }
    promotion_hop_stats: Dict[str, Dict[str, Any]] = {}
    for hop in [2, 3, 4]:
        hop_rows = [row for row in promotion_admitted_rows if (inferred_hop_count(row) or 0) == hop]
        correct = sum(exact_match_value(row) for row in hop_rows)
        promotion_hop_stats[f"{hop}hop"] = {
            "admitted": len(hop_rows),
            "correct": correct,
            "precision": correct / len(hop_rows) if hop_rows else None,
        }
    root_composed_triggered_rows = [row for row in promotion_triggered_rows if root_composed_promotion_enabled(row)]
    root_composed_sources = {"root_memory", "composed_root_memory", "rejected_root_candidate", "rejected_score_candidate_root_level"}
    root_composed_admitted_rows = [
        row for row in promotion_admitted_rows
        if tcc_promotion_selected_source(row) in root_composed_sources
    ]
    gray_zone_admitted_rows = [row for row in promotion_admitted_rows if safe_bool(row.get("gray_zone_promotion_used"))]
    gray_zone_candidate_rows = [
        row for row in promotion_admitted_rows
        if any(safe_bool(record.get("gray_zone_promotion_used")) for record in tcc_promotion_candidates(row))
    ]
    buffer_blocked_records = [
        record for record in promotion_candidate_records
        if str(record.get("source", "") or "").strip().lower() == "buffer"
        and (
            safe_bool(record.get("raw_buffer_promotion_blocked"))
            or "raw_buffer_promotion_blocked" in {str(reason) for reason in as_list(record.get("fail_reasons"))}
        )
    ]
    selected_tcc_scores = [
        safe_float(selected_candidate_tcc(row).get("terminal_chain_closure_score"), default=math.nan)
        for row in rows
        if selected_candidate_tcc(row)
    ]
    selected_tcc_scores = [v for v in selected_tcc_scores if not math.isnan(v)]
    selected_fail_counter: Counter[str] = Counter()
    for row in rows:
        for reason in selected_tcc_fail_reasons(row):
            selected_fail_counter[reason] += 1

    return {
        "source": source_info,
        "overall": {
            "total": total,
            "nonempty": len(nonempty_rows),
            "empty": len(empty_rows),
            "exact_match_count": exact_count,
            "exact_match_rate": exact_count / total if total else 0.0,
            "average_f1": sum(f1_value(row) for row in rows) / total if total else 0.0,
            "average_soft_em": sum(soft_em_value(row) for row in rows) / total if total else 0.0,
            "average_title_hit": sum(title_hit_value(row) for row in rows) / total if total else 0.0,
            "average_steps": sum(safe_float(row.get("steps")) for row in rows) / total if total else 0.0,
            "average_llm_calls": sum(safe_float(row.get("llm_calls")) for row in rows) / total if total else 0.0,
            "average_generated_tokens": sum(safe_float(row.get("generated_tokens")) for row in rows) / total if total else 0.0,
        },
        "empty_answer_stats": {
            "empty_count": len(empty_rows),
            "final_empty_reason_counter": dict(final_empty_reason_counter.most_common()),
            "terminal_reject_reasons_counter": dict(terminal_reject_counter.most_common()),
            "score_admission_precondition_fail_reason_counter": dict(precondition_fail_counter.most_common()),
            "terminal_chain_closure_fail_reason_counter": dict(tcc_fail_counter.most_common()),
            "root_memory_exists_but_empty_count": sum(1 for row in empty_rows if safe_bool(row.get("root_memory_exists"))),
            "anytime_exists_but_empty_count": sum(1 for row in empty_rows if first_nonempty(row, ["anytime_answer"])),
            "composed_root_exists_but_empty_count": sum(1 for row in empty_rows if safe_bool(row.get("has_composed_root_memory"))),
            "final_candidate_count_distribution": distribution(row.get("final_candidate_count", "") for row in rows),
        },
        "by_hop": hop_buckets,
        "mechanisms": {
            "final_chain_buffer_enabled": any(safe_bool(row.get("final_chain_buffer_enabled")) for row in rows),
            "score_admission_enabled": any(safe_bool(row.get("score_based_final_admission_enabled")) for row in rows),
            "anytime_fallback_enabled": any(safe_bool(row.get("anytime_fallback_triggered")) for row in rows),
            "score_based_admission_triggered_count": len(score_based_rows),
            "score_based_admission_correct_count": sum(exact_match_value(row) for row in score_based_rows),
            "terminal_chain_closure_enabled": any(safe_bool(row.get("terminal_chain_closure_enabled")) for row in rows),
            "tcc_triggered_count": len(tcc_rows),
            "tcc_admitted_count": len(tcc_admitted_rows),
            "tcc_rejected_count": len(tcc_rejected_rows),
            "tcc_admitted_correct_count": sum(exact_match_value(row) for row in tcc_admitted_rows),
            "tcc_precision": (
                sum(exact_match_value(row) for row in tcc_admitted_rows) / len(tcc_admitted_rows)
                if tcc_admitted_rows else None
            ),
            "average_tcc_score": (
                sum(score for row in tcc_rows for score in [tcc_score(row)] if score is not None)
                / len([score for row in tcc_rows for score in [tcc_score(row)] if score is not None])
                if [score for row in tcc_rows for score in [tcc_score(row)] if score is not None] else None
            ),
            "tcc_closure_fail_reasons": dict(tcc_fail_counter.most_common()),
            "tcc_final_audit_enabled": any(safe_bool(row.get("tcc_final_audit_enabled")) for row in rows),
            "tcc_final_audit_mode_distribution": distribution(row.get("tcc_final_audit_mode", "") for row in audit_rows),
            "tcc_final_audit_candidate_count": len(audit_records),
            "tcc_final_audit_downgraded_count": len(audit_downgraded_records),
            "tcc_final_audit_changed_answer_count": len(audit_changed_rows),
            "tcc_rerank_policy_distribution": distribution(row.get("tcc_rerank_policy", "") for row in rerank_policy_rows),
            "tcc_rerank_applied_count": len(rerank_applied_rows),
            "tcc_rerank_skipped_count": len(rerank_skipped_rows),
            "skipped_short_hop_protected_count": len(short_hop_protected_rows),
            "longhop_rerank_count": len(longhop_rerank_rows),
            "weak_candidate_rerank_count": len(weak_candidate_rerank_rows),
            "two_hop_rerank_applied_count": len(two_hop_rerank_rows),
            "long_hop_rerank_applied_count": len(long_hop_rerank_rows),
            "tcc_verified_promotion_enabled": any(safe_bool(row.get("tcc_verified_promotion_enabled")) for row in rows),
            "tcc_verified_promotion_triggered_count": len(promotion_triggered_rows),
            "tcc_verified_promotion_admitted_count": len(promotion_admitted_rows),
            "tcc_verified_promotion_correct_count": sum(exact_match_value(row) for row in promotion_admitted_rows),
            "tcc_verified_promotion_precision": (
                sum(exact_match_value(row) for row in promotion_admitted_rows) / len(promotion_admitted_rows)
                if promotion_admitted_rows else None
            ),
            "promotion_source_distribution": distribution(row.get("tcc_promotion_selected_source", "") for row in promotion_admitted_rows),
            "promotion_source_precision": promotion_source_stats,
            "promotion_hop_precision": promotion_hop_stats,
            "terminal_memory_consolidation_enabled": bool(tmc_rows),
            "tmc_triggered_count": len(tmc_triggered_rows),
            "tmc_terminal_memory_count_total": sum(int(safe_float(row.get("terminal_memory_count"), default=0.0)) for row in tmc_rows),
            "tmc_terminal_memory_count_average": (
                sum(safe_float(row.get("terminal_memory_count"), default=0.0) for row in tmc_rows) / len(tmc_rows)
                if tmc_rows else 0.0
            ),
            "tmc_tcc_closed_count": sum(int(safe_float(row.get("tmc_tcc_closed_count"), default=0.0)) for row in tmc_rows),
            "tmc_entered_final_candidate_count": len(tmc_entered_rows),
            "tmc_candidate_selected_count": len(tmc_selected_rows),
            "tmc_candidate_selected_correct_count": sum(exact_match_value(row) for row in tmc_selected_rows),
            "tmc_candidate_precision": (
                sum(exact_match_value(row) for row in tmc_selected_rows) / len(tmc_selected_rows)
                if tmc_selected_rows else None
            ),
            "tmc_final_candidate_entry_fail_reason_counter": dict(tmc_entry_fail_counter.most_common()),
            "memory_repair_goal_count": sum(len(as_list(row.get("memory_repair_goals"))) for row in tmc_rows),
            "iterative_memory_construction_enabled": bool(imc_rows),
            "imc_triggered_count": len([row for row in imc_rows if safe_float(row.get("imc_rounds_executed"), default=0.0) > 0]),
            "imc_rounds_executed_total": sum(int(safe_float(row.get("imc_rounds_executed"), default=0.0)) for row in imc_rows),
            "root_composed_promotion_triggered_count": len(root_composed_triggered_rows),
            "root_composed_promotion_admitted_count": len(root_composed_admitted_rows),
            "root_composed_promotion_correct_count": sum(exact_match_value(row) for row in root_composed_admitted_rows),
            "root_composed_promotion_precision": (
                sum(exact_match_value(row) for row in root_composed_admitted_rows) / len(root_composed_admitted_rows)
                if root_composed_admitted_rows else None
            ),
            "buffer_blocked_count": len(buffer_blocked_records),
            "gray_zone_promotion_count": len(gray_zone_admitted_rows or gray_zone_candidate_rows),
            "gray_zone_promotion_correct_count": sum(exact_match_value(row) for row in (gray_zone_admitted_rows or gray_zone_candidate_rows)),
            "promotion_fail_reason_distribution": dict(promotion_fail_counter.most_common()),
            "tcc_final_audit_tcc_none_nonempty_count": sum(
                1 for row in nonempty_rows
                if safe_bool(row.get("tcc_final_audit_enabled")) and not selected_candidate_tcc(row)
            ),
            "selected_candidate_tcc_score_average": (
                sum(selected_tcc_scores) / len(selected_tcc_scores) if selected_tcc_scores else None
            ),
            "selected_candidate_tcc_fail_reasons": dict(selected_fail_counter.most_common()),
            "bridge_entity_rejected_count": sum(1 for row in rows if bridge_rejected(row)),
            "answer_type_mismatch_rejected_count": sum(1 for row in rows if answer_type_mismatch_rejected(row)),
            "anytime_fallback_triggered_count": len(fallback_rows),
            "anytime_fallback_correct_count": sum(exact_match_value(row) for row in fallback_rows),
            "average_final_chain_score": sum(scores) / len(scores) if scores else None,
            "average_final_chain_score_old": sum(old_scores) / len(old_scores) if old_scores else None,
            "average_final_chain_score_v2": sum(scores) / len(scores) if scores else None,
            "average_last_hop_support": sum(last_hop_scores) / len(last_hop_scores) if last_hop_scores else None,
            "final_chain_score_distribution": score_distribution(scores),
            "score_components_average": {
                key: (sum(component_values.get(key, [])) / len(component_values.get(key, [])) if component_values.get(key, []) else None)
                for key in SCORE_COMPONENT_KEYS
            },
        },
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    return {
        "total": total,
        "nonempty": sum(1 for row in rows if answer_text(row)),
        "empty": sum(1 for row in rows if not answer_text(row)),
        "EM": sum(exact_match_value(row) for row in rows),
        "F1": sum(f1_value(row) for row in rows) / total if total else 0.0,
    }


def case_record(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": sample_key(new or base),
        "question": first_nonempty(new or base, ["question", "query"]),
        "gold_answer": gold_text(new or base),
        "base_pred": answer_text(base),
        "new_pred": answer_text(new),
        "base_exact_match": exact_match_value(base),
        "new_exact_match": exact_match_value(new),
        "base_f1": f1_value(base),
        "new_f1": f1_value(new),
        "base_final_empty_reason": base.get("final_empty_reason", ""),
        "new_final_empty_reason": new.get("final_empty_reason", ""),
        "new_final_chain_score_old": final_chain_score_old(new),
        "new_final_chain_score_v2": final_chain_score(new),
        "new_score_components_v2": score_parts(new),
        "new_last_hop_verification": as_dict(new.get("last_hop_verification")),
        "new_bridge_entity_check": as_dict(new.get("bridge_entity_check")),
        "new_terminal_chain_closure_score": tcc_score(new),
        "new_terminal_chain_closure_info": tcc_info(new),
        "new_terminal_chain_closure_gate_passed": new.get("terminal_chain_closure_gate_passed", ""),
        "new_terminal_chain_closure_reject_reasons": tcc_fail_reasons(new),
        "new_final_candidate_tcc_audit": final_candidate_tcc_audit(new),
        "new_selected_candidate_tcc": selected_candidate_tcc(new),
        "new_tcc_final_audit_changed_answer": new.get("tcc_final_audit_changed_answer", ""),
        "new_tcc_rerank_policy": new.get("tcc_rerank_policy", ""),
        "new_tcc_rerank_applied": new.get("tcc_rerank_applied", ""),
        "new_tcc_rerank_skip_reason": new.get("tcc_rerank_skip_reason", ""),
        "new_tcc_rerank_policy_decision": tcc_rerank_policy_decision(new),
        "new_tcc_verified_promotion_triggered": new.get("tcc_verified_promotion_triggered", ""),
        "new_tcc_promotion_trigger_reason": new.get("tcc_promotion_trigger_reason", ""),
        "new_tcc_promotion_candidate_count": new.get("tcc_promotion_candidate_count", ""),
        "new_tcc_promotion_selected_answer": new.get("tcc_promotion_selected_answer", ""),
        "new_tcc_promotion_selected_source": new.get("tcc_promotion_selected_source", ""),
        "new_tcc_promotion_selected_score": new.get("tcc_promotion_selected_score", ""),
        "new_promotion_side_effect_free": new.get("promotion_side_effect_free", ""),
        "new_original_final_answer_before_promotion": new.get("original_final_answer_before_promotion", ""),
        "new_final_answer_after_promotion": new.get("final_answer_after_promotion", ""),
        "new_promotion_changed_answer": new.get("promotion_changed_answer", ""),
        "new_promotion_changed_answer_reason": new.get("promotion_changed_answer_reason", ""),
        "new_root_composed_promotion_enabled": new.get("root_composed_promotion_enabled", ""),
        "new_promotion_source_allowed": new.get("promotion_source_allowed", ""),
        "new_promotion_source_reject_reason": new.get("promotion_source_reject_reason", ""),
        "new_root_level_metadata_found": new.get("root_level_metadata_found", ""),
        "new_goal_completion_for_promotion": new.get("goal_completion_for_promotion", ""),
        "new_gray_zone_promotion_used": new.get("gray_zone_promotion_used", ""),
        "new_raw_buffer_promotion_blocked": new.get("raw_buffer_promotion_blocked", ""),
        "new_expected_answer_type": new.get("expected_answer_type", ""),
        "new_candidate_answer_type": new.get("candidate_answer_type", ""),
        "new_precondition_passed": new.get("score_admission_precondition_passed", ""),
        "new_precondition_fail_reasons": precondition_fail_reasons(new),
        "new_terminal_reject_reasons": as_list(new.get("terminal_reject_reasons")),
    }


def counter_delta(base_counter: Dict[str, int], new_counter: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    keys = sorted(set(base_counter) | set(new_counter))
    return {
        key: {
            "base": int(base_counter.get(key, 0)),
            "new": int(new_counter.get(key, 0)),
            "delta": int(new_counter.get(key, 0)) - int(base_counter.get(key, 0)),
        }
        for key in keys
    }


def compare_outputs(
    base_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    base_info: Dict[str, Any],
    new_info: Dict[str, Any],
) -> Dict[str, Any]:
    base_by_id = {sample_key(row): row for row in base_rows}
    new_by_id = {sample_key(row): row for row in new_rows}
    aligned_ids = sorted(set(base_by_id) & set(new_by_id))
    aligned = [(base_by_id[key], new_by_id[key]) for key in aligned_ids]

    rescued = [(b, n) for b, n in aligned if not answer_text(b) and answer_text(n)]
    harmed_empty = [(b, n) for b, n in aligned if answer_text(b) and not answer_text(n)]
    base_correct_new_wrong = [(b, n) for b, n in aligned if exact_match_value(b) and not exact_match_value(n)]
    base_wrong_new_correct = [(b, n) for b, n in aligned if not exact_match_value(b) and exact_match_value(n)]
    both_empty = [(b, n) for b, n in aligned if not answer_text(b) and not answer_text(n)]
    both_correct = [(b, n) for b, n in aligned if exact_match_value(b) and exact_match_value(n)]
    both_wrong = [(b, n) for b, n in aligned if answer_text(b) and answer_text(n) and not exact_match_value(b) and not exact_match_value(n)]

    base_single = analyze_single(base_rows, base_info)
    new_single = analyze_single(new_rows, new_info)
    rescued_records = [case_record(b, n) for b, n in rescued]
    harmed_records = [case_record(b, n) for b, n in harmed_empty + base_correct_new_wrong]
    improved_records = [case_record(b, n) for b, n in base_wrong_new_correct]

    by_hop: Dict[str, Any] = {}
    hop_keys = sorted(set(base_single["by_hop"]) | set(new_single["by_hop"]))
    for key in hop_keys:
        base_hop = base_single["by_hop"].get(key, {})
        new_hop = new_single["by_hop"].get(key, {})
        by_hop[key] = {
            "base_nonempty": base_hop.get("nonempty", 0),
            "new_nonempty": new_hop.get("nonempty", 0),
            "base_EM": base_hop.get("EM", 0),
            "new_EM": new_hop.get("EM", 0),
            "base_F1": base_hop.get("average_f1", 0.0),
            "new_F1": new_hop.get("average_f1", 0.0),
            "base_avg_goal_completion": base_hop.get("average_goal_completion", 0.0),
            "new_avg_goal_completion": new_hop.get("average_goal_completion", 0.0),
            "base_avg_final_candidate_count": base_hop.get("average_final_candidate_count", 0.0),
            "new_avg_final_candidate_count": new_hop.get("average_final_candidate_count", 0.0),
        }

    new_score_rows = [row for row in new_rows if is_score_based_triggered(row)]
    new_fallback_rows = [row for row in new_rows if safe_bool(row.get("anytime_fallback_triggered"))]
    new_audit_changed_pairs = [(b, n) for b, n in aligned if safe_bool(n.get("tcc_final_audit_changed_answer"))]
    new_rerank_applied_pairs = [(b, n) for b, n in aligned if tcc_rerank_applied(n)]
    new_rerank_changed_pairs = [
        (b, n) for b, n in new_rerank_applied_pairs
        if normalized_answer_text(answer_text(b)) != normalized_answer_text(answer_text(n))
    ]
    promotion_pairs = [(b, n) for b, n in aligned if safe_bool(n.get("tcc_verified_promotion_triggered"))]
    promotion_admitted_pairs = [(b, n) for b, n in promotion_pairs if tcc_promotion_admitted(n)]
    tmc_pairs = [(b, n) for b, n in aligned if safe_bool(n.get("terminal_memory_consolidation_enabled"))]
    tmc_triggered_pairs = [
        (b, n) for b, n in tmc_pairs
        if (
            safe_bool(n.get("tmc_triggered"))
            if "tmc_triggered" in n
            else safe_bool(n.get("terminal_memory_consolidation_enabled"))
        )
    ]
    tmc_entered_pairs = [(b, n) for b, n in tmc_pairs if safe_bool(n.get("tmc_entered_final_candidate"))]
    tmc_selected_pairs = [(b, n) for b, n in tmc_pairs if safe_bool(n.get("tmc_candidate_selected"))]
    tmc_entry_fail_counter = Counter(
        str(n.get("tmc_final_candidate_entry_fail_reason", "") or "")
        for _, n in tmc_pairs
        if not safe_bool(n.get("tmc_entered_final_candidate"))
        and str(n.get("tmc_final_candidate_entry_fail_reason", "") or "")
    )
    imc_pairs = [(b, n) for b, n in aligned if safe_bool(n.get("iterative_memory_construction_enabled"))]
    empty_to_nonempty_by_promotion = [
        (b, n) for b, n in promotion_admitted_pairs
        if not answer_text(b) and answer_text(n)
    ]
    promotion_changed_to_wrong = [
        (b, n) for b, n in promotion_admitted_pairs
        if exact_match_value(b) and not exact_match_value(n)
    ]
    promotion_correct_pairs = [(b, n) for b, n in promotion_admitted_pairs if exact_match_value(n)]
    promotion_wrong_pairs = [(b, n) for b, n in promotion_admitted_pairs if not exact_match_value(n)]
    promotion_not_admitted_but_answer_changed = [
        (b, n) for b, n in promotion_pairs
        if not tcc_promotion_admitted(n) and safe_bool(n.get("promotion_changed_answer"))
    ]
    promotion_source_precision: Dict[str, Dict[str, Any]] = {}
    for source in sorted({tcc_promotion_selected_source(n) or "unknown" for _, n in promotion_admitted_pairs}):
        pairs = [(b, n) for b, n in promotion_admitted_pairs if (tcc_promotion_selected_source(n) or "unknown") == source]
        correct = sum(exact_match_value(n) for _, n in pairs)
        promotion_source_precision[source] = {
            "admitted": len(pairs),
            "correct": correct,
            "precision": correct / len(pairs) if pairs else None,
        }
    promotion_hop_precision: Dict[str, Dict[str, Any]] = {}
    for hop in [2, 3, 4]:
        pairs = [
            (b, n) for b, n in promotion_admitted_pairs
            if (inferred_hop_count(n) or inferred_hop_count(b) or 0) == hop
        ]
        correct = sum(exact_match_value(n) for _, n in pairs)
        promotion_hop_precision[f"{hop}hop"] = {
            "admitted": len(pairs),
            "correct": correct,
            "precision": correct / len(pairs) if pairs else None,
        }
    root_composed_sources = {"root_memory", "composed_root_memory", "rejected_root_candidate", "rejected_score_candidate_root_level"}
    root_composed_promotion_pairs = [(b, n) for b, n in promotion_pairs if root_composed_promotion_enabled(n)]
    root_composed_promotion_admitted_pairs = [
        (b, n) for b, n in promotion_admitted_pairs
        if tcc_promotion_selected_source(n) in root_composed_sources
    ]
    root_composed_promotion_correct_pairs = [
        (b, n) for b, n in root_composed_promotion_admitted_pairs if exact_match_value(n)
    ]
    gray_zone_promotion_pairs = [
        (b, n) for b, n in promotion_admitted_pairs
        if safe_bool(n.get("gray_zone_promotion_used"))
        or any(safe_bool(record.get("gray_zone_promotion_used")) for record in tcc_promotion_candidates(n))
    ]
    buffer_blocked_records = [
        record for _, n in promotion_pairs
        for record in tcc_promotion_candidates(n)
        if str(record.get("source", "") or "").strip().lower() == "buffer"
        and (
            safe_bool(record.get("raw_buffer_promotion_blocked"))
            or "raw_buffer_promotion_blocked" in {str(reason) for reason in as_list(record.get("fail_reasons"))}
        )
    ]
    two_hop_pairs = [
        (b, n) for b, n in aligned
        if (inferred_hop_count(n) or inferred_hop_count(b) or 0) <= 2
    ]
    two_hop_correct_to_empty = [
        (b, n) for b, n in two_hop_pairs
        if exact_match_value(b) and not answer_text(n)
    ]
    two_hop_correct_to_wrong = [
        (b, n) for b, n in two_hop_pairs
        if exact_match_value(b) and answer_text(n) and not exact_match_value(n)
    ]
    rescued_correct = [n for _, n in rescued if exact_match_value(n)]
    rescued_wrong = [n for _, n in rescued if not exact_match_value(n)]
    score_precision = (
        sum(exact_match_value(row) for row in new_score_rows) / len(new_score_rows)
        if new_score_rows else None
    )
    fallback_precision = (
        sum(exact_match_value(row) for row in new_fallback_rows) / len(new_fallback_rows)
        if new_fallback_rows else None
    )

    base_reason = base_single["empty_answer_stats"]["final_empty_reason_counter"]
    new_reason = new_single["empty_answer_stats"]["final_empty_reason_counter"]
    base_reject = base_single["empty_answer_stats"]["terminal_reject_reasons_counter"]
    new_reject = new_single["empty_answer_stats"]["terminal_reject_reasons_counter"]

    return {
        "base_source": base_info,
        "new_source": new_info,
        "overall_comparison": {
            "base_total": len(base_rows),
            "new_total": len(new_rows),
            "aligned_count": len(aligned),
            "base_nonempty": base_single["overall"]["nonempty"],
            "new_nonempty": new_single["overall"]["nonempty"],
            "base_EM": base_single["overall"]["exact_match_count"],
            "new_EM": new_single["overall"]["exact_match_count"],
            "base_F1": base_single["overall"]["average_f1"],
            "new_F1": new_single["overall"]["average_f1"],
            "base_empty": base_single["overall"]["empty"],
            "new_empty": new_single["overall"]["empty"],
        },
        "rescued_samples": {
            "base_empty_new_nonempty_count": len(rescued),
            "base_empty_new_nonempty_correct_count": sum(exact_match_value(n) for _, n in rescued),
            "base_empty_new_nonempty_avg_f1": sum(f1_value(n) for _, n in rescued) / len(rescued) if rescued else 0.0,
            "list": rescued_records,
        },
        "harmed_samples": {
            "base_nonempty_new_empty_count": len(harmed_empty),
            "base_correct_new_wrong_count": len(base_correct_new_wrong),
            "base_correct_new_wrong_list": [case_record(b, n) for b, n in base_correct_new_wrong],
            "list": harmed_records,
        },
        "improved_samples": {
            "base_wrong_new_correct_count": len(base_wrong_new_correct),
            "base_wrong_new_correct_list": improved_records,
        },
        "unchanged_samples": {
            "both_empty_count": len(both_empty),
            "both_correct_count": len(both_correct),
            "both_wrong_count": len(both_wrong),
        },
        "empty_reason_changes": {
            "final_empty_reason_delta": counter_delta(base_reason, new_reason),
            "terminal_reject_reasons_delta": counter_delta(base_reject, new_reject),
            "tracked_failures": {
                key: {
                    "base": int(base_reason.get(key, 0)),
                    "new": int(new_reason.get(key, 0)),
                    "delta": int(new_reason.get(key, 0)) - int(base_reason.get(key, 0)),
                }
                for key in [
                    "no_root_memory_or_final_candidates",
                    "no_final_candidates_but_anytime_exists",
                    "root_memory_rejected_before_candidate_collection",
                ]
            },
        },
        "by_hop_comparison": by_hop,
        "mechanism_quality": {
            "score_based_admission_triggered_count": len(new_score_rows),
            "score_based_admission_correct_count": sum(exact_match_value(row) for row in new_score_rows),
            "score_based_admission_precision": score_precision,
            "rescued_correct_score_component_average": average_components(rescued_correct),
            "rescued_wrong_score_component_average": average_components(rescued_wrong),
            "rescued_correct_tcc_component_average": average_tcc_components(rescued_correct),
            "rescued_wrong_tcc_component_average": average_tcc_components(rescued_wrong),
            "tcc_final_audit_changed_answer_count": len(new_audit_changed_pairs),
            "tcc_final_audit_changed_to_correct_count": sum(1 for b, n in new_audit_changed_pairs if not exact_match_value(b) and exact_match_value(n)),
            "tcc_final_audit_changed_to_wrong_count": sum(1 for b, n in new_audit_changed_pairs if exact_match_value(b) and not exact_match_value(n)),
            "tcc_rerank_applied_count": len(new_rerank_applied_pairs),
            "tcc_rerank_changed_answer_count": len(new_rerank_changed_pairs),
            "tcc_rerank_changed_to_correct_count": sum(1 for b, n in new_rerank_changed_pairs if not exact_match_value(b) and exact_match_value(n)),
            "tcc_rerank_changed_to_wrong_count": sum(1 for b, n in new_rerank_changed_pairs if exact_match_value(b) and not exact_match_value(n)),
            "two_hop_rerank_applied_count": sum(1 for _, n in new_rerank_applied_pairs if (inferred_hop_count(n) or 0) <= 2),
            "long_hop_rerank_applied_count": sum(1 for _, n in new_rerank_applied_pairs if (inferred_hop_count(n) or 0) >= 3),
            "two_hop_correct_to_empty_count": len(two_hop_correct_to_empty),
            "two_hop_correct_to_wrong_count": len(two_hop_correct_to_wrong),
            "tcc_verified_promotion_triggered_count": len(promotion_pairs),
            "tcc_verified_promotion_admitted_count": len(promotion_admitted_pairs),
            "tcc_verified_promotion_correct_count": sum(exact_match_value(n) for _, n in promotion_admitted_pairs),
            "tcc_verified_promotion_precision": (
                sum(exact_match_value(n) for _, n in promotion_admitted_pairs) / len(promotion_admitted_pairs)
                if promotion_admitted_pairs else None
            ),
            "empty_to_nonempty_by_promotion_count": len(empty_to_nonempty_by_promotion),
            "empty_to_correct_by_promotion_count": sum(exact_match_value(n) for _, n in empty_to_nonempty_by_promotion),
            "promotion_changed_to_wrong_count": len(promotion_changed_to_wrong),
            "promotion_not_admitted_but_answer_changed_count": len(promotion_not_admitted_but_answer_changed),
            "promotion_source_distribution": new_single["mechanisms"].get("promotion_source_distribution", {}),
            "promotion_source_precision": promotion_source_precision,
            "source_buffer_admitted_precision": promotion_source_precision.get("buffer", {}).get("precision"),
            "source_root_memory_admitted_precision": promotion_source_precision.get("root_memory", {}).get("precision"),
            "source_composed_root_memory_admitted_precision": promotion_source_precision.get("composed_root_memory", {}).get("precision"),
            "source_rejected_root_candidate_admitted_precision": promotion_source_precision.get("rejected_root_candidate", {}).get("precision"),
            "source_rejected_score_candidate_root_level_admitted_precision": promotion_source_precision.get("rejected_score_candidate_root_level", {}).get("precision"),
            "root_composed_promotion_triggered_count": len(root_composed_promotion_pairs),
            "root_composed_promotion_admitted_count": len(root_composed_promotion_admitted_pairs),
            "root_composed_promotion_correct_count": len(root_composed_promotion_correct_pairs),
            "root_composed_promotion_precision": (
                len(root_composed_promotion_correct_pairs) / len(root_composed_promotion_admitted_pairs)
                if root_composed_promotion_admitted_pairs else None
            ),
            "buffer_blocked_count": len(buffer_blocked_records),
            "gray_zone_promotion_count": len(gray_zone_promotion_pairs),
            "gray_zone_promotion_correct_count": sum(exact_match_value(n) for _, n in gray_zone_promotion_pairs),
            "promotion_hop_precision": promotion_hop_precision,
            "terminal_memory_consolidation_enabled": bool(tmc_pairs),
            "tmc_triggered_count": len(tmc_triggered_pairs),
            "tmc_terminal_memory_count_total": sum(int(safe_float(n.get("terminal_memory_count"), default=0.0)) for _, n in tmc_pairs),
            "tmc_terminal_memory_count_average": (
                sum(safe_float(n.get("terminal_memory_count"), default=0.0) for _, n in tmc_pairs) / len(tmc_pairs)
                if tmc_pairs else 0.0
            ),
            "tmc_tcc_closed_count": sum(int(safe_float(n.get("tmc_tcc_closed_count"), default=0.0)) for _, n in tmc_pairs),
            "tmc_entered_final_candidate_count": len(tmc_entered_pairs),
            "tmc_candidate_selected_count": len(tmc_selected_pairs),
            "tmc_candidate_selected_correct_count": sum(exact_match_value(n) for _, n in tmc_selected_pairs),
            "tmc_candidate_precision": (
                sum(exact_match_value(n) for _, n in tmc_selected_pairs) / len(tmc_selected_pairs)
                if tmc_selected_pairs else None
            ),
            "tmc_final_candidate_entry_fail_reason_counter": dict(tmc_entry_fail_counter.most_common()),
            "memory_repair_goal_count": sum(len(as_list(n.get("memory_repair_goals"))) for _, n in tmc_pairs),
            "iterative_memory_construction_enabled": bool(imc_pairs),
            "imc_triggered_count": len([1 for _, n in imc_pairs if safe_float(n.get("imc_rounds_executed"), default=0.0) > 0]),
            "imc_rounds_executed_total": sum(int(safe_float(n.get("imc_rounds_executed"), default=0.0)) for _, n in imc_pairs),
            "two_hop_promotion_count": promotion_hop_precision.get("2hop", {}).get("admitted", 0),
            "two_hop_promotion_precision": promotion_hop_precision.get("2hop", {}).get("precision"),
            "three_hop_promotion_count": promotion_hop_precision.get("3hop", {}).get("admitted", 0),
            "three_hop_promotion_precision": promotion_hop_precision.get("3hop", {}).get("precision"),
            "four_hop_promotion_count": promotion_hop_precision.get("4hop", {}).get("admitted", 0),
            "four_hop_promotion_precision": promotion_hop_precision.get("4hop", {}).get("precision"),
            "promotion_fail_reason_distribution": new_single["mechanisms"].get("promotion_fail_reason_distribution", {}),
            "anytime_fallback_triggered_count": len(new_fallback_rows),
            "anytime_fallback_correct_count": sum(exact_match_value(row) for row in new_fallback_rows),
            "anytime_fallback_precision": fallback_precision,
            "final_chain_buffer_candidate_count_average": numeric_average(new_rows, "final_candidate_count"),
            "root_memory_materialized_by_buffer_count": sum(
                1 for row in new_rows
                if safe_bool(row.get("root_memory_exists")) and is_score_based_triggered(row)
            ),
            "root_memory_materialized_by_buffer_correct_count": sum(
                1 for row in new_rows
                if safe_bool(row.get("root_memory_exists")) and is_score_based_triggered(row) and exact_match_value(row)
            ),
        },
        "promotion_cases": {
            "admitted_count": len(promotion_admitted_pairs),
            "admitted": [case_record(b, n) for b, n in promotion_admitted_pairs],
            "correct_count": len(promotion_correct_pairs),
            "correct": [case_record(b, n) for b, n in promotion_correct_pairs],
            "wrong_count": len(promotion_wrong_pairs),
            "wrong": [case_record(b, n) for b, n in promotion_wrong_pairs],
            "promotion_not_admitted_but_answer_changed_count": len(promotion_not_admitted_but_answer_changed),
            "promotion_not_admitted_but_answer_changed_cases": [
                case_record(b, n) for b, n in promotion_not_admitted_but_answer_changed
            ],
        },
        "single_base": base_single,
        "single_new": new_single,
        "_csv_cases": {
            "rescued_cases": rescued_records,
            "harmed_cases": harmed_records,
            "improved_cases": improved_records,
        },
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = dict(payload)
    clean.pop("_csv_cases", None)
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def write_case_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "gold_answer",
        "base_pred",
        "new_pred",
        "base_exact_match",
        "new_exact_match",
        "base_f1",
        "new_f1",
        "base_final_empty_reason",
        "new_final_empty_reason",
        "new_final_chain_score_old",
        "new_final_chain_score_v2",
        "new_score_components_v2",
        "new_last_hop_verification",
        "new_bridge_entity_check",
        "new_terminal_chain_closure_score",
        "new_terminal_chain_closure_info",
        "new_terminal_chain_closure_gate_passed",
        "new_terminal_chain_closure_reject_reasons",
        "new_final_candidate_tcc_audit",
        "new_selected_candidate_tcc",
        "new_tcc_final_audit_changed_answer",
        "new_tcc_rerank_policy",
        "new_tcc_rerank_applied",
        "new_tcc_rerank_skip_reason",
        "new_tcc_rerank_policy_decision",
        "new_tcc_verified_promotion_triggered",
        "new_tcc_promotion_trigger_reason",
        "new_tcc_promotion_candidate_count",
        "new_tcc_promotion_selected_answer",
        "new_tcc_promotion_selected_source",
        "new_tcc_promotion_selected_score",
        "new_expected_answer_type",
        "new_candidate_answer_type",
        "new_precondition_passed",
        "new_precondition_fail_reasons",
        "new_terminal_reject_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))


def print_single_report(title: str, analysis: Dict[str, Any]) -> None:
    overall = analysis["overall"]
    empty = analysis["empty_answer_stats"]
    mech = analysis["mechanisms"]
    print(f"\n== {title} ==")
    print(
        f"rows={overall['total']} nonempty={overall['nonempty']} empty={overall['empty']} "
        f"EM={overall['exact_match_count']}/{overall['total']} ({overall['exact_match_rate']:.3f}) "
        f"F1={overall['average_f1']:.3f} soft_em={overall['average_soft_em']:.3f} "
        f"title_hit={overall['average_title_hit']:.3f}"
    )
    print(
        f"steps={overall['average_steps']:.2f} calls={overall['average_llm_calls']:.2f} "
        f"tokens={overall['average_generated_tokens']:.2f}"
    )
    print("empty reasons:", empty["final_empty_reason_counter"])
    print("terminal rejects:", empty["terminal_reject_reasons_counter"])
    print("score precondition rejects:", empty["score_admission_precondition_fail_reason_counter"])
    print(
        "mechanisms:",
        {
            "buffer": mech["final_chain_buffer_enabled"],
            "score_admission": mech["score_admission_enabled"],
            "score_triggered": mech["score_based_admission_triggered_count"],
            "anytime_fallback": mech["anytime_fallback_triggered_count"],
            "avg_final_chain_score_old": mech["average_final_chain_score_old"],
            "avg_final_chain_score_v2": mech["average_final_chain_score_v2"],
            "avg_last_hop": mech["average_last_hop_support"],
            "bridge_rejected": mech["bridge_entity_rejected_count"],
            "type_mismatch_rejected": mech["answer_type_mismatch_rejected_count"],
            "tcc": mech["terminal_chain_closure_enabled"],
            "tcc_admitted": mech["tcc_admitted_count"],
            "tcc_rejected": mech["tcc_rejected_count"],
            "tcc_precision": mech["tcc_precision"],
            "tcc_audit_candidates": mech["tcc_final_audit_candidate_count"],
            "tcc_audit_downgraded": mech["tcc_final_audit_downgraded_count"],
            "tcc_rerank_applied": mech.get("tcc_rerank_applied_count", 0),
            "tcc_rerank_skipped": mech.get("tcc_rerank_skipped_count", 0),
            "short_hop_protected": mech.get("skipped_short_hop_protected_count", 0),
            "longhop_rerank": mech.get("longhop_rerank_count", 0),
            "weak_candidate_rerank": mech.get("weak_candidate_rerank_count", 0),
            "2hop_rerank": mech.get("two_hop_rerank_applied_count", 0),
            "3plus_rerank": mech.get("long_hop_rerank_applied_count", 0),
            "promotion_triggered": mech.get("tcc_verified_promotion_triggered_count", 0),
            "promotion_admitted": mech.get("tcc_verified_promotion_admitted_count", 0),
            "promotion_correct": mech.get("tcc_verified_promotion_correct_count", 0),
            "promotion_precision": mech.get("tcc_verified_promotion_precision"),
            "tmc_triggered": mech.get("tmc_triggered_count", 0),
            "tmc_terminal_avg": mech.get("tmc_terminal_memory_count_average", 0.0),
            "tmc_closed": mech.get("tmc_tcc_closed_count", 0),
            "tmc_entered_final": mech.get("tmc_entered_final_candidate_count", 0),
            "tmc_selected": mech.get("tmc_candidate_selected_count", 0),
            "tmc_precision": mech.get("tmc_candidate_precision"),
            "repair_goals": mech.get("memory_repair_goal_count", 0),
            "imc_triggered": mech.get("imc_triggered_count", 0),
            "imc_rounds": mech.get("imc_rounds_executed_total", 0),
            "root_composed_promotion_triggered": mech.get("root_composed_promotion_triggered_count", 0),
            "root_composed_promotion_admitted": mech.get("root_composed_promotion_admitted_count", 0),
            "root_composed_promotion_correct": mech.get("root_composed_promotion_correct_count", 0),
            "root_composed_promotion_precision": mech.get("root_composed_promotion_precision"),
            "buffer_blocked": mech.get("buffer_blocked_count", 0),
            "gray_zone_promotion": mech.get("gray_zone_promotion_count", 0),
            "gray_zone_correct": mech.get("gray_zone_promotion_correct_count", 0),
            "tcc_none_nonempty": mech["tcc_final_audit_tcc_none_nonempty_count"],
            "selected_tcc_avg": mech["selected_candidate_tcc_score_average"],
        },
    )


def print_compare_report(payload: Dict[str, Any]) -> None:
    overall = payload["overall_comparison"]
    rescued = payload["rescued_samples"]
    harmed = payload["harmed_samples"]
    improved = payload["improved_samples"]
    mech = payload["mechanism_quality"]
    print("\n== TDCA Diagnostics Comparison ==")
    print(
        f"aligned={overall['aligned_count']} "
        f"nonempty {overall['base_nonempty']} -> {overall['new_nonempty']} "
        f"empty {overall['base_empty']} -> {overall['new_empty']}"
    )
    print(
        f"EM {overall['base_EM']} -> {overall['new_EM']} "
        f"F1 {overall['base_F1']:.3f} -> {overall['new_F1']:.3f}"
    )
    print(
        f"rescued={rescued['base_empty_new_nonempty_count']} "
        f"rescued_correct={rescued['base_empty_new_nonempty_correct_count']} "
        f"rescued_avg_f1={rescued['base_empty_new_nonempty_avg_f1']:.3f}"
    )
    print(
        f"base_nonempty_new_empty={harmed['base_nonempty_new_empty_count']} "
        f"base_correct_new_wrong={harmed['base_correct_new_wrong_count']} "
        f"base_wrong_new_correct={improved['base_wrong_new_correct_count']}"
    )
    print(
        "mechanism precision:",
        {
            "score_based": mech["score_based_admission_precision"],
            "anytime_fallback": mech["anytime_fallback_precision"],
            "score_triggered": mech["score_based_admission_triggered_count"],
            "fallback_triggered": mech["anytime_fallback_triggered_count"],
            "bridge_rejected": payload["single_new"]["mechanisms"]["bridge_entity_rejected_count"],
            "type_mismatch_rejected": payload["single_new"]["mechanisms"]["answer_type_mismatch_rejected_count"],
            "avg_last_hop": payload["single_new"]["mechanisms"]["average_last_hop_support"],
            "tcc_admitted": payload["single_new"]["mechanisms"]["tcc_admitted_count"],
            "tcc_rejected": payload["single_new"]["mechanisms"]["tcc_rejected_count"],
            "tcc_precision": payload["single_new"]["mechanisms"]["tcc_precision"],
            "tcc_audit_candidates": payload["single_new"]["mechanisms"]["tcc_final_audit_candidate_count"],
            "tcc_audit_downgraded": payload["single_new"]["mechanisms"]["tcc_final_audit_downgraded_count"],
            "tcc_audit_changed": payload["single_new"]["mechanisms"]["tcc_final_audit_changed_answer_count"],
            "tcc_rerank_applied": payload["single_new"]["mechanisms"].get("tcc_rerank_applied_count", 0),
            "tcc_rerank_skipped": payload["single_new"]["mechanisms"].get("tcc_rerank_skipped_count", 0),
            "short_hop_protected": payload["single_new"]["mechanisms"].get("skipped_short_hop_protected_count", 0),
            "longhop_rerank": payload["single_new"]["mechanisms"].get("longhop_rerank_count", 0),
            "weak_candidate_rerank": payload["single_new"]["mechanisms"].get("weak_candidate_rerank_count", 0),
            "2hop_rerank": payload["single_new"]["mechanisms"].get("two_hop_rerank_applied_count", 0),
            "3plus_rerank": payload["single_new"]["mechanisms"].get("long_hop_rerank_applied_count", 0),
            "rerank_changed_to_correct": mech.get("tcc_rerank_changed_to_correct_count", 0),
            "rerank_changed_to_wrong": mech.get("tcc_rerank_changed_to_wrong_count", 0),
            "2hop_correct_to_empty": mech.get("two_hop_correct_to_empty_count", 0),
            "2hop_correct_to_wrong": mech.get("two_hop_correct_to_wrong_count", 0),
            "promotion_triggered": mech.get("tcc_verified_promotion_triggered_count", 0),
            "promotion_admitted": mech.get("tcc_verified_promotion_admitted_count", 0),
            "promotion_correct": mech.get("tcc_verified_promotion_correct_count", 0),
            "promotion_precision": mech.get("tcc_verified_promotion_precision"),
            "tmc_triggered": mech.get("tmc_triggered_count", 0),
            "tmc_terminal_avg": mech.get("tmc_terminal_memory_count_average", 0.0),
            "tmc_closed": mech.get("tmc_tcc_closed_count", 0),
            "tmc_entered_final": mech.get("tmc_entered_final_candidate_count", 0),
            "tmc_selected": mech.get("tmc_candidate_selected_count", 0),
            "tmc_precision": mech.get("tmc_candidate_precision"),
            "repair_goals": mech.get("memory_repair_goal_count", 0),
            "imc_triggered": mech.get("imc_triggered_count", 0),
            "imc_rounds": mech.get("imc_rounds_executed_total", 0),
            "root_composed_promotion_triggered": mech.get("root_composed_promotion_triggered_count", 0),
            "root_composed_promotion_admitted": mech.get("root_composed_promotion_admitted_count", 0),
            "root_composed_promotion_correct": mech.get("root_composed_promotion_correct_count", 0),
            "root_composed_promotion_precision": mech.get("root_composed_promotion_precision"),
            "buffer_blocked": mech.get("buffer_blocked_count", 0),
            "gray_zone_promotion": mech.get("gray_zone_promotion_count", 0),
            "gray_zone_correct": mech.get("gray_zone_promotion_correct_count", 0),
            "promotion_empty_to_nonempty": mech.get("empty_to_nonempty_by_promotion_count", 0),
            "promotion_empty_to_correct": mech.get("empty_to_correct_by_promotion_count", 0),
            "promotion_changed_to_wrong": mech.get("promotion_changed_to_wrong_count", 0),
            "tcc_none_nonempty": payload["single_new"]["mechanisms"]["tcc_final_audit_tcc_none_nonempty_count"],
            "selected_tcc_avg": payload["single_new"]["mechanisms"]["selected_candidate_tcc_score_average"],
        },
    )
    print("new score precondition rejects:", payload["single_new"]["empty_answer_stats"]["score_admission_precondition_fail_reason_counter"])
    print("new TCC rejects:", payload["single_new"]["empty_answer_stats"]["terminal_chain_closure_fail_reason_counter"])
    tracked = payload["empty_reason_changes"]["tracked_failures"]
    print("tracked failure deltas:", tracked)
    nonempty_delta = overall["new_nonempty"] - overall["base_nonempty"]
    em_delta = overall["new_EM"] - overall["base_EM"]
    f1_delta = overall["new_F1"] - overall["base_F1"]
    rescued_reasons = Counter(
        str(row.get("base_final_empty_reason", "") or "")
        for row in rescued.get("list", [])
    )
    print("\nassessment:")
    print(f"- nonempty improved: {nonempty_delta > 0} (delta={nonempty_delta})")
    print(f"- EM/F1 improved: {em_delta > 0 or f1_delta > 0} (EM_delta={em_delta}, F1_delta={f1_delta:.3f})")
    print(f"- main rescued failure: {rescued_reasons.most_common(1)[0][0] if rescued_reasons else 'none'}")
    print(f"- obvious harm: {harmed['base_nonempty_new_empty_count'] > 0 or harmed['base_correct_new_wrong_count'] > 0}")
    if mech["score_based_admission_triggered_count"]:
        print(f"- tune final_chain_score_threshold: yes, score-based precision={mech['score_based_admission_precision']}")
    else:
        print("- tune final_chain_score_threshold: not enough score-based admissions observed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TDCA diagnostic summaries and compare two runs.")
    parser.add_argument("--dir", type=str, default="", help="Single TDCA output directory.")
    parser.add_argument("--base_dir", type=str, default="", help="Baseline TDCA output directory.")
    parser.add_argument("--new_dir", type=str, default="", help="New TDCA output directory.")
    parser.add_argument("--output_path", type=str, required=True, help="JSON output path.")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if args.dir:
        rows, info = load_output_dir(Path(args.dir))
        analysis = analyze_single(rows, info)
        print_single_report("TDCA Single Run", analysis)
        write_json(output_path, {"single_analysis": analysis})
        print(f"\nJSON written: {output_path}")
        return

    if not args.base_dir or not args.new_dir:
        raise SystemExit("Provide either --dir or both --base_dir and --new_dir.")

    base_rows, base_info = load_output_dir(Path(args.base_dir))
    new_rows, new_info = load_output_dir(Path(args.new_dir))
    comparison = compare_outputs(base_rows, new_rows, base_info, new_info)
    print_compare_report(comparison)
    write_json(output_path, comparison)
    csv_cases = comparison.get("_csv_cases", {})
    write_case_csv(output_path.parent / "rescued_cases.csv", csv_cases.get("rescued_cases", []))
    write_case_csv(output_path.parent / "harmed_cases.csv", csv_cases.get("harmed_cases", []))
    write_case_csv(output_path.parent / "improved_cases.csv", csv_cases.get("improved_cases", []))
    print(f"\nJSON written: {output_path}")
    print(f"CSV written: {output_path.parent / 'rescued_cases.csv'}")
    print(f"CSV written: {output_path.parent / 'harmed_cases.csv'}")
    print(f"CSV written: {output_path.parent / 'improved_cases.csv'}")


if __name__ == "__main__":
    main()
