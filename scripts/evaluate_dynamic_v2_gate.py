#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _nested_keys(value: Any) -> set[str]:
    """Return every mapping key in a nested trace payload."""
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for child in value.values():
            keys.update(_nested_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_nested_keys(child))
        return keys
    return set()


def _test_gate() -> dict[str, Any]:
    command = [
        "python", "-m", "pytest", "-q",
        "tests_research/test_dynamic_v2_core.py",
        "tests_research/test_dynamic_v2_integration.py",
        "tests_research/test_dynamic_v2_revision_eval.py",
        "tests_research/test_data.py",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "passed": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-5:],
        "stderr_tail": completed.stderr.splitlines()[-5:],
    }


def _mean_normalized_cost(
    run: Path, reference_caps: tuple[int, int, int] = (16, 16_000, 8),
) -> float:
    cost = _json(run / "cost_summary.json")
    metrics = _json(run / "metrics.json")
    count = max(1, int(metrics.get("count", 0)))
    call_cap, token_cap, retrieval_cap = reference_caps
    return (
        float(cost.get("llm_calls", 0)) / count / call_cap
        + float(cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0)) / count / token_cap
        + float(cost.get("retrieval_calls", 0)) / count / retrieval_cap
    ) / 3.0


def _run_identity(run: Path) -> dict[str, Any]:
    manifest = _json(run / "run_manifest.json")
    config = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    predictions = _jsonl(run / "predictions.jsonl")
    return {
        "sample_ids": [str(row.get("qid")) for row in predictions],
        "dataset_sha256": manifest.get("dataset_sha256"),
        "model": manifest.get("model"),
        "prompt_version": manifest.get("prompt_version"),
        "split_seed": manifest.get("split_seed"),
        "allocator_mode": config.get("allocator_mode"),
        "budget_caps": {
            key: config.get(key) for key in (
                "max_llm_calls", "max_total_tokens", "max_retrieval_calls",
            )
        },
        "config_except_allocator": {
            key: value for key, value in config.items() if key != "allocator_mode"
        },
    }


def _pareto(v2_run: Path, controls: list[Path], gate: dict[str, Any]) -> dict[str, Any]:
    v2_metrics = _json(v2_run / "metrics.json")
    v2_dynamic = _json(v2_run / "dynamic_v2_metrics.json")
    v2_f1 = float(v2_metrics.get("f1", 0.0))
    v2_chain = float(v2_metrics.get("full_chain_completion_rate", 0.0))
    v2_cost = _mean_normalized_cost(v2_run)
    v2_provider = _json(v2_run / "cost_summary.json")
    v2_identity = _run_identity(v2_run)
    policy = gate["computation_allocation"]
    comparisons = []
    modes_seen: set[str] = set()
    for control in controls:
        metrics = _json(control / "metrics.json")
        dynamic = _json(control / "dynamic_v2_metrics.json")
        control_f1 = float(metrics.get("f1", 0.0))
        control_chain = float(metrics.get("full_chain_completion_rate", 0.0))
        control_cost = _mean_normalized_cost(control)
        control_provider = _json(control / "cost_summary.json")
        identity = _run_identity(control)
        mode = str(identity.get("allocator_mode", ""))
        modes_seen.add(mode)
        comparable = (
            bool(v2_identity["sample_ids"])
            and v2_identity["sample_ids"] == identity["sample_ids"]
            and v2_identity["dataset_sha256"] == identity["dataset_sha256"]
            and v2_identity["model"] == identity["model"]
            and v2_identity["prompt_version"] == identity["prompt_version"]
            and v2_identity["split_seed"] == identity["split_seed"]
            and v2_identity["budget_caps"] == identity["budget_caps"]
            and v2_identity["config_except_allocator"] == identity["config_except_allocator"]
        )
        cost_reduction = (control_cost - v2_cost) / max(control_cost, 1e-12)
        tolerance = float(policy["pareto_noninferior_quality_tolerance"])
        cost_non_higher = v2_cost <= control_cost * (
            1.0 + float(policy["pareto_matched_cost_tolerance"])
        )
        quality_noninferior = (
            v2_f1 >= control_f1 - tolerance
            and v2_chain >= control_chain - tolerance
        )
        strict_improvement = (
            v2_f1 - control_f1 >= float(policy["pareto_strict_quality_gain"])
            or v2_chain - control_chain >= float(policy["pareto_strict_quality_gain"])
            or cost_reduction >= float(policy["pareto_strict_cost_reduction"])
        )
        point_passed = (
            comparable and mode in {"uniform", "fixed_order"}
            and cost_non_higher and quality_noninferior and strict_improvement
        )
        comparisons.append({
            "control_run": str(control),
            "control_allocator_mode": mode,
            "v2_f1": v2_f1, "control_f1": control_f1,
            "v2_full_chain": v2_chain, "control_full_chain": control_chain,
            "v2_normalized_cost": v2_cost, "control_normalized_cost": control_cost,
            "f1_difference": v2_f1 - control_f1,
            "full_chain_difference": v2_chain - control_chain,
            "relative_cost_reduction": cost_reduction,
            "v2_provider_calls": int(v2_provider.get("provider_calls", 0)),
            "control_provider_calls": int(control_provider.get("provider_calls", 0)),
            "v2_outcome_feedback_trace_rate": float(v2_dynamic.get("complete_outcome_feedback_trace_rate", 0.0)),
            "control_outcome_feedback_trace_rate": float(dynamic.get("complete_outcome_feedback_trace_rate", 0.0)),
            "matched_compute_identity": comparable,
            "cost_non_higher": cost_non_higher,
            "quality_noninferior": quality_noninferior,
            "strict_improvement": strict_improvement,
            "passed": point_passed,
        })
    required_controls = {"uniform", "fixed_order"}
    mode_coverage = required_controls.issubset(modes_seen)
    return {
        "passed": (
            v2_identity.get("allocator_mode") == "adaptive_evc"
            and mode_coverage and len(comparisons) == 2
            and all(row["passed"] for row in comparisons)
        ),
        "adaptive_allocator_mode": v2_identity.get("allocator_mode"),
        "required_control_modes": sorted(required_controls),
        "observed_control_modes": sorted(modes_seen),
        "mode_coverage": mode_coverage,
        "comparisons": comparisons,
    }


def evaluate(
    v1_run: Path,
    v2_run: Path,
    controls: list[Path],
    revision_eval: Path,
    campaign_ledger: Path,
) -> dict[str, Any]:
    gate = _json(Path("configs/dynamic_v2_hard_gate.json"))
    tests = _test_gate()
    v1_metrics = _json(v1_run / "metrics.json")
    v1_dynamic = _json(v1_run / "dynamic_metrics.json")
    v2_metrics = _json(v2_run / "metrics.json")
    v2_dynamic = _json(v2_run / "dynamic_v2_metrics.json")
    v2_rows = _jsonl(v2_run / "dynamic_v2_per_example_metrics.jsonl")
    config = yaml.safe_load((v2_run / "resolved_config.yaml").read_text(encoding="utf-8"))
    reasoning = _jsonl(v2_run / "reasoning_traces.jsonl")
    predictions = _jsonl(v2_run / "predictions.jsonl")
    revision_report = _json(revision_eval)
    revision_metrics = revision_report.get("metrics", {})
    campaign = _json(campaign_ledger)
    forbidden = {
        "answers", "gold_document_ids", "gold_titles", "oracle_decomposition",
        "supporting_paragraphs", "hop_count",
    }
    leaked_keys = sorted(forbidden.intersection(_nested_keys(reasoning)))
    allocation_events = [
        row for row in reasoning if row.get("event") == "allocation_reconciled"
    ]
    allocation_candidates = [
        candidate
        for row in reasoning if row.get("event") == "meta_decision"
        for candidate in row.get("allocation_candidates", [])
    ]
    expected_allocation_count = sum(int(row.get("allocation_count", 0)) for row in v2_rows)
    allocations_fully_reconciled = (
        bool(allocation_events) and len(allocation_events) == expected_allocation_count
    )
    predicted_evc_recorded = bool(allocation_candidates) and all(
        row.get("predicted_evc") is not None
        and bool(row.get("evc_components_raw"))
        and bool(row.get("evc_components_normalized"))
        and bool(row.get("requested_budget"))
        and bool(row.get("pre_state_summary"))
        and row.get("feedback_prior") is not None
        for row in allocation_candidates
    )
    actual_cost_recorded = bool(allocation_events) and all(
        bool(row.get("actual_cost"))
        and bool(row.get("post_state_summary"))
        and bool(row.get("state_delta"))
        and bool(row.get("actual_utility_components_raw"))
        and bool(row.get("actual_utility_components_normalized"))
        and row.get("actual_utility") is not None
        for row in allocation_events
    )
    join_cases = sum(bool(row.get("auditable_three_or_four_hop_join_case")) for row in v2_rows)
    downstream_nary_cases = sum(
        bool(row.get("auditable_three_or_four_hop_join_case"))
        and int(row.get("nary_join_downstream_used_count", 0)) > 0
        for row in v2_rows
    )
    terminal_by_qid = {
        str(row.get("qid")): str(row.get("termination_outcome")) for row in v2_rows
    }
    status_to_terminal = {
        "answer": "ANSWER", "abstain": "ABSTAIN",
        "budget_exhausted": "BUDGET_EXHAUSTED",
    }
    termination_rows_consistent = bool(predictions) and len(predictions) == len(v2_rows) and all(
        status_to_terminal.get(str(row.get("status"))) == terminal_by_qid.get(str(row.get("qid")))
        and ((str(row.get("status")) == "answer") == bool(row.get("answer")))
        for row in predictions
    )
    observed_outcomes = set(terminal_by_qid.values())
    required_outcomes = set(gate["termination"]["required_outcomes"])
    ledger_paths = {
        str(Path(row.get("path", "")).resolve())
        for row in campaign.get("artifacts", []) if row.get("path")
    }
    required_ledger_paths = {
        str(path.resolve()) for path in [v1_run, v2_run, *controls, revision_eval]
    }
    checks = {
        "infrastructure.zero_leakage": (
            not leaked_keys and not config.get("oracle_evidence") and not config.get("oracle_decomposition")
        ),
        "infrastructure.zero_invariant_violation": (
            tests["passed"] and float(v2_metrics.get("infrastructure_failure_rate", 1.0)) == 0.0
        ),
        "infrastructure.controller_only_mutation": (
            tests["passed"] and float(v2_dynamic.get("controller_state_hash_present_rate", 0.0)) == 1.0
        ),
        "reasoning.candidate_presence_gain": (
            float(v2_dynamic.get("candidate_presence_rate", 0.0))
            >= float(v1_dynamic.get("gold_candidate_generated_rate", 0.0))
            + float(gate["reasoning_capability"]["candidate_presence_absolute_gain_over_v1"])
        ),
        "reasoning.full_chain_gain": (
            float(v2_metrics.get("full_chain_completion_rate", 0.0))
            >= float(v1_metrics.get("full_chain_completion_rate", 0.0))
            + float(gate["reasoning_capability"]["full_chain_completion_absolute_gain_over_v1"])
        ),
        "reasoning.auditable_join_cases": (
            join_cases >= int(gate["reasoning_capability"]["minimum_auditable_three_or_four_hop_join_cases"])
        ),
        "reasoning.downstream_used_nary_join_cases": (
            downstream_nary_cases
            >= int(gate["reasoning_capability"]["minimum_downstream_used_nary_join_cases"])
        ),
        "dynamic.adversarial_revision": tests["passed"],
        "dynamic.public_revision_suite_balance": (
            int(revision_metrics.get("tp", 0)) + int(revision_metrics.get("fn", 0))
            >= int(gate["dynamic_behavior"]["minimum_should_revise_cases"])
            and int(revision_metrics.get("tn", 0)) + int(revision_metrics.get("fp", 0))
            >= int(gate["dynamic_behavior"]["minimum_should_not_revise_cases"])
        ),
        "dynamic.natural_revision_precision_recall_fpr": (
            float(revision_metrics.get("precision", 0.0))
            >= float(gate["dynamic_behavior"]["minimum_natural_revision_precision"])
            and float(revision_metrics.get("recall", 0.0))
            >= float(gate["dynamic_behavior"]["minimum_natural_revision_recall"])
            and float(revision_metrics.get("false_positive_rate", 1.0))
            <= float(gate["dynamic_behavior"]["maximum_natural_revision_false_positive_rate"])
            and bool(revision_metrics.get("complete_predictions"))
            and bool(revision_metrics.get("zero_invariant_violations"))
        ),
        "dynamic.non_uniform_allocation": any(bool(row.get("non_uniform_allocation")) for row in v2_rows),
        "allocation.complete_evc_trace": (
            bool(v2_rows)
            and allocations_fully_reconciled
            and all(bool(row.get("complete_evc_trace")) for row in v2_rows)
        ),
        "allocation.complete_outcome_feedback_trace": (
            bool(v2_rows)
            and all(bool(row.get("complete_outcome_feedback_trace")) for row in v2_rows)
        ),
        "allocation.feedback_influences_later_allocation": any(
            bool(row.get("feedback_influenced_allocation")) for row in v2_rows
        ),
        "allocation.predicted_evc_recorded": predicted_evc_recorded,
        "allocation.actual_cost_recorded": actual_cost_recorded,
        "allocation.campaign_api_budget": (
            int(campaign.get("provider_calls", 10**18))
            <= int(gate["computation_allocation"]["provider_call_campaign_cap"])
            and int(campaign.get("provider_reported_tokens", 10**18))
            <= int(gate["computation_allocation"]["provider_reported_token_campaign_cap"])
            and bool(campaign.get("complete", False))
            and required_ledger_paths.issubset(ledger_paths)
        ),
        "termination.no_unsupported_answer": int(v2_dynamic.get("unsupported_answer_count", -1)) == 0,
        "termination.separated_outcomes": (
            tests["passed"] and termination_rows_consistent
            and required_outcomes.issubset(observed_outcomes)
        ),
    }
    pareto = _pareto(v2_run, controls, gate)
    checks["allocation.pareto_improvement"] = pareto["passed"]
    return {
        "schema_version": "dynamic-hypergraph-v2-gate-evaluation-v2",
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "v1_run": str(v1_run), "v2_run": str(v2_run),
            "control_runs": [str(value) for value in controls],
            "test_gate": tests, "leaked_keys": leaked_keys,
            "auditable_three_or_four_hop_join_cases": join_cases,
            "downstream_used_nary_join_cases": downstream_nary_cases,
            "revision_evaluation": str(revision_eval),
            "revision_metrics": revision_metrics,
            "campaign_ledger": campaign,
            "allocation_event_count": len(allocation_events),
            "allocation_candidate_count": len(allocation_candidates),
            "expected_allocation_count": expected_allocation_count,
            "allocations_fully_reconciled": allocations_fully_reconciled,
            "predicted_evc_recorded": predicted_evc_recorded,
            "actual_cost_recorded": actual_cost_recorded,
            "termination_rows_consistent": termination_rows_consistent,
            "observed_termination_outcomes": sorted(observed_outcomes),
            "pareto": pareto,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Dynamic Hypergraph v2 hard gate")
    parser.add_argument("--v1-run", type=Path, required=True)
    parser.add_argument("--v2-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, action="append", required=True)
    parser.add_argument("--revision-eval", type=Path, required=True)
    parser.add_argument("--campaign-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.v1_run, args.v2_run, args.control_run,
        args.revision_eval, args.campaign_ledger,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
