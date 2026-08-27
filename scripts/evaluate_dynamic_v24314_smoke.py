#!/usr/bin/env python3
"""Evaluate the frozen v2.4.3.14 independent Smoke-A without provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.verify_artifact import verify
from tdca_research.utils import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _nested_keys(value: Any) -> set[str]:
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


def evaluate(
    run: Path,
    baseline_v1_run: Path,
    preregistration: Path,
    campaign_ledger: Path,
) -> dict[str, Any]:
    prereg = _json(preregistration)
    limits = prereg["hard_checks"]
    expected_count = int(prereg["complete_examples"])
    artifact = verify(run, expected_count=expected_count)
    manifest = _json(run / "run_manifest.json")
    baseline_manifest = _json(baseline_v1_run / "run_manifest.json")
    metrics = _json(run / "metrics.json")
    baseline_metrics = _json(baseline_v1_run / "metrics.json")
    dynamic = _json(run / "dynamic_v2_metrics.json")
    baseline_dynamic = _json(baseline_v1_run / "dynamic_metrics.json")
    rows = _jsonl(run / "dynamic_v2_per_example_metrics.jsonl")
    predictions = _jsonl(run / "predictions.jsonl")
    traces = _jsonl(run / "reasoning_traces.jsonl")
    failures = (run / "failures.jsonl").read_text(encoding="utf-8")
    config = yaml.safe_load(
        (run / "resolved_config.yaml").read_text(encoding="utf-8")
    ) or {}
    campaign = _json(campaign_ledger)

    forbidden = {
        "answers",
        "gold_document_ids",
        "gold_titles",
        "oracle_decomposition",
        "supporting_paragraphs",
        "hop_count",
    }
    leaked_keys = sorted(forbidden.intersection(_nested_keys(traces)))
    join_cases = sum(
        bool(row.get("auditable_three_or_four_hop_join_case")) for row in rows
    )
    downstream_nary_cases = sum(
        bool(row.get("auditable_three_or_four_hop_join_case"))
        and int(row.get("nary_join_downstream_used_count", 0)) > 0
        for row in rows
    )
    allocation_events = [
        row for row in traces if row.get("event") == "allocation_reconciled"
    ]
    allocation_candidates = [
        candidate
        for row in traces
        if row.get("event") == "meta_decision"
        for candidate in row.get("allocation_candidates", [])
    ]
    expected_allocations = sum(int(row.get("allocation_count", 0)) for row in rows)
    predicted_evc_recorded = bool(allocation_candidates) and all(
        candidate.get("predicted_evc") is not None
        and bool(candidate.get("evc_components_raw"))
        and bool(candidate.get("evc_components_normalized"))
        and bool(candidate.get("requested_budget"))
        for candidate in allocation_candidates
    )
    actual_cost_recorded = bool(allocation_events) and all(
        bool(event.get("actual_cost"))
        and event.get("actual_utility") is not None
        for event in allocation_events
    )
    status_to_outcome = {
        "answer": "ANSWER",
        "abstain": "ABSTAIN",
        "budget_exhausted": "BUDGET_EXHAUSTED",
    }
    outcome_by_qid = {
        str(row.get("qid")): str(row.get("termination_outcome")) for row in rows
    }
    termination_consistent = (
        len(predictions) == expected_count
        and len(rows) == expected_count
        and all(
            status_to_outcome.get(str(row.get("status")))
            == outcome_by_qid.get(str(row.get("qid")))
            and ((row.get("status") == "answer") == bool(row.get("answer")))
            for row in predictions
        )
    )
    observed_outcomes = set(outcome_by_qid.values())
    required_outcomes = {"ANSWER", "ABSTAIN", "BUDGET_EXHAUSTED"}
    call_limit = int(prereg["command_overrides"]["campaign_provider_call_cap"])
    token_limit = int(prereg["command_overrides"]["campaign_provider_token_cap"])
    campaign_usage = campaign.get("usage") or {}
    expected_cache_namespace = str(prereg["independent_api_cache_namespace"])
    cache_keys = [
        str(event.get("cache_key", ""))
        for event in campaign.get("events", [])
        if event.get("event") == "request_reserved"
    ]
    cache_root = Path(str(config.get("api_cache_dir", ".cache/tdca_research/llm")))
    expected_cache_root = cache_root / expected_cache_namespace
    expected_cache_files = (
        {path.stem for path in expected_cache_root.rglob("*.json")}
        if expected_cache_root.exists()
        else set()
    )

    candidate_rate = float(dynamic.get("candidate_presence_rate", 0.0))
    baseline_candidate_rate = float(
        baseline_dynamic.get("gold_candidate_generated_rate", 0.0)
    )
    chain_rate = float(metrics.get("full_chain_completion_rate", 0.0))
    baseline_chain_rate = float(
        baseline_metrics.get("full_chain_completion_rate", 0.0)
    )
    evidence = {
        "artifact_verified": bool(artifact["verified"]),
        "sample_count": int(metrics.get("count", 0)),
        "same_frozen_sample_ids": (
            manifest.get("sample_ids") == baseline_manifest.get("sample_ids")
        ),
        "same_dataset_sha256": (
            manifest.get("dataset_sha256") == baseline_manifest.get("dataset_sha256")
        ),
        "expected_independent_cache_namespace": expected_cache_namespace,
        "campaign_cache_key_count": len(cache_keys),
        "expected_namespace_cache_file_count": len(expected_cache_files),
        "campaign_cache_keys_all_in_expected_namespace": bool(cache_keys)
        and set(cache_keys).issubset(expected_cache_files),
        "leaked_trace_keys": leaked_keys,
        "infrastructure_failure_count": int(artifact["infrastructure_failures"]),
        "graph_invariant_violation_count": failures.count("GraphInvariantError"),
        "controller_only_mutation_violation_count": failures.count(
            "outside the V2 controller"
        ),
        "unsupported_answer_count": int(dynamic.get("unsupported_answer_count", -1)),
        "baseline_v1_candidate_presence_rate": baseline_candidate_rate,
        "candidate_presence_rate": candidate_rate,
        "candidate_presence_gain_over_v1": candidate_rate - baseline_candidate_rate,
        "baseline_v1_full_chain_completion_rate": baseline_chain_rate,
        "full_chain_completion_rate": chain_rate,
        "full_chain_completion_gain_over_v1": chain_rate - baseline_chain_rate,
        "auditable_three_or_four_hop_join_case_count": join_cases,
        "downstream_used_nary_join_case_count": downstream_nary_cases,
        "non_uniform_allocation_rate": float(
            dynamic.get("non_uniform_allocation_rate", 0.0)
        ),
        "complete_evc_trace_rate": float(dynamic.get("complete_evc_trace_rate", 0.0)),
        "allocation_event_count": len(allocation_events),
        "expected_allocation_event_count": expected_allocations,
        "predicted_evc_recorded": predicted_evc_recorded,
        "actual_cost_recorded": actual_cost_recorded,
        "observed_termination_outcomes": sorted(observed_outcomes),
        "termination_rows_consistent": termination_consistent,
        "campaign_status": campaign.get("status"),
        "campaign_provider_calls": int(campaign_usage.get("provider_calls", -1)),
        "campaign_provider_reported_tokens": int(
            campaign_usage.get("provider_reported_tokens", -1)
        ),
        "campaign_pending_reserved_tokens": int(
            campaign_usage.get("pending_reserved_tokens", -1)
        ),
    }
    checks = {
        "artifact_complete": evidence["artifact_verified"]
        and evidence["sample_count"] == expected_count,
        "frozen_split_identity": evidence["same_frozen_sample_ids"]
        and evidence["same_dataset_sha256"],
        "independent_cache_identity": (
            evidence["campaign_cache_keys_all_in_expected_namespace"]
        ),
        "zero_leakage": not leaked_keys
        and not bool(config.get("oracle_evidence"))
        and not bool(config.get("oracle_decomposition")),
        "zero_infrastructure_failure": evidence["infrastructure_failure_count"]
        <= int(limits["infrastructure_failure_count_max"]),
        "zero_graph_invariant_violation": evidence["graph_invariant_violation_count"]
        == 0,
        "controller_only_mutation": evidence["controller_only_mutation_violation_count"]
        <= int(limits["controller_invariant_violation_count_max"])
        and all(bool(row.get("controller_state_hash_present")) for row in rows),
        "zero_unsupported_answer": evidence["unsupported_answer_count"]
        <= int(limits["unsupported_answer_count_max"]),
        "candidate_presence_gain": evidence["candidate_presence_gain_over_v1"]
        + 1e-12
        >= float(limits["candidate_presence_gain_over_v1_min"]),
        "full_chain_completion_gain": evidence["full_chain_completion_gain_over_v1"]
        + 1e-12
        >= float(limits["full_chain_completion_gain_over_v1_min"]),
        "auditable_three_or_four_hop_joins": join_cases
        >= int(limits["auditable_three_or_four_hop_join_case_count_min"]),
        "non_uniform_allocation": evidence["non_uniform_allocation_rate"]
        > float(limits["non_uniform_allocation_rate_min_exclusive"]),
        "complete_evc_trace": evidence["complete_evc_trace_rate"]
        >= float(limits["complete_evc_trace_rate"])
        and len(allocation_events) == expected_allocations
        and predicted_evc_recorded
        and actual_cost_recorded,
        "termination_outcomes_separated": termination_consistent
        and (
            not bool(limits["answer_abstain_budget_exhausted_separated"])
            or required_outcomes.issubset(observed_outcomes)
        ),
        "campaign_budget_reconciled": campaign.get("status") == "active"
        and evidence["campaign_pending_reserved_tokens"] == 0
        and 0 <= evidence["campaign_provider_calls"] <= call_limit
        and 0 <= evidence["campaign_provider_reported_tokens"] <= token_limit,
    }
    passed = all(checks.values())
    return {
        "schema_version": "dynamic-hypergraph-v2.4.3.14-independent-smoke-a-gate-v1",
        "inference_calls_made_by_gate": 0,
        "run": str(run),
        "baseline_v1_run": str(baseline_v1_run),
        "preregistration": str(preregistration),
        "campaign_ledger": str(campaign_ledger),
        "evidence": evidence,
        "checks": checks,
        "passed": passed,
        "decision": "GO_SHADOW_B" if passed else "ITERATE",
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-v1-run", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/dynamic_v24314_smoke_preregistration.json"),
    )
    parser.add_argument("--campaign-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.run,
        args.baseline_v1_run,
        args.preregistration,
        args.campaign_ledger,
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "failed_checks": report["failed_checks"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
