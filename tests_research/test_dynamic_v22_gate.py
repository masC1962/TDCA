import json
from pathlib import Path

import yaml

from scripts.build_dynamic_v22_budget_curve import build
from scripts.evaluate_dynamic_v22_gate import _campaign_audit, _strict_pareto
from tdca_research.run import _load_config


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fake_run(
    root: Path,
    name: str,
    mode: str,
    *,
    f1: float,
    calls: int,
    tokens: int,
    retrievals: int,
    call_cap: int = 16,
    token_cap: int = 16000,
    retrieval_cap: int = 8,
) -> Path:
    run = root / name
    run.mkdir()
    write_json(run / "metrics.json", {
        "count": 1, "f1": f1, "exact_match": f1,
        "full_chain_completion_rate": 0.70,
    })
    write_json(run / "dynamic_v2_metrics.json", {
        "candidate_presence_rate": 0.70,
    })
    write_json(run / "cost_summary.json", {
        "llm_calls": calls, "retrieval_calls": retrievals,
        "prompt_tokens": tokens, "completion_tokens": 0,
        "provider_calls": calls,
        "provider_prompt_tokens": tokens, "provider_completion_tokens": 0,
        "cache_hits": 0,
    })
    write_json(run / "run_manifest.json", {
        "dataset_sha256": "dataset", "model": "qwen-plus",
        "prompt_version": "dynamic-hypergraph-v2.2", "split_seed": 20260820,
    })
    (run / "predictions.jsonl").write_text(
        json.dumps({"qid": "q1", "status": "answer", "answer": "A"}) + "\n",
        encoding="utf-8",
    )
    write_json(run / "partial_progress.json", {"status": "complete"})
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({
        "allocator_mode": mode,
        "max_llm_calls": call_cap,
        "max_total_tokens": token_cap,
        "max_retrieval_calls": retrieval_cap,
        "campaign_id": "v22",
        "shared_setting": "identical",
    }), encoding="utf-8")
    return run


def test_v22_pareto_requires_strict_dominance_against_both_controls(tmp_path):
    adaptive = fake_run(
        tmp_path, "adaptive", "adaptive_evc", f1=0.70,
        calls=8, tokens=8000, retrievals=4,
    )
    uniform = fake_run(
        tmp_path, "uniform", "uniform", f1=0.65,
        calls=8, tokens=8000, retrievals=4,
    )
    fixed = fake_run(
        tmp_path, "fixed", "fixed_order", f1=0.70,
        calls=9, tokens=9000, retrievals=5,
    )
    assert _strict_pareto(adaptive, [uniform, fixed])["passed"]
    write_json(fixed / "metrics.json", {
        "count": 1, "f1": 0.71, "exact_match": 0.71,
        "full_chain_completion_rate": 0.70,
    })
    assert not _strict_pareto(adaptive, [uniform, fixed])["passed"]


def test_v22_campaign_audit_reconciles_every_reserved_http_attempt():
    gate = {"computation_allocation": {
        "provider_call_campaign_cap": 2500,
        "provider_reported_token_campaign_cap": 2_500_000,
    }}
    ledger = {
        "schema_version": "tdca-campaign-budget-v1",
        "status": "active",
        "limits": {"provider_calls": 2500, "provider_reported_tokens": 2_500_000},
        "usage": {"provider_calls": 1, "provider_reported_tokens": 12},
        "pending": {},
        "events": [
            {"event": "request_reserved", "request_id": "r1"},
            {"event": "request_settled", "request_id": "r1", "provider_reported_tokens": 12},
        ],
    }
    assert _campaign_audit(ledger, gate)["passed"]
    ledger["events"] = ledger["events"][:1]
    assert not _campaign_audit(ledger, gate)["passed"]


def test_budget_curve_requires_two_complete_three_mode_matched_points(tmp_path):
    runs = []
    for call_cap, token_cap, retrieval_cap in ((8, 8000, 4), (16, 16000, 8)):
        for mode in ("adaptive_evc", "uniform", "fixed_order"):
            runs.append(fake_run(
                tmp_path, f"{mode}_{call_cap}", mode,
                f1=0.70, calls=4, tokens=4000, retrievals=2,
                call_cap=call_cap, token_cap=token_cap, retrieval_cap=retrieval_cap,
            ))
    report = build(runs)
    assert report["complete"]
    assert report["point_count"] == 2
    assert not build(runs[:-1])["complete"]


def test_v22_configs_share_development_campaign_and_separate_heldout_accounting():
    root = Path(__file__).parents[1]
    smoke = _load_config(str(root / "configs/dynamic_hypergraph_v22_qwen_smoke20.yaml"), None)
    development = _load_config(
        str(root / "configs/dynamic_hypergraph_v22_qwen_development50.yaml"), None,
    )
    heldout = _load_config(
        str(root / "configs/dynamic_hypergraph_v22_qwen_heldout200.yaml"), None,
    )
    assert smoke.campaign_id == development.campaign_id
    assert smoke.campaign_ledger_path == development.campaign_ledger_path
    assert heldout.campaign_id != development.campaign_id
    assert heldout.campaign_ledger_path != development.campaign_ledger_path
    assert {smoke.prompt_version, development.prompt_version, heldout.prompt_version} == {
        "dynamic-hypergraph-v2.2.3-immutable-root",
    }
    assert development.campaign_provider_call_cap == 2500
    assert development.campaign_provider_token_cap == 2_500_000
