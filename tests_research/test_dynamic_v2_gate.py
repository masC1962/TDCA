import json
from pathlib import Path

import yaml

from scripts.evaluate_dynamic_v2_gate import _pareto


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fake_run(
    root: Path, name: str, mode: str, *, f1: float, chain: float,
    calls: int, tokens: int, retrievals: int, token_cap: int = 16000,
) -> Path:
    run = root / name
    run.mkdir()
    write_json(run / "metrics.json", {"count": 1, "f1": f1, "full_chain_completion_rate": chain})
    write_json(run / "dynamic_v2_metrics.json", {"complete_outcome_feedback_trace_rate": 1.0})
    write_json(run / "cost_summary.json", {
        "llm_calls": calls, "retrieval_calls": retrievals,
        "prompt_tokens": tokens, "completion_tokens": 0,
        "provider_calls": calls, "provider_prompt_tokens": tokens, "provider_completion_tokens": 0,
    })
    write_json(run / "run_manifest.json", {
        "dataset_sha256": "dataset", "model": "qwen-plus", "prompt_version": "v2", "split_seed": 20260820,
    })
    write_json(run / "predictions.jsonl", {"qid": "q1"})
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({
        "allocator_mode": mode, "max_llm_calls": 16,
        "max_total_tokens": token_cap, "max_retrieval_calls": 8,
        "shared_setting": "identical",
    }), encoding="utf-8")
    return run


def gate_policy():
    return {"computation_allocation": {
        "pareto_noninferior_quality_tolerance": 0.01,
        "pareto_strict_quality_gain": 0.01,
        "pareto_strict_cost_reduction": 0.05,
        "pareto_matched_cost_tolerance": 0.05,
    }}


def test_pareto_requires_adaptive_to_dominate_both_matched_controls(tmp_path: Path):
    adaptive = fake_run(tmp_path, "adaptive", "adaptive_evc", f1=0.70, chain=0.70,
                        calls=8, tokens=8000, retrievals=4)
    uniform = fake_run(tmp_path, "uniform", "uniform", f1=0.65, chain=0.65,
                       calls=8, tokens=8000, retrievals=4)
    fixed = fake_run(tmp_path, "fixed", "fixed_order", f1=0.70, chain=0.70,
                     calls=10, tokens=10000, retrievals=5)
    report = _pareto(adaptive, [uniform, fixed], gate_policy())
    assert report["passed"]
    assert report["mode_coverage"]
    assert all(row["matched_compute_identity"] for row in report["comparisons"])


def test_pareto_rejects_budget_or_config_mismatch(tmp_path: Path):
    adaptive = fake_run(tmp_path, "adaptive", "adaptive_evc", f1=0.70, chain=0.70,
                        calls=8, tokens=8000, retrievals=4)
    uniform = fake_run(tmp_path, "uniform", "uniform", f1=0.60, chain=0.60,
                       calls=8, tokens=8000, retrievals=4, token_cap=12000)
    fixed = fake_run(tmp_path, "fixed", "fixed_order", f1=0.60, chain=0.60,
                     calls=8, tokens=8000, retrievals=4)
    report = _pareto(adaptive, [uniform, fixed], gate_policy())
    assert not report["passed"]
    assert not report["comparisons"][0]["matched_compute_identity"]
