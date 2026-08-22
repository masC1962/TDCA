import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "build_dynamic_v2_campaign_ledger.py"
    spec = importlib.util.spec_from_file_location("dynamic_v2_campaign_ledger", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_usage_adjustment_is_explicit_and_auditable(tmp_path: Path):
    artifact = tmp_path / "adjustment.json"
    artifact.write_text(json.dumps({
        "schema_version": "dynamic-v2-usage-adjustment-v1",
        "accounting_complete": True,
        "reason": "recover provider usage from a safely interrupted cached run",
        "source_paths": ["partial/predictions.jsonl"],
        "usage": {"provider_calls": 3, "provider_reported_tokens": 1200},
    }), encoding="utf-8")
    row = _module().usage(artifact)
    assert row["type"] == "usage_adjustment"
    assert row["complete"]
    assert row["provider_calls"] == 3
    assert row["provider_reported_tokens"] == 1200
    assert row["source_paths"] == ["partial/predictions.jsonl"]


def test_usage_adjustment_fails_closed_without_accounting_attestation(tmp_path: Path):
    artifact = tmp_path / "adjustment.json"
    artifact.write_text(json.dumps({
        "schema_version": "dynamic-v2-usage-adjustment-v1",
        "usage": {"provider_calls": 3, "provider_reported_tokens": 1200},
    }), encoding="utf-8")
    assert not _module().usage(artifact)["complete"]
