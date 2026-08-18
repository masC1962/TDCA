import json
from pathlib import Path

import pytest

from tdca_research.utils import sha256_file


def _load_script():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "verify_analysis_manifest.py"
    spec = importlib.util.spec_from_file_location("verify_analysis_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_hash_contract(tmp_path, monkeypatch, capsys):
    module = _load_script()
    artifact = tmp_path / "result.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "artifacts": {str(artifact): sha256_file(artifact)}
    }), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["verify", str(manifest), "--expected-count", "1"])
    module.main()
    assert '"verified": true' in capsys.readouterr().out


def test_manifest_rejects_mutation(tmp_path, monkeypatch):
    module = _load_script()
    artifact = tmp_path / "result.json"
    artifact.write_text("before", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "artifacts": {str(artifact): sha256_file(artifact)}
    }), encoding="utf-8")
    artifact.write_text("after", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["verify", str(manifest)])
    with pytest.raises(ValueError, match="checksum_mismatches"):
        module.main()
