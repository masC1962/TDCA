import json
from types import SimpleNamespace

import pytest

from tdca_research.campaign import CampaignBudgetExceeded, CampaignBudgetLedger
from tdca_research.config import ResearchConfig
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.experiments import ArtifactWriter
from tdca_research.llm import DeterministicMockLLM
from tdca_research.llm.openai_client import OpenAICompatibleLLM
from tdca_research.runtime import _resolved_api_cache_dir, run
from tdca_research.utils import sha256_file


def ledger(tmp_path, *, calls=2, tokens=100):
    return CampaignBudgetLedger(
        tmp_path / "campaign.json",
        campaign_id="v22-test",
        provider_call_cap=calls,
        provider_token_cap=tokens,
    )


def test_campaign_ledger_reserves_before_call_and_settles_actual_tokens(tmp_path):
    value = ledger(tmp_path)
    first = value.reserve(
        cache_key="a", cache_path=tmp_path / "a.json", reserved_tokens=30,
    )
    pending = value.snapshot()
    assert pending["usage"] == {
        "provider_calls": 1,
        "provider_reported_tokens": 0,
        "pending_reserved_tokens": 30,
        "effective_provider_tokens": 30,
    }
    value.settle(first, prompt_tokens=8, completion_tokens=4, outcome="success")
    settled = value.snapshot()
    assert settled["usage"]["provider_calls"] == 1
    assert settled["usage"]["provider_reported_tokens"] == 12
    assert settled["usage"]["effective_provider_tokens"] == 12
    assert not settled["pending"]


def test_campaign_ledger_fails_closed_without_exceeding_call_cap(tmp_path):
    value = ledger(tmp_path, calls=1)
    request = value.reserve(
        cache_key="a", cache_path=tmp_path / "a.json", reserved_tokens=10,
    )
    value.settle(request, prompt_tokens=2, completion_tokens=1, outcome="success")
    with pytest.raises(CampaignBudgetExceeded) as caught:
        value.reserve(
            cache_key="b", cache_path=tmp_path / "b.json", reserved_tokens=10,
        )
    snapshot = caught.value.snapshot
    assert snapshot["usage"]["provider_calls"] == 1
    assert snapshot["status"] == "exhausted"
    assert snapshot["last_stop_reason"] == "provider_call_cap"


def test_campaign_ledger_fails_preflight_on_conservative_token_reservation(tmp_path):
    value = ledger(tmp_path, calls=5, tokens=50)
    request = value.reserve(
        cache_key="a", cache_path=tmp_path / "a.json", reserved_tokens=20,
    )
    value.settle(request, prompt_tokens=7, completion_tokens=3, outcome="success")
    with pytest.raises(CampaignBudgetExceeded):
        value.reserve(
            cache_key="b", cache_path=tmp_path / "b.json", reserved_tokens=41,
        )
    snapshot = value.snapshot()
    assert snapshot["usage"]["provider_reported_tokens"] == 10
    assert snapshot["usage"]["provider_calls"] == 1


def test_openai_client_never_sends_http_after_campaign_cap(tmp_path, monkeypatch):
    value = ledger(tmp_path, calls=1, tokens=100)
    request = value.reserve(
        cache_key="prior", cache_path=tmp_path / "prior.json", reserved_tokens=10,
    )
    value.settle(request, prompt_tokens=2, completion_tokens=1, outcome="success")
    monkeypatch.setenv("LLM_API_KEY", "test-only")
    client = OpenAICompatibleLLM(
        base_url="https://example.invalid/v1", model_name="test",
        cache_dir=str(tmp_path / "cache"), prompt_version="test",
        campaign_ledger=value,
    )
    calls = []

    def forbidden_http(**kwargs):
        calls.append(kwargs)
        raise AssertionError("HTTP request must not run after campaign cap")

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden_http)),
    )
    with pytest.raises(CampaignBudgetExceeded):
        client.generate_text([{"role": "user", "content": "hello"}], max_tokens=8)
    assert calls == []


def test_campaign_ledger_reconciles_killed_request_from_durable_cache(tmp_path):
    path = tmp_path / "campaign.json"
    value = ledger(tmp_path)
    cache = tmp_path / "response.json"
    value.reserve(cache_key="a", cache_path=cache, reserved_tokens=30)
    cache.write_text(json.dumps({
        "text": "answer", "prompt_tokens": 9, "completion_tokens": 2,
        "finish_reason": "stop", "metadata": {},
    }), encoding="utf-8")
    reopened = CampaignBudgetLedger(
        path, campaign_id="v22-test", provider_call_cap=2, provider_token_cap=100,
    )
    snapshot = reopened.snapshot()
    assert snapshot["usage"]["provider_calls"] == 1
    assert snapshot["usage"]["provider_reported_tokens"] == 11
    assert snapshot["usage"]["pending_reserved_tokens"] == 0
    assert any(row["event"] == "request_reconciled_from_cache" for row in snapshot["events"])


def test_campaign_config_is_all_or_nothing():
    with pytest.raises(ValueError, match="configured together"):
        ResearchConfig(campaign_id="incomplete")
    complete = ResearchConfig(
        campaign_id="v22", campaign_ledger_path="ledger.json",
        campaign_provider_call_cap=2500, campaign_provider_token_cap=2_500_000,
    )
    assert complete.campaign_provider_call_cap == 2500


def test_cache_isolation_includes_campaign_method_allocator_and_prompt():
    config = DynamicV2ResearchConfig(
        campaign_id="v22", campaign_ledger_path="ledger.json",
        campaign_provider_call_cap=2500, campaign_provider_token_cap=2_500_000,
        isolate_api_cache_by_experiment_arm=True,
        api_cache_dir="cache", allocator_mode="uniform", prompt_version="prompt/v22",
    )
    assert _resolved_api_cache_dir(config).replace("\\", "/") == (
        "cache/v22/dynamic_hypergraph_tdca_v2/uniform/prompt_v22"
    )


def test_artifact_writer_marks_budget_interruption_without_fake_prediction(tmp_path):
    writer = ArtifactWriter(tmp_path)
    writer.mark_interrupted(
        reason="campaign_budget_exhausted", completed=3, total=50,
        next_qid="q4", details={"campaign": {"status": "exhausted"}},
    )
    progress = json.loads((tmp_path / "partial_progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "interrupted"
    assert progress["completed"] == 3
    assert progress["next_qid"] == "q4"


def test_runtime_resumes_exact_durable_question_prefix(tmp_path):
    dataset = tmp_path / "data.jsonl"
    rows = [
        {"id": "q1", "question": "First?", "answer": "A1", "paragraphs": [
            {"id": "p1", "title": "First", "paragraph_text": "A1.", "is_supporting": True},
            {"id": "n1", "title": "Noise", "paragraph_text": "Irrelevant.", "is_supporting": False},
        ]},
        {"id": "q2", "question": "Second?", "answer": "A2", "paragraphs": [
            {"id": "p2", "title": "Second", "paragraph_text": "A2.", "is_supporting": True},
            {"id": "n2", "title": "Noise", "paragraph_text": "Irrelevant.", "is_supporting": False},
        ]},
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "seed": 20260820, "dataset_sha256": sha256_file(dataset),
        "splits": {"smoke": ["q1", "q2"]},
    }), encoding="utf-8")
    config = ResearchConfig(
        method="closed_book", dataset_path=str(dataset), split="smoke",
        split_seed=20260820, split_manifest_path=str(manifest),
        output_root=str(tmp_path / "outputs"), max_total_tokens=2000,
    )

    class StopOnSecond(DeterministicMockLLM):
        def __init__(self):
            super().__init__(text_responses=["A1"])
            self.calls = 0

        def generate_text(self, messages, max_tokens, temperature=0.0):
            self.calls += 1
            if self.calls == 2:
                raise CampaignBudgetExceeded("test stop", {"status": "exhausted"})
            return super().generate_text(messages, max_tokens, temperature)

    with pytest.raises(CampaignBudgetExceeded):
        run(config, mock=StopOnSecond())
    run_dir = next((tmp_path / "outputs").iterdir())
    progress = json.loads((run_dir / "partial_progress.json").read_text(encoding="utf-8"))
    assert progress["completed"] == 1
    assert progress["next_qid"] == "q2"
    assert len((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    completed = run(
        config, mock=DeterministicMockLLM(text_responses=["A2"]), resume_dir=run_dir,
    )
    assert completed == run_dir
    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["qid"] for row in predictions] == ["q1", "q2"]
    assert json.loads((run_dir / "partial_progress.json").read_text(encoding="utf-8"))["status"] == "complete"
