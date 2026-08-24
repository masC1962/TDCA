import json

import pytest

from tdca_research.config import ResearchConfig
from tdca_research.llm.openai_client import OpenAICompatibleLLM, _declared_array_root_key


def test_json_parser_accepts_fenced_json():
    assert OpenAICompatibleLLM._parse_json('```json\n{"ok": true}\n```') == {"ok": True}


def test_json_parser_recovers_only_complete_rows_from_length_truncated_batch():
    raw = '{"claims":[{"subject":"A","value":"B"},{"subject":"unfinished"'
    assert OpenAICompatibleLLM._parse_json(
        raw, allow_complete_array_prefix=True,
    ) == {"claims": [{"subject": "A", "value": "B"}]}
    with pytest.raises(json.JSONDecodeError):
        OpenAICompatibleLLM._parse_json(raw)


def test_json_parser_does_not_recover_partial_or_unapproved_schemas():
    with pytest.raises(json.JSONDecodeError):
        OpenAICompatibleLLM._parse_json(
            '{"scores":[{"candidate_id":"unfinished"',
            allow_complete_array_prefix=True,
        )


def test_bare_array_recovery_is_declared_per_structured_schema():
    assert _declared_array_root_key("dynamic_v2_event_graph_editor_v1") == "operations"
    assert _declared_array_root_key("dynamic_v2_typed_claim_extraction_v1") == "claims"
    assert _declared_array_root_key("dynamic_v2_independent_verification_v1_pass_2") == "scores"
    assert _declared_array_root_key("dynamic_terminal_derivation_v1") == ""
    with pytest.raises(json.JSONDecodeError):
        OpenAICompatibleLLM._parse_json(
            '{"answer":[{"value":"A"}', allow_complete_array_prefix=True,
        )


def test_api_retry_policy_is_explicit_and_validated():
    config = ResearchConfig(request_timeout_seconds=30, max_api_attempts=2)
    assert config.request_timeout_seconds == 30
    assert config.max_api_attempts == 2
