import json

import pytest

from tdca_research.config import ResearchConfig
from tdca_research.llm.openai_client import OpenAICompatibleLLM


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
    with pytest.raises(json.JSONDecodeError):
        OpenAICompatibleLLM._parse_json(
            '{"answer":[{"value":"A"}', allow_complete_array_prefix=True,
        )


def test_api_retry_policy_is_explicit_and_validated():
    config = ResearchConfig(request_timeout_seconds=30, max_api_attempts=2)
    assert config.request_timeout_seconds == 30
    assert config.max_api_attempts == 2
