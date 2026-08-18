import json

from tdca_research.config import ResearchConfig
from tdca_research.llm.openai_client import OpenAICompatibleLLM


def test_json_parser_accepts_fenced_json():
    assert OpenAICompatibleLLM._parse_json('```json\n{"ok": true}\n```') == {"ok": True}


def test_api_retry_policy_is_explicit_and_validated():
    config = ResearchConfig(request_timeout_seconds=30, max_api_attempts=2)
    assert config.request_timeout_seconds == 30
    assert config.max_api_attempts == 2
