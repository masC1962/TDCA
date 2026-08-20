from tdca_research.llm.openai_client import _is_provider_refusal


def test_provider_content_policy_errors_are_classified_as_refusals():
    assert _is_provider_refusal(RuntimeError("code=data_inspection_failed"))
    assert _is_provider_refusal(RuntimeError("content_policy_violation"))
    assert not _is_provider_refusal(RuntimeError("connection timed out"))
