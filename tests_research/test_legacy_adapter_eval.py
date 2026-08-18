from external_baselines.evaluate_legacy_artifact import _decode_answers


def test_legacy_gold_answer_decoder_supports_json_list_and_scalar():
    assert _decode_answers({"gold_answers": '["one", "uno"]'}) == ["one", "uno"]
    assert _decode_answers({"gold": "two"}) == ["two"]
