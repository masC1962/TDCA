from external_baselines.evaluate_hipporag_artifact import recover_answer


def test_recover_answer_prefers_upstream_parser():
    assert recover_answer("Paris", "Answer: London") == ("Paris", "upstream_parser")


def test_recover_answer_line_when_upstream_parser_failed():
    assert recover_answer(None, "Reasoning.\nFinal Answer: **Paris**") == ("Paris", "adapter_answer_line_recovery")


def test_recover_markdown_answer_line_when_upstream_parser_failed():
    assert recover_answer("", "Reasoning.\n### **Final Answer:** Paris") == ("Paris", "adapter_answer_line_recovery")
