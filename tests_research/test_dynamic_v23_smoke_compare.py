from scripts.compare_dynamic_v23_smoke import compare


def row(label, *, candidate, chain, corr, calls=100, unsupported=0):
    return {
        "label": label, "artifact_verified": True, "infrastructure_failures": 0,
        "exact_match": 0.4, "f1": 0.5, "candidate_presence": candidate,
        "full_chain_completion": chain, "selective_accuracy": 0.6,
        "budget_exhaustion": 0.1, "unsupported_answers": unsupported,
        "llm_calls": calls, "tokens": 10000, "retrieval_calls": 40,
        "mean_claim_count": 7.0, "mean_allocation_count": 12.0,
        "evc_utility_spearman": corr, "real_operation_choice_rate": 0.2,
        "extraction_bottleneck_count": 1,
        "join_attempt_count": 3, "join_accepted_count": 1,
        "join_charged_count": 2, "join_answer_used_count": 1,
        "join_llm_calls": 1,
        "per_example": {},
    }


def test_smoke_comparison_fails_closed_on_chain_regression():
    baseline = row("baseline", candidate=0.5, chain=0.65, corr=-0.05)
    candidate = row("candidate", candidate=0.7, chain=0.60, corr=0.2)
    report = compare(baseline, [candidate])
    assert report["decision"] == "NO_GO_FIX_BEFORE_CONTROLS"
    assert report["failed_checks"] == ["full_chain_non_regression"]


def test_smoke_comparison_opens_only_when_every_check_passes():
    baseline = row("baseline", candidate=0.5, chain=0.6, corr=-0.05)
    candidate = row("candidate", candidate=0.7, chain=0.65, corr=0.2)
    report = compare(baseline, [candidate])
    assert report["decision"] == "GO_MATCHED_CONTROLS"
    assert report["passed"]
