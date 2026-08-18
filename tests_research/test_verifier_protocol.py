from tdca_research.budget import Budget
from tdca_research.llm import DeterministicMockLLM
from tdca_research.models import Claim, Passage, ReasoningSlot, RetrievalHit, Usage
from tdca_research.verification import ClaimVerifier


def test_generic_entity_type_cannot_be_falsely_rejected_as_type_mismatch():
    llm = DeterministicMockLLM(json_responses=[{
        "evidence_relevance": 1, "relation_entailment": 1, "answer_type_match": 0,
        "dependency_consistency": 1, "contradiction_detected": False, "confidence": 1,
        "reasons": [],
    }])
    hit = RetrievalHit(Passage("p", "Green Bay", "Green Bay is the county seat."), 1, 1, "bm25", "q")
    claim = Claim("c", "county", "seat", "Green Bay", "entity", "s", ["p"], ["Green Bay is the county seat."])
    verified, reasons = ClaimVerifier(llm, Budget(2, 2000, 0, Usage()), 200, 1000).verify(
        claim, ReasoningSlot("s", "What is the seat?", answer_type="entity", terminal=True),
        "What is the seat?", [hit], True,
    )
    assert verified.type_score == 1.0
    assert "answer_type_mismatch" not in reasons
