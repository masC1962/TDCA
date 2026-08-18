from tdca_research.models import Claim, Passage, RetrievalHit
from tdca_research.verification import ClaimVerifier


def test_grounding_accepts_visible_title_span_as_well_as_body_span():
    hit = RetrievalHit(Passage("p", "George Berkeley", "He was a philosopher."), 1.0, 1, "bm25", "q")
    title_claim = Claim("c", "", "", "George Berkeley", "person", "s", ["p"], ["George Berkeley"])
    body_claim = Claim("d", "", "", "philosopher", "entity", "s", ["p"], ["He was a philosopher"])
    assert ClaimVerifier._spans_are_grounded(title_claim, [hit])
    assert ClaimVerifier._spans_are_grounded(body_claim, [hit])
