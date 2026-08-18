from tdca_research.memory import WorkingMemory
from tdca_research.models import Claim, ClaimStatus


def claim(identifier, obj, confidence):
    return Claim(identifier, "entity", "located in", obj, "location", "slot", calibrated_confidence=confidence, status=ClaimStatus.VERIFIED)


def test_contradiction_and_supersession():
    memory = WorkingMemory()
    old = memory.add(claim("old", "North", 0.6))
    new = memory.add(claim("new", "South", 0.9))
    assert old.status == ClaimStatus.SUPERSEDED
    assert new.status == ClaimStatus.VERIFIED
    assert new.contradiction_ids == ["old"]
    assert memory.best("slot").claim_id == "new"


def test_rejected_claim_never_becomes_best():
    memory = WorkingMemory()
    bad = claim("bad", "Wrong", 0.99)
    bad.status = ClaimStatus.REJECTED
    memory.add(bad)
    assert memory.best("slot") is None

