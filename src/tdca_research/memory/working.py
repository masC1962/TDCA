from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Claim, ClaimStatus
from ..utils import normalize_text


@dataclass
class WorkingMemory:
    claims: dict[str, Claim] = field(default_factory=dict)

    def add(self, claim: Claim) -> Claim:
        if claim.claim_id in self.claims:
            raise ValueError(f"duplicate claim id {claim.claim_id}")
        for existing in self.claims.values():
            if existing.target_slot != claim.target_slot or existing.status not in {ClaimStatus.PROPOSED, ClaimStatus.VERIFIED}:
                continue
            if normalize_text(existing.object) == normalize_text(claim.object):
                if claim.calibrated_confidence > existing.calibrated_confidence:
                    existing.status = ClaimStatus.SUPERSEDED
                else:
                    claim.status = ClaimStatus.SUPERSEDED
            elif existing.subject and claim.subject and normalize_text(existing.subject) == normalize_text(claim.subject) and normalize_text(existing.relation) == normalize_text(claim.relation):
                existing.contradiction_ids.append(claim.claim_id)
                claim.contradiction_ids.append(existing.claim_id)
                if claim.status == ClaimStatus.VERIFIED and existing.status == ClaimStatus.VERIFIED:
                    if claim.calibrated_confidence > existing.calibrated_confidence:
                        existing.status = ClaimStatus.SUPERSEDED
                    elif existing.calibrated_confidence > claim.calibrated_confidence:
                        claim.status = ClaimStatus.SUPERSEDED
        self.claims[claim.claim_id] = claim
        return claim

    def set_status(self, claim_id: str, status: ClaimStatus) -> Claim:
        claim = self.claims[claim_id]
        claim.status = status
        return claim

    def all(self) -> list[Claim]:
        return list(self.claims.values())

    def verified(self, slot_id: str | None = None) -> list[Claim]:
        return [
            claim for claim in self.claims.values()
            if claim.status == ClaimStatus.VERIFIED and (slot_id is None or claim.target_slot == slot_id)
        ]

    def best(self, slot_id: str, include_proposed: bool = False) -> Claim | None:
        allowed = {ClaimStatus.VERIFIED}
        if include_proposed:
            allowed.add(ClaimStatus.PROPOSED)
        candidates = [claim for claim in self.claims.values() if claim.target_slot == slot_id and claim.status in allowed]
        return max(candidates, key=lambda claim: claim.calibrated_confidence, default=None)

