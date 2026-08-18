from __future__ import annotations

import time

from ..budget import Budget, BudgetExceeded
from ..config import ResearchConfig
from ..llm import BaseLLM, InfrastructureError, StructuredOutputError
from ..memory import WorkingMemory
from ..models import ClaimStatus, Prediction, QAExample, RunStatus, SlotStatus, Usage
from ..planning import Planner, bind_slot_question, ready_slots
from ..retrieval import BaseRetriever
from ..scheduling import Scheduler, SlotSignals, diffuse_temperatures
from ..verification import ClaimVerifier
from .extractor import ClaimExtractor
from .finalizer import Finalizer


class StructuredReasoner:
    def __init__(self, llm: BaseLLM, retriever: BaseRetriever, config: ResearchConfig) -> None:
        self.llm = llm
        self.retriever = retriever
        self.config = config

    def solve(self, example: QAExample) -> tuple[Prediction, list[dict], list[dict]]:
        started = time.perf_counter()
        usage = Usage()
        budget = Budget(self.config.max_llm_calls, self.config.max_total_tokens, self.config.final_reserve_tokens, usage)
        memory = WorkingMemory()
        retrieval_trace: list[dict] = []
        reasoning_trace: list[dict] = []
        plan = None
        best_unverified: str | None = None
        best_unverified_confidence = -1.0
        rejection_reasons: list[str] = []
        all_hits = []
        try:
            planner = Planner(self.llm, budget, self.config.planner_max_tokens, self.config.temperature)
            plan = planner.create(example, oracle=self.config.oracle_decomposition)
            if self.config.memory_mode == "none" or not self.config.use_dependency_dag:
                # A real direct-reasoning ablation: discard all decomposed state.
                from ..models import ReasoningPlan, ReasoningSlot
                from ..planning import validate_plan
                plan = validate_plan(ReasoningPlan(
                    question=example.question,
                    slots=[ReasoningSlot(
                        slot_id="slot_root", subquestion_template=example.question,
                        answer_type="entity", output_variable="$answer", terminal=True,
                        confidence=0.1,
                    )],
                    plan_type="direct_ablation",
                    source="ablation_no_memory" if self.config.memory_mode == "none" else "ablation_no_dag",
                ))
            extractor = ClaimExtractor(
                self.llm, budget, self.config.claim_max_tokens, self.config.evidence_char_budget,
                self.config.temperature, self.config.evidence_compaction,
            )
            verifier = ClaimVerifier(
                self.llm, budget, self.config.verifier_max_tokens, self.config.evidence_char_budget,
                self.config.min_claim_confidence, self.config.temperature, self.config.evidence_compaction,
            )
            scheduler = Scheduler(self.config.scheduler, self.config.beam_width)
            temperatures = {slot.slot_id: max(0.1, slot.confidence) for slot in plan.slots}
            step = 0
            while step < self.config.max_steps:
                candidates = ready_slots(plan, memory.all())
                if not candidates:
                    break
                signals = self._signals(plan, candidates, memory, temperatures)
                scheduled = scheduler.rank(candidates, signals)
                if not scheduled:
                    break
                scheduled = scheduled if self.config.scheduler == "beam" else scheduled[:1]
                made_progress = False
                for slot in scheduled:
                    if step >= self.config.max_steps:
                        break
                    step += 1
                    progressed, candidate, reasons = self._execute_slot(
                        example, plan, slot, signals, memory, extractor, verifier, budget,
                        retrieval_trace, reasoning_trace, all_hits, step,
                    )
                    made_progress = progressed or made_progress
                    if candidate is not None and candidate.status != ClaimStatus.VERIFIED:
                        if candidate.calibrated_confidence >= best_unverified_confidence:
                            best_unverified = candidate.object
                            best_unverified_confidence = candidate.calibrated_confidence
                    rejection_reasons.extend(reasons)
                    if self.config.scheduler in {"diffusion", "tdca"}:
                        temperatures = diffuse_temperatures(plan, temperatures, self.config.diffusion_alpha, self.config.diffusion_decay)
                        temperatures[slot.slot_id] = 0.0
                terminal = [candidate for candidate in plan.slots if candidate.terminal]
                if terminal and all(memory.best(candidate.slot_id) is not None for candidate in terminal):
                    break
                if not made_progress and not ready_slots(plan, memory.all()):
                    break
            prediction = self._finish(example, plan, memory, all_hits, budget, best_unverified, rejection_reasons)
        except StructuredOutputError as exc:
            # The provider call happened even though decoding failed. Preserve its
            # tokens/call/attempt before returning a distinct infrastructure failure.
            try:
                budget.record_generation(exc.generation)
            except BudgetExceeded:
                pass
            prediction = Prediction(
                qid=example.qid, question=example.question, status=RunStatus.INFRASTRUCTURE_FAILURE,
                answer=None, confidence=0.0, stop_reason="structured_output_failure",
                best_unverified_candidate=best_unverified, rejection_reasons=rejection_reasons,
                claims=memory.all(), plan=plan, retrieved=all_hits, usage=usage, error=str(exc),
            )
        except (TypeError, ValueError, KeyError) as exc:
            # Provider JSON may be syntactically valid but violate a field type.
            # Keep the failure local to this question; do not hide arbitrary
            # programming errors with a blanket Exception handler.
            prediction = Prediction(
                qid=example.qid, question=example.question, status=RunStatus.INFRASTRUCTURE_FAILURE,
                answer=None, confidence=0.0, stop_reason="invalid_structured_payload",
                best_unverified_candidate=best_unverified, rejection_reasons=rejection_reasons,
                claims=memory.all(), plan=plan, retrieved=all_hits, usage=usage,
                error=f"{type(exc).__name__}: {exc}",
            )
        except InfrastructureError as exc:
            budget.record_infrastructure_failure(exc)
            prediction = Prediction(
                qid=example.qid, question=example.question, status=RunStatus.INFRASTRUCTURE_FAILURE,
                answer=None, confidence=0.0, stop_reason="infrastructure_failure", best_unverified_candidate=best_unverified,
                rejection_reasons=rejection_reasons, claims=memory.all(), plan=plan, retrieved=all_hits,
                usage=usage, error=str(exc),
            )
        except BudgetExceeded:
            # Never make another provider call after the guard has fired. A
            # terminal claim here has already passed the configured verifier.
            prediction = self._finish(
                example, plan, memory, all_hits, budget, best_unverified,
                rejection_reasons + ["budget_exhausted"], allow_llm=False,
            )
        usage.wall_seconds = time.perf_counter() - started
        prediction.usage = usage
        return prediction, retrieval_trace, reasoning_trace

    def _execute_slot(self, example, plan, slot, signals, memory, extractor, verifier, budget, retrieval_trace, reasoning_trace, all_hits, step):
        if self.config.explicit_variable_binding:
            bound_question, dependency_claim_ids = bind_slot_question(slot, plan, memory.all())
        else:
            # The no-binding ablation retrieves with the root question; it never
            # substitutes hidden/gold variables into a successor query.
            bound_question = example.question
            dependency_claim_ids = [claim.claim_id for dependency in slot.dependencies for claim in memory.verified(dependency)]
            slot.bound_question = bound_question
        slot.status = SlotStatus.RUNNING
        hits = self._retrieve(example, bound_question, budget, slot)
        seen_passages = {hit.passage.passage_id for hit in all_hits}
        all_hits.extend(hit for hit in hits if hit.passage.passage_id not in seen_passages)
        retrieval_trace.append({"step": step, "slot_id": slot.slot_id, "query": bound_question, "hits": [hit.to_dict() for hit in hits]})
        claim = extractor.extract(slot, bound_question, hits, dependency_claim_ids, step, example.question)
        if claim is None:
            slot.status = SlotStatus.FAILED
            reasoning_trace.append({"step": step, "slot_id": slot.slot_id, "status": "no_candidate"})
            return False, None, ["no_candidate"]
        if self.config.memory_mode == "text":
            claim.subject = ""
            claim.relation = ""
            claim.answer_type = "text"
        if self.config.verifier == "independent":
            dependencies_complete = all(memory.best(dependency) is not None for dependency in slot.dependencies)
            dependency_claims = [
                memory.best(dependency) for dependency in slot.dependencies
                if memory.best(dependency) is not None
            ]
            claim, reasons = verifier.verify(
                claim, slot, bound_question, hits,
                dependencies_complete=dependencies_complete,
                root_question=example.question,
                dependency_claims=dependency_claims,
            )
        else:
            grounded = verifier._spans_are_grounded(claim, hits)
            reasons = [] if grounded else ["source_span_not_grounded"]
            if self.config.verifier == "none":
                claim.calibrated_confidence = min(claim.calibrated_confidence, 0.60) if grounded else 0.0
            # self verification uses only the extractor's own reported confidence.
            claim.status = ClaimStatus.VERIFIED if grounded and claim.calibrated_confidence >= self.config.min_claim_confidence else ClaimStatus.REJECTED
            if claim.status == ClaimStatus.REJECTED and not reasons:
                reasons.append("self_or_no_verifier_below_threshold")
        memory.add(claim)
        if claim.status == ClaimStatus.VERIFIED:
            slot.status = SlotStatus.COMPLETE
            slot.confidence = claim.calibrated_confidence
        else:
            slot.status = SlotStatus.FAILED
        reasoning_trace.append({
            "step": step, "slot_id": slot.slot_id, "bound_question": bound_question,
            "claim": claim.to_dict(), "verification_reasons": reasons,
            "scheduler_signal": signals[slot.slot_id].__dict__,
        })
        return claim.status == ClaimStatus.VERIFIED, claim, reasons

    def _retrieve(self, example: QAExample, query: str, budget: Budget, slot=None):
        budget.record_retrieval()
        if self.config.oracle_evidence:
            wanted = set(example.gold_document_ids)
            plan_index = self._slot_index(slot.slot_id) if slot is not None else None
            if self.config.oracle_decomposition and plan_index is not None:
                if plan_index <= len(example.oracle_decomposition):
                    step = example.oracle_decomposition[plan_index - 1]
                    support_id = step.get("paragraph_support_idx")
                    support = step.get("support_paragraph") if isinstance(step.get("support_paragraph"), dict) else {}
                    if support_id is None:
                        support_id = support.get("idx", support.get("id"))
                    if support_id is not None:
                        wanted = {str(support_id)}
            source_passages = example.passages
            if self.config.setting == "global":
                source_passages = self._retriever_passages(self.retriever)
            passages = [passage for passage in source_passages if passage.passage_id in wanted]
            from ..models import RetrievalHit
            return [RetrievalHit(passage, 1.0, rank, "oracle_evidence", query) for rank, passage in enumerate(passages, start=1)]
        return self.retriever.search(query, self.config.top_k)

    @staticmethod
    def _slot_index(slot_id: str) -> int | None:
        import re

        match = re.search(r"(\d+)$", slot_id)
        return int(match.group(1)) if match else None

    @staticmethod
    def _retriever_passages(retriever):
        if hasattr(retriever, "passages"):
            return list(retriever.passages)
        if hasattr(retriever, "sparse"):
            return StructuredReasoner._retriever_passages(retriever.sparse)
        if hasattr(retriever, "base"):
            return StructuredReasoner._retriever_passages(retriever.base)
        raise ValueError("oracle evidence cannot enumerate this retriever's global corpus")

    @staticmethod
    def _signals(plan, candidates, memory, temperatures):
        children = {slot.slot_id: 0 for slot in plan.slots}
        for slot in plan.slots:
            for dependency in slot.dependencies:
                children[dependency] += 1
        signals = {}
        for slot in candidates:
            existing = memory.best(slot.slot_id, include_proposed=True)
            confidence = existing.calibrated_confidence if existing else 0.0
            signals[slot.slot_id] = SlotSignals(
                expected_information_gain=max(0.05, 1.0 - confidence),
                dependency_unlock_value=1.0 + children[slot.slot_id],
                evidence_gap=max(0.05, 1.0 - confidence),
                confidence_need=max(0.05, 1.0 - confidence),
                expected_cost=1.0 + 0.25 * len(slot.dependencies),
                value=slot.confidence,
                temperature=temperatures.get(slot.slot_id, 0.0),
            )
        return signals

    def _finish(self, example, plan, memory, all_hits, budget, best_unverified, rejection_reasons, *, allow_llm=True):
        terminal_claims = []
        if plan is not None:
            for slot in plan.slots:
                if slot.terminal:
                    claim = memory.best(slot.slot_id)
                    if claim is not None:
                        terminal_claims.append(claim)
        if not terminal_claims:
            return Prediction(
                qid=example.qid, question=example.question, status=RunStatus.ABSTAIN, answer=None, confidence=0.0,
                stop_reason="no_verified_terminal_candidate", best_unverified_candidate=best_unverified,
                rejection_reasons=list(dict.fromkeys(rejection_reasons)), claims=memory.all(), plan=plan, retrieved=all_hits, usage=budget.usage,
            )
        selected = max(terminal_claims, key=lambda claim: claim.calibrated_confidence)
        if selected.calibrated_confidence < self.config.min_answer_confidence:
            return Prediction(
                qid=example.qid, question=example.question, status=RunStatus.ABSTAIN, answer=None,
                confidence=selected.calibrated_confidence, stop_reason="terminal_candidate_below_threshold",
                best_unverified_candidate=selected.object, rejection_reasons=list(dict.fromkeys(rejection_reasons)),
                claims=memory.all(), plan=plan, retrieved=all_hits, usage=budget.usage,
            )
        if self.config.finalization == "direct" or not allow_llm:
            return Prediction(
                qid=example.qid, question=example.question, status=RunStatus.ANSWER, answer=selected.object,
                confidence=selected.calibrated_confidence,
                stop_reason="direct_terminal_candidate_ablation" if allow_llm else "verified_terminal_budget_exhausted",
                best_unverified_candidate=selected.object,
                rejection_reasons=list(dict.fromkeys(rejection_reasons)), claims=memory.all(), plan=plan,
                retrieved=all_hits, usage=budget.usage,
            )
        finalizer = Finalizer(
            self.llm, budget, self.config.final_max_tokens, self.config.evidence_char_budget,
            self.config.min_answer_confidence, self.config.temperature, self.config.evidence_compaction,
        )
        answer, confidence, final_reasons = finalizer.finalize(example.question, plan, selected, memory.verified(), all_hits)
        if answer is None:
            return Prediction(
                qid=example.qid, question=example.question, status=RunStatus.ABSTAIN, answer=None,
                confidence=confidence, stop_reason="final_verification_rejected",
                best_unverified_candidate=selected.object,
                rejection_reasons=list(dict.fromkeys(rejection_reasons + final_reasons)), claims=memory.all(),
                plan=plan, retrieved=all_hits, usage=budget.usage,
            )
        return Prediction(
            qid=example.qid, question=example.question, status=RunStatus.ANSWER, answer=answer,
            confidence=confidence, stop_reason="verified_terminal_candidate",
            best_unverified_candidate=None, rejection_reasons=[], claims=memory.all(), plan=plan,
            retrieved=all_hits, usage=budget.usage,
        )
