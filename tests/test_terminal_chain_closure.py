import unittest

from final_chain_buffer import FinalChainBuffer
from terminal_chain_closure import evaluate_terminal_chain_closure


class TerminalChainClosureTest(unittest.TestCase):
    def test_closure_prefers_complete_terminal_chain(self):
        question = "What city was the director of Inception born in?"
        buffer = FinalChainBuffer()
        buffer.add_candidate(
            target_question="Who directed Inception?",
            slot_key="bridge",
            slot_role="bridge_entity",
            answer_text="Christopher Nolan",
            support_score=0.8,
        )
        candidate = {
            "answer": "London",
            "target_question": question,
            "root_aligned": True,
            "root_alignment": 1.0,
            "dependency_satisfaction": 0.75,
            "last_hop_support": 0.72,
            "last_hop_verification": {"last_hop_support": 0.72, "last_hop_reason": "path_terminal_relation_match"},
            "support_score": 0.9,
            "depends_on": ["bridge_memory"],
            "composed_from_count": 1,
            "inferred_hop_count": 2,
        }
        score, info = evaluate_terminal_chain_closure(
            candidate,
            question,
            {"requires_structured_reasoning": True, "slots": [{"kind": "bridge", "slot_role": "bridge_entity", "question": "Who directed Inception?"}]},
            buffer,
        )
        self.assertGreaterEqual(score, 0.70)
        self.assertTrue(info["candidate_is_terminal_leaf"])

    def test_closure_penalizes_consumed_bridge_candidate(self):
        question = "What city was the director of Inception born in?"
        buffer = FinalChainBuffer()
        buffer.add_candidate(
            target_question="Who directed Inception?",
            slot_key="bridge",
            slot_role="bridge_entity",
            answer_text="Christopher Nolan",
            support_score=0.8,
        )
        candidate = {
            "answer": "Christopher Nolan",
            "target_question": "Who directed Inception?",
            "root_aligned": False,
            "root_alignment": 0.35,
            "dependency_satisfaction": 0.0,
            "last_hop_support": 0.55,
            "last_hop_verification": {"last_hop_support": 0.55, "last_hop_reason": "root_aligned_memory"},
            "support_score": 1.0,
            "is_bridge_entity": True,
            "inferred_hop_count": 2,
        }
        score, info = evaluate_terminal_chain_closure(
            candidate,
            question,
            {"requires_structured_reasoning": True, "slots": [{"kind": "bridge", "slot_role": "bridge_entity", "question": "Who directed Inception?"}]},
            buffer,
        )
        self.assertLess(score, 0.70)
        self.assertFalse(info["candidate_is_terminal_leaf"])
        self.assertIn("tcc_candidate_not_terminal", info["closure_fail_reasons"])


if __name__ == "__main__":
    unittest.main()
