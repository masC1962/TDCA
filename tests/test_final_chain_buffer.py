import unittest

from final_chain_buffer import FinalChainBuffer, canonical_slot_key, score_final_chain_candidate


class FinalChainBufferTest(unittest.TestCase):
    def test_group_best_and_score(self):
        question = "What city was the director of Inception born in?"
        slot_key = canonical_slot_key(question, "location", "root_answer", "Inception")
        buffer = FinalChainBuffer()
        buffer.add_candidate(
            target_question=question,
            slot_key=slot_key,
            slot_role="root_answer",
            answer_text="London",
            answer_type="location",
            evidence_ids=["doc1"],
            support_score=0.9,
            source="root_memory",
        )

        self.assertEqual(buffer.best_for_slot(slot_key).answer_text, "London")
        score, parts = score_final_chain_candidate(
            {
                "root_aligned": True,
                "coverage_ratio": 1.0,
                "support_score": 0.9,
                "type_score": 1.0,
                "answer_type_match": 1.0,
                "dependency_satisfaction": 0.8,
                "last_hop_support": 0.9,
                "composed_from_count": 1,
            },
            question,
            {"slots": [{"question": question, "slot_type": "location", "slot_role": "root_answer"}]},
            buffer,
        )
        self.assertGreaterEqual(score, 0.90)
        self.assertEqual(parts["root_alignment"], 1.0)
        self.assertEqual(parts["last_hop_support"], 0.9)


if __name__ == "__main__":
    unittest.main()
