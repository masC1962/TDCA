import json
import unittest

from core_models import HeteroGraph, Node, NodeType
from config import TDCAConfig
from final_chain_buffer import FinalChainBuffer
from tdca_scheduler import TDCAScheduler
from terminal_memory_consolidation import (
    consolidate_terminal_memories,
    evaluate_terminal_memories_with_tcc,
)


class TerminalMemoryConsolidationTest(unittest.TestCase):
    def _build_mock_chain(self):
        question = "What is the terminal answer reached from the starting entity?"
        graph = HeteroGraph()
        memory_a = Node(
            node_id="memory_A",
            node_type=NodeType.MEMORY,
            content="The starting entity leads to entity1.",
            value=0.90,
            metadata={
                "answer_text": "entity1",
                "target_question": "Which entity follows the starting entity?",
                "slot_role": "bridge_entity",
                "support_score": 0.90,
            },
        )
        memory_b = Node(
            node_id="memory_B",
            node_type=NodeType.MEMORY,
            content="entity1 leads to entity2.",
            value=0.91,
            metadata={
                "answer_text": "entity2",
                "target_question": "Which entity follows entity1?",
                "slot_role": "bridge_entity",
                "support_score": 0.91,
                "depends_on": ["memory_A"],
            },
        )
        memory_c = Node(
            node_id="memory_C",
            node_type=NodeType.MEMORY,
            content="entity2 leads to the terminal answer.",
            value=0.94,
            metadata={
                "answer_text": "terminal answer",
                "target_question": question,
                "target_question_norm": question.lower(),
                "slot_role": "root_answer",
                "support_score": 0.94,
                "depends_on": ["memory_B"],
                "composed_from": ["memory_A", "memory_B"],
                "terminal": True,
                "path_terminal": True,
                "last_hop_support": 0.90,
            },
        )
        for memory in [memory_a, memory_b, memory_c]:
            graph.add_node(memory)
        goal_plan = {
            "requires_structured_reasoning": True,
            "slots": [
                {"question": "Which entity follows the starting entity?", "slot_role": "bridge_entity"},
                {"question": "Which entity follows entity1?", "slot_role": "bridge_entity"},
                {"question": question, "slot_role": "root_answer"},
            ],
        }
        return question, graph, goal_plan, memory_c

    def test_dependency_chain_consolidates_and_tcc_can_close_or_reject(self):
        question, graph, goal_plan, root_memory = self._build_mock_chain()
        buffer = FinalChainBuffer()
        terminal_graph = consolidate_terminal_memories(
            question=question,
            goal_plan=goal_plan,
            graph=graph,
            final_chain_buffer=buffer,
            current_run_memory_node_ids={"memory_A", "memory_B", "memory_C"},
            root_memory=root_memory,
        )

        self.assertGreater(terminal_graph["unit_count"], 0)
        self.assertGreater(terminal_graph["terminal_count"], 0)
        terminal = next(item for item in terminal_graph["terminals"] if item["answer"] == "terminal answer")
        dependency_chain = set(terminal["depends_on"] + terminal["composed_from"])
        self.assertEqual(dependency_chain, {"memory_A", "memory_B"})

        closed_results = evaluate_terminal_memories_with_tcc(
            terminal_memory_graph=terminal_graph,
            question=question,
            goal_plan=goal_plan,
            final_chain_buffer=buffer,
            graph=graph,
            score_threshold=0.0,
        )
        closed = next(item for item in closed_results if item["answer"] == "terminal answer")
        self.assertEqual(closed["candidate"]["source"], "terminal_memory")
        self.assertTrue(closed["tcc_passed"])
        json.dumps(closed_results)

        scheduler = TDCAScheduler(
            llm=None,
            graph=graph,
            evaluator=None,
            evidence_store=None,
            memory_bank=None,
            config=TDCAConfig(),
        )
        scheduler.terminal_memory_graph = terminal_graph
        scheduler.tmc_tcc_results = closed_results
        debug_unit = next(
            item for item in scheduler._terminal_memory_debug_units()
            if item["answer_candidate"] == "terminal answer"
        )
        self.assertEqual(set(debug_unit["dependency_chain"]), {"memory_A", "memory_B", "memory_C"})
        self.assertGreater(debug_unit["terminality"], 0.0)

        rejected_results = evaluate_terminal_memories_with_tcc(
            terminal_memory_graph=terminal_graph,
            question=question,
            goal_plan=goal_plan,
            final_chain_buffer=buffer,
            graph=graph,
            score_threshold=1.01,
        )
        rejected = next(item for item in rejected_results if item["answer"] == "terminal answer")
        self.assertFalse(rejected["tcc_passed"])
        self.assertIn("tcc_score_below_threshold", rejected["fail_reasons"])

    def test_terminal_memory_enters_final_candidate_collection(self):
        config = TDCAConfig()
        config.enable_terminal_memory_consolidation = True
        config.final_answer_judge_max_candidates = 5
        scheduler = TDCAScheduler(
            llm=None,
            graph=HeteroGraph(),
            evaluator=None,
            evidence_store=None,
            memory_bank=None,
            config=config,
        )
        scheduler.tmc_triggered = True
        scheduler.terminal_memory_graph = {"terminal_count": 1}
        scheduler.tmc_tcc_results = [{
            "terminal_id": "tmc_answer",
            "answer": "terminal answer",
            "tcc_score": 0.91,
            "tcc_passed": True,
            "terminal_memory": {
                "terminal_id": "tmc_answer",
                "answer": "terminal answer",
                "consolidated_from": ["mem:memory_A", "mem:memory_B", "mem:memory_C"],
                "dependency_coverage": 1.0,
            },
        }]

        scheduler._attempt_compose_root_memory = lambda question: None
        scheduler._attempt_infer_final_chain_root_memory = lambda question: None
        scheduler._attempt_score_based_final_chain_root_memory = lambda question: None
        scheduler._is_final_chain_candidate_memory = lambda question, memory: False
        scheduler._candidate_from_tmc_result = lambda question, result: {
            "answer": result["answer"],
            "source": "terminal_memory",
            "candidate_source": "terminal_memory",
            "terminal_memory_id": result["terminal_id"],
            "consolidated_from": result["terminal_memory"]["consolidated_from"],
            "dependency_coverage": 1.0,
            "terminality": 0.9,
            "terminal_chain_closure_score": result["tcc_score"],
            "terminal_chain_closure_info": {"terminality": 0.9},
            "base_score": result["tcc_score"],
            "root_goal_satisfied": True,
        }
        scheduler._candidate_rerank_score = lambda question, candidate: candidate["base_score"]
        scheduler._candidate_terminal_tier = lambda question, candidate: 1
        scheduler._apply_tcc_final_audit = lambda question, candidates: candidates

        candidates = scheduler._collect_answer_candidates(
            "Mock question",
            "",
            None,
            [],
            None,
        )

        self.assertEqual(candidates[0]["source"], "terminal_memory")
        self.assertTrue(scheduler.tmc_entered_final_candidate)
        self.assertEqual(scheduler.tmc_final_candidate_entry_fail_reason, "")
        self.assertEqual(scheduler.tmc_final_candidate_records[0]["terminal_memory_id"], "tmc_answer")

        scheduler._reset_terminal_memory_sample_state()
        self.assertEqual(scheduler.terminal_memory_graph, {})
        self.assertEqual(scheduler.tmc_tcc_results, [])
        self.assertFalse(scheduler.tmc_entered_final_candidate)


if __name__ == "__main__":
    unittest.main()
