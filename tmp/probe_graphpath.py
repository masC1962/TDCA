import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TDCAConfig
from core_models import HeteroGraph
from knowledge_memory import EvidenceStore, MemoryBank
from llm_evaluator import MockLLM, ValueEvaluator
from tdca_scheduler import TDCAScheduler
cfg=TDCAConfig(); cfg.llm_backend='mock'
llm=MockLLM(); graph=HeteroGraph(); ev=EvidenceStore('batch_outputs/musique_ans_v1.0_dev_20260520_104731/runs/000_2hop__460946_294723/runtime_evidence.jsonl'); mem=MemoryBank('data/demo_memories.jsonl'); sch=TDCAScheduler(llm, graph, ValueEvaluator(llm=llm,value_weights=cfg.value_weights), ev, mem, cfg)
for q,t in [('Who is the Green performer mentioned in the evidence?','person'),('Who is the spouse of Steve Hillage?','person')]:
    items,_=sch._retrieve_context(q)
    print(q)
    print([it.metadata.get('title') for it in items])
    print('path=', sch._graph_path_answer_for_slot(q,t,items))
    print('slot=', sch._extract_answer_for_slot(q,t,items))
