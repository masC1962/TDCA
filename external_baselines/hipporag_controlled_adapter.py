from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def gold_docs(samples: list[dict]) -> list[list[str]]:
    return [[
        f"{paragraph['title']}\n{paragraph.get('text', paragraph.get('paragraph_text', ''))}"
        for paragraph in sample.get("paragraphs", []) if paragraph.get("is_supporting", True)
    ] for sample in samples]


def gold_answers(samples: list[dict]) -> list[list[str]]:
    answers = []
    for sample in samples:
        value = sample.get("answer", sample.get("gold_ans", []))
        row = [value] if isinstance(value, str) else list(value)
        row.extend(sample.get("answer_aliases", []))
        answers.append(list(dict.fromkeys(str(item) for item in row)))
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled official-code HippoRAG adapter")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", default="reproduce/dataset")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm_base_url", required=True)
    parser.add_argument("--llm_name", required=True)
    parser.add_argument("--embedding_name", default="Transformers/sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    source = Path(args.data_dir)
    samples = json.loads((source / f"{args.dataset}.json").read_text(encoding="utf-8"))
    corpus = json.loads((source / f"{args.dataset}_corpus.json").read_text(encoding="utf-8"))
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    run_dir = f"{args.save_dir}_{args.dataset}"
    config = BaseConfig(
        save_dir=run_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset="musique",
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=args.force_rebuild,
        force_openie_from_scratch=args.force_rebuild,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=5,
        embedding_batch_size=8,
        openie_mode="online",
    )
    started = time.perf_counter()
    rag = HippoRAG(global_config=config)
    rag.index(docs)
    solutions, responses, metadata, retrieval_metrics, qa_metrics = rag.rag_qa(
        queries=[sample["question"] for sample in samples],
        gold_docs=gold_docs(samples),
        gold_answers=gold_answers(samples),
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    artifact = {
        "implementation": "official_code_controlled_configuration",
        "repository_commit": commit,
        "dataset": args.dataset,
        "setting": "mini_global_union",
        "sample_ids": [str(sample.get("id")) for sample in samples],
        "sample_count": len(samples),
        "corpus_count": len(corpus),
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "embedding_name": args.embedding_name,
        "force_rebuild": args.force_rebuild,
        "wall_seconds": time.perf_counter() - started,
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "rows": [
            {"qid": str(sample.get("id")), "response": response, "metadata": meta, **solution.to_dict()}
            for sample, solution, response, meta in zip(samples, solutions, responses, metadata)
        ],
        "warning": "Official HippoRAG code at a pinned commit with Qwen-plus and a controlled MiniLM embedding; not the paper-default NV-Embed-v2 reproduction and not distractor-equivalent.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "qa_metrics": qa_metrics, "retrieval_metrics": retrieval_metrics}))


if __name__ == "__main__":
    main()
