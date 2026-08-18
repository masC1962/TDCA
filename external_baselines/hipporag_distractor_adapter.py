from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import sqlite3
from dataclasses import replace
from pathlib import Path

from hipporag import HippoRAG
from hipporag.embedding_model.Transformers import TransformersEmbeddingModel
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.utils.config_utils import BaseConfig


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def cache_usage(cache_file: Path) -> dict[str, int]:
    if not cache_file.exists():
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    connection = sqlite3.connect(cache_file)
    metadata_rows = connection.execute("SELECT metadata FROM cache").fetchall()
    connection.close()
    metadata = [json.loads(row[0]) for row in metadata_rows]
    return {
        "calls": len(metadata),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in metadata),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official HippoRAG independently on each distractor context")
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--shared_cache_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm_base_url", required=True)
    parser.add_argument("--llm_name", required=True)
    parser.add_argument("--embedding_name", default="Transformers/sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    samples = json.loads(Path(args.dataset_file).read_text(encoding="utf-8"))
    common = BaseConfig(
        save_dir=args.save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset="musique",
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=20,
        linking_top_k=5,
        qa_top_k=5,
        embedding_batch_size=8,
        openie_mode="online",
    )
    shared_llm = CacheOpenAI(cache_dir=args.shared_cache_dir, global_config=common)
    cache_file = Path(shared_llm.cache_file_name)
    cache_before = cache_usage(cache_file)
    shared_embedding = TransformersEmbeddingModel(common, args.embedding_name)
    rows = []
    started = time.perf_counter()
    for sample_index, sample in enumerate(samples):
        qid = str(sample["id"])
        config = replace(common, save_dir=str(Path(args.save_dir) / safe_id(qid)))
        docs = [
            f"{paragraph['title']}\n{paragraph.get('text', paragraph.get('paragraph_text', ''))}"
            for paragraph in sample.get("paragraphs", [])
        ]
        gold_docs = [[
            f"{paragraph['title']}\n{paragraph.get('text', paragraph.get('paragraph_text', ''))}"
            for paragraph in sample.get("paragraphs", []) if paragraph.get("is_supporting", True)
        ]]
        value = sample.get("answer", [])
        answers = ([value] if isinstance(value, str) else list(value)) + list(sample.get("answer_aliases", []))
        rag = HippoRAG(
            global_config=config,
            extraction_llm=shared_llm,
            qa_llm=shared_llm,
            embedding_model=shared_embedding,
        )
        rag.index(docs)
        solutions, responses, metadata, retrieval, qa = rag.rag_qa(
            queries=[sample["question"]], gold_docs=gold_docs, gold_answers=[answers],
        )
        rows.append({
            "qid": qid,
            "retrieval_metrics": retrieval,
            "qa_metrics": qa,
            "response": responses[0],
            "metadata": metadata[0],
            **solutions[0].to_dict(),
        })
        checkpoint = {
            "implementation": "official_code_controlled_configuration",
            "setting": "distractor_per_question_graph",
            "completed_count": sample_index + 1,
            "sample_count": len(samples),
            "rows": rows,
            "warning": "Partial checkpoint only; never use partial aggregates as a reported result.",
        }
        Path(f"{args.output}.partial").write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    def mean_metric(group: str, key: str) -> float:
        return sum(float(row[group][key]) for row in rows) / max(1, len(rows))

    retrieval_keys = list(rows[0]["retrieval_metrics"]) if rows else []
    qa_keys = list(rows[0]["qa_metrics"]) if rows else []
    artifact = {
        "implementation": "official_code_controlled_configuration",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "setting": "distractor_per_question_graph",
        "sample_count": len(rows),
        "sample_ids": [row["qid"] for row in rows],
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "embedding_name": args.embedding_name,
        "wall_seconds": time.perf_counter() - started,
        "shared_cache_before": cache_before,
        "shared_cache_after": cache_usage(cache_file),
        "retrieval_metrics": {key: mean_metric("retrieval_metrics", key) for key in retrieval_keys},
        "qa_metrics": {key: mean_metric("qa_metrics", key) for key in qa_keys},
        "rows": rows,
        "warning": "Official HippoRAG code with Qwen-plus and controlled MiniLM embedding; each question has an independent graph over its original distractor passages. This is not the paper-default NV-Embed-v2 configuration.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    Path(f"{args.output}.partial").unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "qa_metrics": artifact["qa_metrics"], "retrieval_metrics": artifact["retrieval_metrics"]}))


if __name__ == "__main__":
    main()
