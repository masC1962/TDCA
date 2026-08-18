from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdca_research.data import load_examples
from tdca_research.evaluation.metrics import exact_match, token_f1


def _decode_answers(row: dict) -> list[str]:
    value = row.get("gold_answers")
    if isinstance(value, list):
        return [str(answer) for answer in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(answer) for answer in parsed]
    gold = row.get("gold")
    return [str(gold)] if gold is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently score a frozen legacy TDCA summary")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--dataset_name", default="musique")
    args = parser.parse_args()

    source_answers = {}
    if args.dataset:
        source_answers = {
            example.qid: example.answers
            for example in load_examples(args.dataset, args.dataset_name)
        }

    rows = []
    with Path(args.input).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            answer = str(source.get("pred") or "").strip()
            qid = str(source.get("sample_id") or source.get("id"))
            gold = source_answers.get(qid) or _decode_answers(source)
            rows.append({
                "qid": qid,
                "answer": answer,
                "exact_match": exact_match(answer, gold),
                "f1": token_f1(answer, gold),
                "answered": bool(answer),
                "status": "answer" if answer else "abstain",
                "llm_calls": source.get("llm_calls"),
                "completion_tokens": source.get("generated_tokens"),
                "stop_reason": source.get("stop_reason"),
            })

    result = {
        "count": len(rows),
        "exact_match": sum(row["exact_match"] for row in rows) / max(1, len(rows)),
        "f1": sum(row["f1"] for row in rows) / max(1, len(rows)),
        "answered_rate": sum(row["answered"] for row in rows) / max(1, len(rows)),
        "rows": rows,
        "warning": "Frozen legacy implementation independently rescored with tdca_research EM/F1; prompt-token usage is unavailable from legacy instrumentation.",
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(f"{args.output}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("count", "exact_match", "f1", "answered_rate")}))


if __name__ == "__main__":
    main()
