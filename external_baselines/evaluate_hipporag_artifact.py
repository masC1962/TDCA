from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tdca_research.evaluation.metrics import exact_match, token_f1


def recover_answer(answer: str | None, response: str) -> tuple[str, str]:
    if answer and answer.strip():
        return answer.strip(), "upstream_parser"
    matches = re.findall(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{0,2})?(?:final\s+)?answer(?:\*{0,2})?\s*:\s*(.+?)\s*$",
        response or "",
    )
    if matches:
        answer = matches[-1].strip().strip("*").strip()
        if answer:
            return answer, "adapter_answer_line_recovery"
    return "", "unrecoverable"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HippoRAG answer parsing and independently score saved rows")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = []
    for row in artifact["rows"]:
        answer, source = recover_answer(row.get("answer"), row.get("response", ""))
        gold = [str(value) for value in row.get("gold_answers", [])]
        rows.append({
            "qid": row["qid"], "answer": answer, "answer_source": source,
            "status": "answer" if answer else "abstain",
            "exact_match": exact_match(answer, gold), "f1": token_f1(answer, gold),
        })
    parser_recoveries = sum(row["answer_source"] != "upstream_parser" and row["answer_source"] != "unrecoverable" for row in rows)
    parser_failures = sum(row["answer_source"] == "unrecoverable" for row in rows)
    result = {
        "count": len(rows),
        "exact_match": sum(row["exact_match"] for row in rows) / max(1, len(rows)),
        "f1": sum(row["f1"] for row in rows) / max(1, len(rows)),
        "parser_recoveries": parser_recoveries,
        "parser_failures": parser_failures,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    jsonl = Path(f"{args.output}.jsonl")
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("count", "exact_match", "f1", "parser_recoveries", "parser_failures")}))


if __name__ == "__main__":
    main()
