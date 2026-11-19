"""Merge and summarize Jev judgments for the six held-out conditions."""
import argparse
import json
from pathlib import Path

CONDITIONS = ("base", "rag", "lora", "lora_rag", "random_rag", "schema")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge(results_dir, output, summary):
    merged = []
    summary_rows = []
    for condition in CONDITIONS:
        path = Path(results_dir) / f"jev_{condition}_v1.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 62 or len({r["case_id"] for r in rows}) != 62:
            raise ValueError(f"Expected 62 unique cases for {condition}")
        if any(r.get("judge_status") != "ok" for r in rows):
            raise ValueError(f"Missing Jev judgment for {condition}")
        probabilities = {
            "faithfulness": [r["faithfulness_probability"] for r in rows],
            "answer_relevancy": [r["answer_relevancy_probability"] for r in rows],
        }
        summary_rows.append({
            "condition": condition,
            "n": len(rows),
            "judge_model_returned": sorted({r.get("judge_returned") for r in rows}),
            "faithfulness_binary": sum(r["faithfulness"] for r in rows),
            "faithfulness_rate": sum(r["faithfulness"] for r in rows) / len(rows),
            "faithfulness_probability_mean": sum(probabilities["faithfulness"]) / len(rows),
            "answer_relevancy_binary": sum(r["answer_relevancy"] for r in rows),
            "answer_relevancy_rate": sum(r["answer_relevancy"] for r in rows) / len(rows),
            "answer_relevancy_probability_mean": sum(probabilities["answer_relevancy"]) / len(rows),
        })
        for row in rows:
            merged.append({
                "case_id": row["case_id"],
                "condition": row["condition"],
                "status": row["status"],
                "faithfulness": row["faithfulness"],
                "answer_relevancy": row["answer_relevancy"],
                "faithfulness_probability": row["faithfulness_probability"],
                "answer_relevancy_probability": row["answer_relevancy_probability"],
                "judge_model": row.get("judge_returned"),
                "judge_prompt_hash": row.get("judge_prompt_hash"),
            })
    output = Path(output)
    summary = Path(summary)
    for path in (output, summary):
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary.write_text(json.dumps({"conditions": summary_rows}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("revision/results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    merge(args.results_dir, args.output, args.summary)


if __name__ == "__main__":
    main()
