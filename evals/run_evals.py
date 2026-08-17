"""Run all evals and write one consolidated report.

Sections:
1. recall@k benchmark (recall_benchmark.py)
2. hallucination check (hallucination_check.py) - LLM-judge estimate
3. gate validation on evals/adversarial_queries.jsonl (inline here)

Report is written to evals/results/report_<timestamp>.md (gitignored) and a
short summary is printed to the console. No pass/fail thresholds are
hard-coded - this is the first real measurement at this corpus size.

Run:  python evals/run_evals.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.pipeline import INSUFFICIENT_EVIDENCE_MESSAGE, answer_query

from evals.hallucination_check import run_hallucination_check
from evals.recall_benchmark import run_recall_benchmark

ADVERSARIAL_PATH = Path(__file__).resolve().parent / "adversarial_queries.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_adversarial_queries() -> list[str]:
    return [
        json.loads(line)["query"]
        for line in ADVERSARIAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_gate_validation() -> dict:
    """Answer each adversarial query end-to-end; count gate triggers."""
    queries = _load_adversarial_queries()
    caught: list[str] = []
    missed: list[dict] = []
    for query in queries:
        result = answer_query(query)
        if result["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE:
            caught.append(query)
            print(f"GATE: {query}")
        else:
            missed.append({"query": query, "answer": result["answer"]})
            print(f"MISS: {query}")

    rate = len(caught) / len(queries) if queries else 0.0
    print(f"gate trigger rate: {round(rate, 4)} ({len(caught)}/{len(queries)} caught)")
    return {"total": len(queries), "caught": caught, "missed": missed, "rate": rate}


def main() -> None:
    recall = run_recall_benchmark()
    hallucination = run_hallucination_check()
    gate = run_gate_validation()

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"report_{now}.md"

    lines: list[str] = []
    lines.append("# RAG Evaluation Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Eval corpus: {recall['total']} wiki-derived queries")
    lines.append("")

    lines.append("## 1. Recall@k")
    lines.append("")
    lines.append("| k | recall@k | hits |")
    lines.append("|---|---|---|")
    for k in sorted(recall["recall"]):
        lines.append(
            f"| {k} | {recall['recall'][k]:.4f} | {recall['hits'][k]} |"
        )
    lines.append("")

    lines.append("## 2. Hallucination check")
    lines.append("")
    lines.append(
        "> **LLM-judge estimate, not ground truth.** The judge is itself an "
        "LLM and can be wrong; treat the rate as an approximation."
    )
    lines.append("")
    rate = hallucination["rate"]
    lines.append(
        f"- **Rate: {rate if rate is None else round(rate, 4)}** "
        f"({hallucination['flagged']} flagged / {hallucination['judged']} judged; "
        f"{hallucination['skipped']} gate-skipped, not counted)"
    )
    lines.append("")
    for result in hallucination["results"]:
        status = "CLEAN" if result["clean"] else "FLAGGED"
        lines.append(f"### {status}: {result['query']}")
        lines.append("")
        lines.append(f"**Answer:** {result['answer']}")
        lines.append("")
        lines.append(f"**Judge output:** {result['verdict']}")
        lines.append("")
    lines.append("")

    lines.append("## 3. Gate validation (adversarial queries)")
    lines.append("")
    lines.append(
        f"- **Gate trigger rate: {round(gate['rate'], 4)}** "
        f"({len(gate['caught'])}/{gate['total']} correctly gated)"
    )
    lines.append("")
    if gate["missed"]:
        lines.append("### Queries that slipped through the gate")
        lines.append("")
        for item in gate["missed"]:
            lines.append(f"- {item['query']}")
            lines.append(f"  - Answer snippet: {item['answer'][:180]}")
        lines.append("")
    else:
        lines.append("No adversarial queries slipped through the gate.")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=== EVAL SUMMARY ===")
    print(
        "Recall@1/3/5: "
        + " / ".join(f"{recall['recall'][k]:.3f}" for k in (1, 3, 5))
    )
    print(f"Hallucination rate (LLM-judge estimate): {round(rate, 4) if rate is not None else 'n/a'}")
    print(f"Gate trigger rate: {round(gate['rate'], 4)} ({len(gate['caught'])}/{gate['total']})")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()