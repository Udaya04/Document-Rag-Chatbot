"""LLM-judge hallucination estimate over a fixed subset of eval_set.jsonl.

Picks a FIXED RANDOM sample of 15 eval queries (seed 42, documented choice)
and runs each through the full ``answer_query`` pipeline. Results gated to
INSUFFICIENT_EVIDENCE_MESSAGE are skipped (nothing to judge). Each real answer
is fact-checked by an LLM judge given ONLY the text of the cited chunks; the
judge lists unsupported claims or responds exactly "NONE". The reported rate
is an LLM-judge ESTIMATE, not ground truth.

Run:  python evals/hallucination_check.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.client import get_db
from src.generation import llm_client
from src.generation.pipeline import INSUFFICIENT_EVIDENCE_MESSAGE, answer_query

logger = logging.getLogger(__name__)

SUBSET_SIZE = 15
SEED = 42

JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checker. You will be given a CONTEXT (verbatim "
    "retrieved passages) and an ANSWER produced by an AI assistant. "
    "List every factual claim in the ANSWER that is NOT supported by the "
    "CONTEXT. If every claim in the ANSWER is supported by the CONTEXT, "
    "respond with exactly the single word NONE. Do not mention claims that "
    "are supported. Return only the list of unsupported claims, or the word "
    "NONE."
)


def _load_entries() -> list[dict]:
    path = Path(__file__).resolve().parent / "eval_set.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _chunk_text(db, chunk_id: str) -> str:
    doc = db["chunks"].find_one({"chunk_id": chunk_id}, {"text": 1})
    return doc.get("text", "") if doc else ""


def _judge_answer(
    db, answer: str, citations: list[dict], context_chunks: list[dict] | None = None
) -> tuple[str, bool]:
    context_parts = [
        _chunk_text(db, citation.get("chunk_id"))
        for citation in citations
        if citation.get("chunk_id")
    ]
    if not context_parts:
        logger.warning(
            "citation extraction returned empty; falling back to context_chunks"
        )
        for chunk in context_chunks or []:
            text = chunk.get("text", "") if isinstance(chunk, dict) else ""
            if text:
                context_parts.append(text)
    context = "\n\n".join(part.strip() for part in context_parts if part.strip())
    if not context:
        context = "(no cited chunks available)"
    user_prompt = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    verdict = llm_client.generate(JUDGE_SYSTEM_PROMPT, user_prompt)["text"].strip()
    is_clean = verdict.replace(".", "").strip().upper() == "NONE"
    return verdict, is_clean


def run_hallucination_check(limit: int | None = None) -> dict:
    """Run the check and return {rate, judged, skipped, results}."""
    db = get_db()
    entries = _load_entries()
    subset = random.Random(SEED).sample(entries, min(SUBSET_SIZE, len(entries)))
    if limit:
        subset = subset[:limit]

    results: list[dict] = []
    skipped = 0
    for entry in subset:
        query = entry["query"]
        result = answer_query(query)
        if result["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE:
            skipped += 1
            print(f"SKIP (gate triggered): {query}")
            continue
        verdict, is_clean = _judge_answer(
            db, result["answer"], result.get("citations", []), result.get("context_chunks", [])
        )
        results.append(
            {
                "query": query,
                "answer": result["answer"],
                "verdict": verdict,
                "clean": is_clean,
            }
        )
        print(f"{'CLEAN' if is_clean else 'FLAG'}: {query}")

    flagged = sum(1 for r in results if not r["clean"])
    rate = flagged / len(results) if results else None

    print(
        f"hallucination rate (LLM-judge estimate): "
        f"{round(rate, 4) if rate is not None else 'n/a'} "
        f"({flagged} flagged / {len(results)} judged, {skipped} gate-skipped)"
    )
    return {"rate": rate, "flagged": flagged, "judged": len(results), "skipped": skipped, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_hallucination_check(args.limit)


if __name__ == "__main__":
    main()