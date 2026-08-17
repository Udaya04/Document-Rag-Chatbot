"""Recall@k benchmark over evals/eval_set.jsonl.

For each eval query the real retrieval path is exercised exactly as a user
would see it: ``hybrid_search`` then ``rerank`` (the final pre-gate list).
recall@k = fraction of eval queries whose source doc appears in the top-k
reranked chunks (matched by parent_doc_id == source_doc_id).

Run:  python evals/recall_benchmark.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import get_reranker

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.jsonl"


def _load_entries(limit: int | None) -> list[dict]:
    entries = [
        json.loads(line)
        for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return entries[:limit] if limit else entries


def run_recall_benchmark(limit: int | None = None) -> dict:
    """Run the benchmark and return {total, hits, recall}."""
    entries = _load_entries(limit)
    reranker = get_reranker()
    hits = {1: 0, 3: 0, 5: 0}

    for entry in entries:
        query = entry["query"]
        target = entry["source_doc_id"]
        reranked = reranker.rerank(query, hybrid_search(query))
        for k in hits:
            if any(
                chunk.get("parent_doc_id") == target
                for chunk in reranked[:k]
            ):
                hits[k] += 1

    total = len(entries)
    recall = {k: hits[k] / total if total else 0.0 for k in hits}

    print(f"Recall@k over {total} eval queries")
    print(f"{'k':<6}{'recall':<10}{'hits'}")
    for k in sorted(recall):
        print(f"{k:<6}{recall[k]:<10.3f}{hits[k]}")

    return {"total": total, "hits": hits, "recall": recall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_recall_benchmark(args.limit)


if __name__ == "__main__":
    main()