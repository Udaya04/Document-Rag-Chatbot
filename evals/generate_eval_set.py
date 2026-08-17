"""Generate evals/eval_set.jsonl: 30 synthetic ground-truth Q/A pairs.

Samples 30 distinct ``wiki_`` doc ids with a fixed seed, fetches the longest
chunk per doc, and asks an LLM judge to write ONE specific factual question
answerable only from that passage. Idempotent: an existing eval_set.jsonl
with >= 30 entries is reused unless ``--force`` is passed.

Run:  python evals/generate_eval_set.py [--force]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.client import get_db
from src.generation import llm_client

TARGET = 30
SEED = 42
EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.jsonl"

QUESTION_SYSTEM_PROMPT = (
    "You are generating questions for a retrieval evaluation. "
    "Write ONE specific factual question that is answerable only from the "
    "given passage. Return only the question text, with no preamble, no "
    "numbering, and no quotation marks."
)


def _load_wiki_doc_ids(db, count: int) -> list[str]:
    doc_ids = [
        doc["_id"]
        for doc in db["docs"].find(
            {"metadata.source_file": {"$regex": "wiki_"}},
            {"_id": 1},
        )
    ]
    if len(doc_ids) < count:
        raise RuntimeError(f"only {len(doc_ids)} wiki docs found; need {count}")
    return random.Random(SEED).sample(doc_ids, count)


def _longest_chunk(db, doc_id: str) -> dict | None:
    chunks = list(
        db["chunks"].find(
            {"parent_doc_id": doc_id},
            {"text": 1, "chunk_id": 1},
        )
    )
    if not chunks:
        return None
    return max(chunks, key=lambda chunk: len(chunk.get("text", "")))


def _title_from_text(text: str) -> str:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.lower().startswith("title:"):
        return first_line[len("title:"):].strip()
    return ""


def _ask_question(passage: str) -> str:
    response = llm_client.generate(QUESTION_SYSTEM_PROMPT, passage)
    return response["text"].strip().strip('"').strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate eval_set.jsonl even if it already has >= 30 entries",
    )
    args = parser.parse_args()

    if not args.force and EVAL_SET_PATH.exists():
        existing = [
            line.strip()
            for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(existing) >= TARGET:
            print(
                f"{EVAL_SET_PATH.name} already has {len(existing)} entries; "
                f"skipping (use --force to regenerate)"
            )
            return

    db = get_db()
    doc_ids = _load_wiki_doc_ids(db, TARGET)

    entries: list[dict[str, str]] = []
    for doc_id in doc_ids:
        chunk = _longest_chunk(db, doc_id)
        if chunk is None:
            print(f"WARNING: no chunks for doc {doc_id}; skipping")
            continue
        title = _title_from_text(chunk.get("text", ""))
        query = _ask_question(chunk.get("text", ""))
        entries.append(
            {
                "query": query,
                "source_doc_id": doc_id,
                "source_title": title,
            }
        )
        print(f"[{len(entries)}/{TARGET}] {title or doc_id}: {query}")

    EVAL_SET_PATH.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    print(f"wrote {len(entries)} entries to {EVAL_SET_PATH}")


if __name__ == "__main__":
    main()