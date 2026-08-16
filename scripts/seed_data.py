"""Seed data/raw/ with a CS/tech-focused subset of English Wikipedia.

Streams the HF "wikimedia/wikipedia" dataset (20231101.en), matches articles
by CS/tech keywords in the *title* (case-insensitive substring), and writes
each match as data/raw/wiki_{article_id}.txt. Uses the existing ingestion
pipeline afterwards; this script only produces raw text files.

Run a dry-run:  python scripts/seed_data.py --limit 100
Run the full seed: python scripts/seed_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.observability.logger import logger

DATASET_ID = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"

TARGET_COUNT = 20000
MAX_SCAN = 500_000
PROGRESS_EVERY = 1_000

RAW_DIR = Path("data/raw")

CS_KEYWORDS = [
    "algorithm",
    "programming language",
    "compiler",
    "operating system",
    "computer network",
    "database",
    "data structure",
    "software engineering",
    "machine learning",
    "artificial intelligence",
    "cryptography",
    "distributed system",
    "computer architecture",
    "cybersecurity",
    "web development",
    "computer vision",
    "natural language processing",
    "version control",
    "api",
    "microservice",
    "cloud computing",
    "computer science",
    "programming",
    "source code",
    "computer program",
    "information retrieval",
    "search engine",
    "computer graphics",
    "embedded system",
    "computer hardware",
    "internet protocol",
    "virtual machine",
    "container",
    "open source software",
    "linux",
    "python",
    "javascript",
    "java",
    "rust",
    "sql",
]

FINAL_KEYWORDS = [keyword.lower() for keyword in CS_KEYWORDS]


def _title_matches(title: str) -> bool:
    lowered = title.lower()
    return any(keyword in lowered for keyword in FINAL_KEYWORDS)


def _write_article(article_id: str, title: str, text: str) -> None:
    path = RAW_DIR / f"wiki_{article_id}.txt"
    path.write_text(f"Title: {title}\n\n{text}", encoding="utf-8")


def seed(limit: int | None, max_scan: int) -> tuple[int, int]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Seeding from {} config={} limit={} max_scan={}",
        DATASET_ID,
        DATASET_CONFIG,
        limit,
        max_scan,
    )
    logger.info("Keywords used ({}): {}", len(FINAL_KEYWORDS), FINAL_KEYWORDS)

    stream = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)
    matched = 0
    scanned = 0
    for row in stream:
        scanned += 1
        title = row["title"] or ""
        text = row["text"] or ""
        if _title_matches(title) and text.strip():
            _write_article(str(row["id"]), title, text)
            matched += 1
            if limit is not None and matched >= limit:
                break
        if scanned >= max_scan:
            break
        if scanned % PROGRESS_EVERY == 0:
            logger.info("Progress: scanned={} matched={}", scanned, matched)

    logger.info(
        "Seed complete: matched={} scanned={}",
        matched,
        scanned,
    )
    return matched, scanned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many matched articles (dry-run count, overrides TARGET_COUNT)",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=MAX_SCAN,
        help="Hard cap on articles examined regardless of match count",
    )
    args = parser.parse_args()
    seed(args.limit, args.max_scan)


if __name__ == "__main__":
    main()