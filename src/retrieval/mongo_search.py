"""Keyword (BM25) search over the chunks collection using Atlas Search."""

from __future__ import annotations

import logging
from typing import Any

from pymongo.database import Database

from src.db.client import get_db

logger = logging.getLogger(__name__)

_SEARCH_INDEX_NAME = "default"


def keyword_search(
    query: str,
    limit: int = 10,
    db: Database | None = None,
) -> list[dict[str, Any]]:
    """Run a full-text BM25 search and return matching chunks.

    Args:
        query: Free-text search query.
        limit: Maximum number of results to return.
        db: Optional database handle; defaults to the app database.

    Returns:
        A list of documents with ``_id``, ``chunk_id``, ``text``,
        ``parent_doc_id`` and ``score`` (``$meta: "searchScore"``).
    """
    database = db if db is not None else get_db()
    pipeline: list[dict[str, Any]] = [
        {
            "$search": {
                "index": _SEARCH_INDEX_NAME,
                "text": {"query": query, "path": "text"},
            }
        },
        {"$limit": limit},
        {
            "$project": {
                "_id": 1,
                "chunk_id": 1,
                "text": 1,
                "parent_doc_id": 1,
                "score": {"$meta": "searchScore"},
            }
        },
    ]
    logger.info("Keyword search query=%r limit=%d", query, limit)
    return list(database["chunks"].aggregate(pipeline))
