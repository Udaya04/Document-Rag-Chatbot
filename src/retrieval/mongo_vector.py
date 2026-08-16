"""Approximate-nearest-neighbour (ANN) vector search over the chunks collection."""

from __future__ import annotations

import logging
from typing import Any

from pymongo.database import Database

from src.db.client import get_db
from src.retrieval.embedder import get_embedder

logger = logging.getLogger(__name__)

_VECTOR_INDEX_NAME = "vector_index"


def vector_search(
    query: str,
    limit: int = 10,
    db: Database | None = None,
) -> list[dict[str, Any]]:
    """Embed the query and run Atlas Vector Search over the chunk embeddings.

    Args:
        query: Text to find semantically similar chunks for.
        limit: Maximum number of results to return.
        db: Optional database handle; defaults to the app database.

    Returns:
        A list of documents with ``_id``, ``chunk_id``, ``text``,
        ``parent_doc_id`` and ``score`` (``$meta: "vectorSearchScore"``).
    """
    database = db if db is not None else get_db()
    query_vector = get_embedder().embed_text(query)

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": _VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": limit * 20,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 1,
                "chunk_id": 1,
                "text": 1,
                "parent_doc_id": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    logger.info("Vector search query=%r limit=%d", query, limit)
    return list(database["chunks"].aggregate(pipeline))
