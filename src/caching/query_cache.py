"""Query-result caching: full answer_query() outputs keyed by normalized query.

Both real answers and "insufficient evidence" results are cached, so repeated
identical queries skip retrieval, rerank, scoring, and the Groq LLM call.
Session memory and re-ingestion invalidation are intentionally out of scope.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from pymongo.collection import Collection

from src.db.client import get_collection

CACHE_TTL_SECONDS = 3600

_COLLECTION_NAME = "cache"


def get_cache_key(query: str) -> str:
    """Normalize ``query`` (lowercase, stripped) and return its sha256 digest."""
    normalized = query.lower().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _ensure_ttl_index(collection: Collection[dict[str, Any]]) -> None:
    """Idempotently create a TTL index on ``created_at`` for auto-expiry.

    NOTE: if CACHE_TTL_SECONDS is changed later, Mongo raises an index-conflict
    error on the next create_index call because the expireAfterSeconds value
    differs from the existing index. The old index must be dropped first (no
    automatic handling is built for this by design).
    """
    collection.create_index(
        "created_at",
        expireAfterSeconds=CACHE_TTL_SECONDS,
        name="created_at_ttl",
    )


def get_cached_answer(query: str) -> dict[str, Any] | None:
    """Return the cached result for ``query``, or None on a miss.

    Mongo's TTL background sweep runs on its own ~60s cycle, so a doc may
    occasionally still be found up to ~60s past nominal expiry; this is a
    known, acceptable imprecision -- no manual expiry-checking compensates.
    """
    collection = get_collection(_COLLECTION_NAME)
    _ensure_ttl_index(collection)
    doc = collection.find_one({"_id": get_cache_key(query)})
    if doc is None:
        return None
    return doc.get("result")


def set_cached_answer(query: str, result: dict[str, Any]) -> None:
    """Upsert the full ``answer_query`` result for ``query`` into the cache."""
    collection = get_collection(_COLLECTION_NAME)
    _ensure_ttl_index(collection)
    collection.replace_one(
        {"_id": get_cache_key(query)},
        {
            "_id": get_cache_key(query),
            "query": query,
            "result": result,
            "created_at": datetime.utcnow(),
        },
        upsert=True,
    )
