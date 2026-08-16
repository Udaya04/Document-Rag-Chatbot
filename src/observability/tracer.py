"""Per-query decision-path tracing into the Mongo `traces` collection.

Each answer_query() invocation produces one trace document capturing the
retrieval / rerank / confidence / generation / citations decision path, token
usage, timing, and any failure. The traces collection is the metrics source of
truth; there is no separate metrics module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection

from src.db.client import get_collection

TRACE_TTL_SECONDS = 7 * 24 * 3600

_COLLECTION_NAME = "traces"

_ttl_index_ensured = False


def _ensure_trace_ttl_index(collection: Collection[dict[str, Any]]) -> None:
    """Idempotently create a TTL index on ``started_at``, once per process.

    The create_index call is guarded by a module-level flag so it does not run
    on every trace write. NOTE: as with the cache collection, changing
    TRACE_TTL_SECONDS later requires dropping the old index first.
    """
    global _ttl_index_ensured
    if _ttl_index_ensured:
        return
    collection.create_index(
        "started_at",
        expireAfterSeconds=TRACE_TTL_SECONDS,
        name="started_at_ttl",
    )
    _ttl_index_ensured = True


def start_trace(query: str) -> dict[str, Any]:
    """Return a new in-progress trace for ``query``."""
    return {
        "query": query,
        "started_at": datetime.now(timezone.utc),
        "stages": {},
    }


def record_stage(trace: dict[str, Any], stage: str, **data: Any) -> None:
    """Record ``data`` under ``stage`` in ``trace`` (mutates in place)."""
    trace["stages"][stage] = data


def finalize_trace(
    trace: dict[str, Any],
    success: bool,
    error: str | None = None,
) -> None:
    """Compute duration, set success/error, and persist the trace to Mongo."""
    started_at = trace["started_at"]
    duration_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000.0
    trace["duration_ms"] = duration_ms
    trace["success"] = success
    trace["error"] = error

    collection = get_collection(_COLLECTION_NAME)
    _ensure_trace_ttl_index(collection)
    collection.insert_one(trace)
