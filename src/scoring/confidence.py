"""Source-confidence scoring for retrieved chunks.

Confidence is query-time only: it combines per-source trust, freshness, and the
overlap of retrieval methods that surfaced the chunk. It is never persisted
back onto chunk documents in Mongo.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

from src.db.client import get_db
from src.scoring import config

logger = logging.getLogger(__name__)

_MISSING_DATE_DEFAULT = 0.3


def freshness_score(
    date: str | datetime | None,
    decay_days: float = config.FRESHNESS_DECAY_DAYS,
) -> float:
    """Score recency on an exponential decay curve, clamped to [0, 1].

    A date that cannot be parsed (or is absent) yields a conservative default.
    Note: the pipeline has no publication-date field, so callers currently pass
    no date and freshness is the conservative default.
    """
    if date is None:
        return _MISSING_DATE_DEFAULT
    try:
        if isinstance(date, str):
            parsed = datetime.fromisoformat(date)
        else:
            parsed = date
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - parsed).total_seconds() / 86400.0
    except (ValueError, TypeError):
        logger.debug("Unparseable date %r; using conservative default", date)
        return _MISSING_DATE_DEFAULT
    if age_days < 0.0:
        return 1.0
    return math.exp(-age_days / decay_days)


def trust_score(source: str | None) -> float:
    """Score source trust via substring overrides, falling back to the default."""
    if source is None:
        return config.DEFAULT_TRUST_SCORE
    for needle, override in config.TRUST_OVERRIDES.items():
        if needle and needle in source:
            return override
    return config.DEFAULT_TRUST_SCORE


def overlap_score(retrieved_by: list[str] | None) -> float:
    """Score retrieval-method agreement: both -> 1.0, one -> 0.5, none -> 0.0."""
    if not retrieved_by:
        return 0.0
    return len(set(retrieved_by)) / 2.0


def relevance_score(rerank_score: float | None) -> float:
    """Sigmoid-map a cross-encoder logit to a relevance score in (0, 1).

    A missing score (chunk never reranked) yields the neutral 0.5. The sigmoid
    is overflow-safe for very negative inputs by branching on the sign of x
    instead of computing 1 / (1 + exp(-x)) directly; never raises or overflows.
    """
    if rerank_score is None:
        return 0.5
    if rerank_score >= 0:
        return 1.0 / (1.0 + math.exp(-rerank_score))
    exp_x = math.exp(rerank_score)
    return exp_x / (1.0 + exp_x)


def score_chunk(
    chunk: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float:
    """Combine the four sub-scores into one confidence score, clamped to [0, 1].

    Pure: reads ``date``/``source``/``retrieved_by``/``rerank_score`` only
    from the chunk dict.
    """
    weights = weights or config.CONFIDENCE_WEIGHTS
    freshness = freshness_score(chunk.get("date"))
    trust = trust_score(chunk.get("source"))
    overlap = overlap_score(chunk.get("retrieved_by"))
    relevance = relevance_score(chunk.get("rerank_score"))
    confidence = (
        weights["freshness"] * freshness
        + weights["trust"] * trust
        + weights["overlap"] * overlap
        + weights["relevance"] * relevance
    )
    return max(0.0, min(1.0, confidence))


def score_and_filter(
    chunks: list[dict[str, Any]],
    threshold: float = config.CONFIDENCE_THRESHOLD,
    weights: dict[str, float] | None = None,
    db: Any = None,
) -> list[dict[str, Any]]:
    """Attach ``confidence_score`` to each chunk, keep those >= threshold, sort desc.

    Input chunks are not mutated. Chunks without a direct ``source`` field are
    resolved in ONE batched lookup against the ``docs`` collection (keyed by
    ``_id`` == ``parent_doc_id``); no per-chunk queries. An empty result is
    valid output and is not an error.
    """
    weights = weights or config.CONFIDENCE_WEIGHTS
    if not chunks:
        return []

    missing_source = [c for c in chunks if not c.get("source")]
    source_by_doc: dict[str, str] = {}
    if missing_source:
        doc_ids = {
            c.get("parent_doc_id")
            for c in missing_source
            if c.get("parent_doc_id") is not None
        }
        if doc_ids:
            if db is None:
                db = get_db()
            docs = db.docs.find(
                {"_id": {"$in": list(doc_ids)}},
                {"metadata.source_file": 1},
            )
            for doc in docs:
                source_file = (doc.get("metadata") or {}).get("source_file")
                if source_file is not None:
                    source_by_doc[doc["_id"]] = source_file

    scored: list[dict[str, Any]] = []
    for chunk in chunks:
        enriched = dict(chunk)
        if not enriched.get("source"):
            source = source_by_doc.get(enriched.get("parent_doc_id"))
            if source is not None:
                enriched["source"] = source
        enriched["confidence_score"] = score_chunk(enriched, weights)
        if enriched["confidence_score"] >= threshold:
            scored.append(enriched)

    scored.sort(key=lambda c: c["confidence_score"], reverse=True)
    return scored