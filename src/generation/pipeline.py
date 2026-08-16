"""Full RAG orchestration: retrieval -> rerank -> confidence gate -> generation."""

from __future__ import annotations

from typing import Any

from src.caching.query_cache import get_cached_answer, set_cached_answer
from src.generation.citation_mapper import map_citations
from src.generation.constrained_generator import generate_answer
from src.observability.logger import logger
from src.observability.tracer import finalize_trace, record_stage, start_trace
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import get_reranker
from src.scoring.confidence import score_and_filter

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient evidence found to answer this question."


def _score_snapshot(chunks: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    """Lean per-chunk trace data (chunk_id + score only, never full text)."""
    return [
        {"chunk_id": c.get("chunk_id"), "score": c.get(score_key)}
        for c in chunks
    ]


def answer_query(query: str) -> dict[str, Any]:
    """Answer ``query`` grounded in retrieved context, gated by confidence.

    Each stage uses its own existing defaults (no tunable parameters here).
    When no chunk clears the confidence threshold the LLM is never called and
    a fixed insufficient-evidence answer is returned. Identical queries that
    were answered before are served straight from the cache. The decision path
    is traced and logged as a side effect; the returned shape is unchanged.
    """
    trace = start_trace(query)
    try:
        cached = get_cached_answer(query)
        record_stage(trace, "cache", hit=cached is not None)
        if cached is not None:
            logger.info("Cache hit for query={!r}", query)
            finalize_trace(trace, success=True)
            return cached

        logger.info("Cache miss for query={!r}", query)

        results = hybrid_search(query)
        record_stage(trace, "retrieval", chunks=_score_snapshot(results, "rrf_score"))

        reranked = get_reranker().rerank(query, results)
        record_stage(trace, "rerank", chunks=_score_snapshot(reranked, "rerank_score"))

        scored = score_and_filter(reranked)
        record_stage(trace, "confidence", chunks=_score_snapshot(scored, "confidence_score"))

        if not scored:
            logger.info("Confidence gate triggered for query={!r}", query)
            result = {
                "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
                "citations": [],
                "context_chunks": [],
            }
            set_cached_answer(query, result)
            record_stage(trace, "generation", skipped=True, reason="insufficient_evidence")
            record_stage(trace, "citations", citations=[])
            finalize_trace(trace, success=True)
            return result

        generated = generate_answer(query, scored)
        logger.info("LLM called for query={!r}", query)
        record_stage(trace, "generation", token_usage=generated.get("token_usage"))

        mapped = map_citations(generated["answer"], scored)
        record_stage(trace, "citations", citations=mapped["citations"])

        result = {
            "answer": mapped["answer"],
            "citations": mapped["citations"],
            "context_chunks": scored,
        }
        set_cached_answer(query, result)
        finalize_trace(trace, success=True)
        return result
    except Exception as e:
        logger.error("answer_query failed for query={!r}: {}", query, e)
        finalize_trace(trace, success=False, error=str(e))
        raise