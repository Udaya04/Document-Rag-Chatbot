"""Hybrid retrieval: reciprocal rank fusion of keyword and vector search."""

from __future__ import annotations

import logging
from typing import Any

from src.retrieval.mongo_search import keyword_search
from src.retrieval.mongo_vector import vector_search

logger = logging.getLogger(__name__)

_RRF_CONSTANT = 60


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[str]],
    k: int = _RRF_CONSTANT,
) -> dict[str, float]:
    """Fuse multiple ranked lists into a single scored mapping by rank position.

    Each item at rank ``rank`` (1-based) in a list contributes ``1 / (k + rank)``;
    an item's fused score is the sum of its contributions across all lists.
    RRF deliberately ignores raw score magnitude and uses only rank order, which
    is why it can combine BM25 relevance scores and cosine similarity scores even
    though those two systems score on completely different, incomparable scales.

    Args:
        ranked_id_lists: Lists of item ids, each ordered best-first.
        k: Smoothing constant; larger values dampen the rank contribution.

    Returns:
        Mapping of item id to fused score. Ids absent from all lists are absent.
    """
    fused_scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused_scores


def hybrid_search(
    query: str,
    top_k: int = 10,
    candidates_per_method: int = 20,
) -> list[dict[str, Any]]:
    """Run keyword and vector search, fuse with RRF, and return top chunks.

    Args:
        query: Free-text search query.
        top_k: Number of fused results to return.
        candidates_per_method: How many results to fetch from each search method
            before fusion (should be >= top_k).

    Returns:
        Deduplicated chunk dicts sorted by fused RRF score descending, each with
        the fused score attached as ``rrf_score`` and ``retrieved_by`` (the list
        of retrieval methods that surfaced the chunk: ``"keyword"``,
        ``"vector"``, or both). Note ``vector_search`` embeds the query
        internally, so no separate embedding call is needed here.
    """
    keyword_results = keyword_search(query, candidates_per_method)
    vector_results = vector_search(query, candidates_per_method)
    logger.info(
        "Hybrid search query=%r keyword=%d vector=%d",
        query,
        len(keyword_results),
        len(vector_results),
    )

    ranked_id_lists = [
        [result["chunk_id"] for result in keyword_results if result.get("chunk_id")],
        [result["chunk_id"] for result in vector_results if result.get("chunk_id")],
    ]
    fused_scores = reciprocal_rank_fusion(ranked_id_lists)

    merged: dict[str, dict[str, Any]] = {}
    for result in [*keyword_results, *vector_results]:
        chunk_id = result.get("chunk_id")
        if chunk_id is not None and chunk_id not in merged:
            merged[chunk_id] = dict(result)

    keyword_ids = {result["chunk_id"] for result in keyword_results if result.get("chunk_id")}
    vector_ids = {result["chunk_id"] for result in vector_results if result.get("chunk_id")}
    for chunk_id, result in merged.items():
        retrieved_by: list[str] = []
        if chunk_id in keyword_ids:
            retrieved_by.append("keyword")
        if chunk_id in vector_ids:
            retrieved_by.append("vector")
        result["retrieved_by"] = retrieved_by

    fused_results: list[dict[str, Any]] = []
    for chunk_id, result in merged.items():
        if chunk_id in fused_scores:
            result["rrf_score"] = fused_scores[chunk_id]
            fused_results.append(result)

    fused_results.sort(key=lambda result: result["rrf_score"], reverse=True)
    return fused_results[:top_k]