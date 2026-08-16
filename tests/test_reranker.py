from __future__ import annotations

import pytest

from src.retrieval.reranker import Reranker, get_reranker


@pytest.fixture(autouse=True)
def _reset_reranker() -> None:
    Reranker._instance = None
    yield
    Reranker._instance = None


@pytest.mark.slow
class TestReranker:
    def test_singleton(self) -> None:
        assert get_reranker() is get_reranker()

    def test_rerank_sorts_scores_and_does_not_mutate(self) -> None:
        reranker = Reranker()
        query = "What is MongoDB Atlas?"
        candidates = [
            {"chunk_id": "a", "text": "The weather in Berlin is rainy today."},
            {"chunk_id": "b", "text": "MongoDB Atlas is a fully managed cloud database service."},
            {"chunk_id": "c", "text": "Atlas Vector Search supports semantic search over embeddings."},
        ]
        before = [dict(c) for c in candidates]

        results = reranker.rerank(query, candidates, top_k=2)

        assert len(results) == 2
        assert all("rerank_score" in r for r in results)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert [c["chunk_id"] for c in candidates] == ["a", "b", "c"]
        assert all("rerank_score" not in c for c in candidates)
        assert candidates == before

    def test_rerank_empty(self) -> None:
        assert Reranker().rerank("q", []) == []