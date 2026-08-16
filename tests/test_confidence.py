from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import get_reranker
from src.scoring import config, confidence
from src.scoring.confidence import (
    freshness_score,
    overlap_score,
    relevance_score,
    score_and_filter,
    score_chunk,
    trust_score,
)


def _atlas_available() -> bool:
    try:
        from src.config import settings

        return bool(settings.MONGODB_URI)
    except Exception:
        return False


class TestFreshnessScore:
    def test_recent_date_scores_high(self) -> None:
        now = datetime.now(UTC)
        assert freshness_score(now.isoformat()) == pytest.approx(1.0, abs=1e-3)

    def test_very_old_date_scores_low(self) -> None:
        assert freshness_score("2015-01-01") < 0.05

    def test_missing_or_invalid_uses_conservative_default(self) -> None:
        assert freshness_score(None) == 0.3
        assert freshness_score("not-a-date") == 0.3

    def test_future_date_clamped_to_one(self) -> None:
        assert freshness_score("2999-01-01") == 1.0


class TestTrustScore:
    def test_default_when_no_override(self) -> None:
        assert trust_score("any-source.pdf") == config.DEFAULT_TRUST_SCORE

    def test_none_source_uses_default(self) -> None:
        assert trust_score(None) == config.DEFAULT_TRUST_SCORE

    def test_substring_override_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "TRUST_OVERRIDES", {"unreliable": 0.2})
        assert trust_score("unreliable-blog.com") == 0.2
        assert trust_score("official-docs.gov") == config.DEFAULT_TRUST_SCORE

    def test_wiki_source_uses_override(self) -> None:
        assert trust_score("data\\raw\\wiki_123.txt") == 0.8


class TestOverlapScore:
    def test_both_one_none(self) -> None:
        assert overlap_score(["keyword", "vector"]) == 1.0
        assert overlap_score(["keyword"]) == 0.5
        assert overlap_score(["vector"]) == 0.5
        assert overlap_score([]) == 0.0
        assert overlap_score(None) == 0.0


class TestRelevanceScore:
    def test_zero_is_neutral(self) -> None:
        assert relevance_score(0) == 0.5

    def test_none_is_neutral(self) -> None:
        assert relevance_score(None) == 0.5

    def test_large_positive(self) -> None:
        assert relevance_score(10) > 0.99

    def test_large_negative_no_overflow(self) -> None:
        assert relevance_score(-10) < 0.01

    def test_never_out_of_unit_interval(self) -> None:
        import math

        for x in (None, -1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0):
            score = relevance_score(x)
            assert not math.isnan(score)
            assert 0.0 <= score <= 1.0


class TestScoreChunk:
    def test_hand_computed(self) -> None:
        chunk = {
            "source": "some-source",
            "retrieved_by": ["keyword", "vector"],
            "rerank_score": 0,
        }
        # freshness 0.3 (no date), trust 0.6, overlap 1.0, relevance 0.5
        # -> (0.3 + 0.6 + 1.0 + 0.5) / 4 = 0.6 exactly
        assert score_chunk(chunk) == pytest.approx(0.6, abs=1e-9)

    def test_clamped_to_unit_interval(self) -> None:
        chunk = {
            "date": datetime.now(UTC).isoformat(),
            "source": "x",
            "retrieved_by": ["keyword", "vector"],
            "rerank_score": 0,
        }
        assert 0.0 <= score_chunk(chunk) <= 1.0
        # empty chunk: freshness 0.3, trust 0.6, overlap 0.0, relevance 0.5
        assert score_chunk({}) == pytest.approx(0.35, abs=1e-9)


class TestScoreAndFilter:
    def test_fabricated_below_and_above_threshold(self) -> None:
        now = datetime.now(UTC).isoformat()
        chunks = [
            {
                "chunk_id": "A",
                "parent_doc_id": "p1",
                "date": now,
                "source": "official.pdf",
                "retrieved_by": ["keyword", "vector"],
                "rerank_score": 8,
            },
            {
                "chunk_id": "B",
                "parent_doc_id": "p2",
                "date": "2018-01-01",
                "source": "unreliable-site",
                "retrieved_by": ["keyword"],
                "rerank_score": -8,
            },
            {
                "chunk_id": "C",
                "parent_doc_id": "p3",
                "date": "2015-01-01",
                "source": "whatever",
                "retrieved_by": ["vector"],
                "rerank_score": -8,
            },
            {
                "chunk_id": "D",
                "parent_doc_id": "p4",
                "date": now,
                "source": "docs-a",
                "retrieved_by": ["vector"],
                "rerank_score": 2,
            },
            {
                "chunk_id": "E",
                "parent_doc_id": "p5",
                "date": now,
                "source": "docs-b",
                "retrieved_by": ["keyword"],
                "rerank_score": 2,
            },
        ]

        survivors = score_and_filter(chunks)

        assert [c["chunk_id"] for c in survivors] == ["A", "D", "E"]
        assert all(c["confidence_score"] >= config.CONFIDENCE_THRESHOLD for c in survivors)
        scores = [c["confidence_score"] for c in survivors]
        assert scores == sorted(scores, reverse=True)

    def test_input_not_mutated(self) -> None:
        chunks = [
            {"chunk_id": "X", "source": "s", "retrieved_by": ["keyword"], "rerank_score": 0},
            {"chunk_id": "Y", "source": "s", "retrieved_by": ["vector"], "rerank_score": 0},
        ]
        before = [dict(c) for c in chunks]
        score_and_filter(chunks)
        assert all("confidence_score" not in c for c in chunks)
        assert chunks == before

    def test_empty_input_returns_empty(self) -> None:
        assert score_and_filter([]) == []


@pytest.mark.slow
class TestLiveSmoke:
    def test_hybrid_rerank_score_end_to_end(self) -> None:
        if not _atlas_available():
            pytest.skip("no Atlas credentials available")
        results = hybrid_search("fully managed cloud database", top_k=5, candidates_per_method=10)
        reranked = get_reranker().rerank("fully managed cloud database", results, top_k=5)
        scored = score_and_filter(reranked)
        assert all("confidence_score" in c for c in scored)
        scores = [c["confidence_score"] for c in scored]
        assert scores == sorted(scores, reverse=True)