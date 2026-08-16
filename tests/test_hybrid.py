from __future__ import annotations

import pytest

from src.retrieval import hybrid
from src.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion

K = 60


class TestReciprocalRankFusion:
    def test_hand_computed_ordering(self) -> None:
        # list1: a@1, b@2, c@3 ; list2: b@1, d@2, a@3
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])
        assert fused == pytest.approx(
            {
                "a": 1 / (K + 1) + 1 / (K + 3),
                "b": 1 / (K + 2) + 1 / (K + 1),
                "c": 1 / (K + 3),
                "d": 1 / (K + 2),
            }
        )
        assert sorted(fused, key=fused.get, reverse=True) == ["b", "a", "d", "c"]

    def test_single_list(self) -> None:
        fused = reciprocal_rank_fusion([["x", "y"]])
        assert fused == {"x": 1 / (K + 1), "y": 1 / (K + 2)}

    def test_empty_lists(self) -> None:
        assert reciprocal_rank_fusion([]) == {}
        assert reciprocal_rank_fusion([[], []]) == {}

    def test_custom_k(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b"]], k=10)
        assert fused == {"a": 1 / 11, "b": 1 / 12}


class TestHybridSearchFusion:
    def test_dedupes_and_sorts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw = [{"chunk_id": "k1", "text": "kw1"}, {"chunk_id": "k2", "text": "kw2"}]
        vec = [{"chunk_id": "k2", "text": "vec2"}, {"chunk_id": "k3", "text": "vec3"}]
        monkeypatch.setattr(hybrid, "keyword_search", lambda q, l: kw)
        monkeypatch.setattr(hybrid, "vector_search", lambda q, l: vec)

        results = hybrid_search("test query", top_k=3, candidates_per_method=2)
        ids = [r["chunk_id"] for r in results]
        assert ids == ["k2", "k1", "k3"]  # k2 overlaps both -> highest fused score
        assert all("rrf_score" in r for r in results)
        assert results[0]["rrf_score"] >= results[1]["rrf_score"] >= results[2]["rrf_score"]
        assert results[0]["rrf_score"] == pytest.approx(1 / (K + 2) + 1 / (K + 1))

    def test_top_k_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw = [{"chunk_id": f"k{i}", "text": "x"} for i in range(10)]
        monkeypatch.setattr(hybrid, "keyword_search", lambda q, l: kw)
        monkeypatch.setattr(hybrid, "vector_search", lambda q, l: [])

        results = hybrid_search("q", top_k=3, candidates_per_method=10)
        assert len(results) == 3

    def test_retrieved_by_tracks_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw = [{"chunk_id": "both", "text": "x"}, {"chunk_id": "kw_only", "text": "y"}]
        vec = [{"chunk_id": "both", "text": "z"}, {"chunk_id": "vec_only", "text": "w"}]
        monkeypatch.setattr(hybrid, "keyword_search", lambda q, l: kw)
        monkeypatch.setattr(hybrid, "vector_search", lambda q, l: vec)

        results = hybrid_search("q", top_k=3, candidates_per_method=2)
        by_id = {r["chunk_id"]: r["retrieved_by"] for r in results}
        assert by_id["both"] == ["keyword", "vector"]
        assert by_id["kw_only"] == ["keyword"]
        assert by_id["vec_only"] == ["vector"]


def _atlas_available() -> bool:
    try:
        from src.config import settings

        return bool(settings.MONGODB_URI)
    except Exception:
        return False


@pytest.mark.slow
class TestHybridSearchLive:
    def test_runs_against_live_db(self) -> None:
        if not _atlas_available():
            pytest.skip("no Atlas credentials available")
        results = hybrid_search("atlas vector search", top_k=5, candidates_per_method=10)
        assert results, "expected at least one result from the live 4-chunk DB"
        assert all("rrf_score" in r for r in results)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)