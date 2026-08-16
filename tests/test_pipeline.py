from __future__ import annotations

import pytest

from src.config import settings
from src.generation import llm_client
from src.generation import pipeline
from src.generation.pipeline import INSUFFICIENT_EVIDENCE_MESSAGE, answer_query

CHUNKS = [
    {
        "chunk_id": "c1",
        "parent_doc_id": "p1",
        "source": "a.txt",
        "text": "first",
        "rerank_score": 5.0,
        "retrieved_by": ["keyword", "vector"],
        "confidence_score": 0.8,
    },
    {
        "chunk_id": "c2",
        "parent_doc_id": "p2",
        "source": "b.md",
        "text": "second",
        "rerank_score": 3.0,
        "retrieved_by": ["vector"],
        "confidence_score": 0.7,
    },
]


def _groq_available() -> bool:
    return bool(settings.GROQ_API_KEY)


def _atlas_available() -> bool:
    try:
        return bool(settings.MONGODB_URI)
    except Exception:
        return False


class _FakeReranker:
    def rerank(self, query: str, results: list) -> list:
        return results


class TestGate:
    def test_empty_scored_returns_insufficient_and_skips_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pipeline, "get_cached_answer", lambda q: None)
        monkeypatch.setattr(pipeline, "set_cached_answer", lambda q, r: None)
        monkeypatch.setattr(pipeline, "start_trace", lambda q: {"query": q, "stages": {}})
        monkeypatch.setattr(pipeline, "record_stage", lambda trace, stage, **data: None)
        monkeypatch.setattr(
            pipeline, "finalize_trace", lambda trace, success, error=None: None
        )
        monkeypatch.setattr(pipeline, "hybrid_search", lambda q: [])
        monkeypatch.setattr(pipeline, "get_reranker", lambda: _FakeReranker())
        monkeypatch.setattr(
            pipeline,
            "generate_answer",
            lambda q, c: pytest.fail("generate_answer must NOT be called on the empty path"),
        )
        monkeypatch.setattr(
            llm_client,
            "generate",
            lambda *a, **k: pytest.fail("llm_client.generate must NOT be called"),
        )

        result = answer_query("any query")

        assert result == {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "citations": [],
            "context_chunks": [],
        }


class TestOrchestration:
    def test_non_empty_scored_calls_llm_and_returns_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pipeline, "get_cached_answer", lambda q: None)
        monkeypatch.setattr(pipeline, "set_cached_answer", lambda q, r: None)
        monkeypatch.setattr(pipeline, "start_trace", lambda q: {"query": q, "stages": {}})
        monkeypatch.setattr(pipeline, "record_stage", lambda trace, stage, **data: None)
        monkeypatch.setattr(
            pipeline, "finalize_trace", lambda trace, success, error=None: None
        )
        monkeypatch.setattr(pipeline, "hybrid_search", lambda q: CHUNKS)
        monkeypatch.setattr(pipeline, "get_reranker", lambda: _FakeReranker())
        monkeypatch.setattr(pipeline, "score_and_filter", lambda c: c)
        monkeypatch.setattr(
            pipeline,
            "generate_answer",
            lambda q, c: {"answer": "grounded [1] [2]", "context_chunks": c},
        )

        result = answer_query("about first")

        assert result["answer"] == "grounded [1] [2]"
        assert [c["marker"] for c in result["citations"]] == [1, 2]
        assert result["context_chunks"] == CHUNKS


@pytest.mark.slow
class TestLiveSmoke:
    def test_on_topic_query_answers_with_citation(self) -> None:
        if not _groq_available():
            pytest.skip("no GROQ_API_KEY available")
        if not _atlas_available():
            pytest.skip("no Atlas credentials available")

        result = answer_query("What is MongoDB Atlas?")

        assert result["answer"], "expected a non-empty answer"
        assert result["answer"] != INSUFFICIENT_EVIDENCE_MESSAGE
        assert result["citations"], "expected at least one valid citation"
        assert result["context_chunks"]

    def test_off_topic_query_reports_gate(self) -> None:
        if not _groq_available():
            pytest.skip("no GROQ_API_KEY available")
        if not _atlas_available():
            pytest.skip("no Atlas credentials available")

        from src.retrieval.hybrid import hybrid_search
        from src.retrieval.reranker import get_reranker
        from src.scoring.confidence import score_and_filter

        query = "What is the capital of France?"
        results = hybrid_search(query)
        reranked = get_reranker().rerank(query, results)
        scored = score_and_filter(reranked)

        print("\nOFF-TOPIC confidence scores:", [c["confidence_score"] for c in scored])
        print("OFF-TOPIC rerank scores:", [c.get("rerank_score") for c in scored])
        print("OFF-TOPIC retrieved_by:", [c.get("retrieved_by") for c in scored])

        result = answer_query(query)
        print("OFF-TOPIC gate triggered:", result["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE)
        print("OFF-TOPIC answer:", repr(result["answer"]))


@pytest.mark.slow
class TestQueryCaching:
    def test_repeated_identical_query_served_from_cache_one_llm_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not _groq_available():
            pytest.skip("no GROQ_API_KEY available")
        if not _atlas_available():
            pytest.skip("no Atlas credentials available")

        from src.caching.query_cache import get_cache_key
        from src.db.client import get_collection

        query = "What is MongoDB Atlas?"
        get_collection("cache").delete_one({"_id": get_cache_key(query)})

        calls: list[str] = []
        monkeypatch.setattr(
            llm_client,
            "generate",
            lambda sys_p, user_p: calls.append(user_p)
            or {
                "text": "grounded [1]",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

        result1 = answer_query(query)
        result2 = answer_query(query)

        assert len(calls) == 1, f"expected exactly one LLM call, got {len(calls)}"
        assert result1 == result2

    def test_different_query_still_triggers_real_llm_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not _groq_available():
            pytest.skip("no GROQ_API_KEY available")
        if not _atlas_available():
            pytest.skip("no Atlas credentials available")

        from src.caching.query_cache import get_cache_key
        from src.db.client import get_collection

        query = "What is Atlas Vector Search?"
        get_collection("cache").delete_one({"_id": get_cache_key(query)})

        calls: list[str] = []
        monkeypatch.setattr(
            llm_client,
            "generate",
            lambda sys_p, user_p: calls.append(user_p)
            or {
                "text": "grounded [1]",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

        result = answer_query(query)

        assert len(calls) == 1, "different query must trigger a real LLM call, not a cache hit"
        assert result["answer"], "expected a non-empty answer"
        assert result["answer"] != INSUFFICIENT_EVIDENCE_MESSAGE


class _FakeTraceCollection:
    def __init__(self) -> None:
        self.docs: list[dict] = []

    def insert_one(self, doc: dict) -> None:
        self.docs.append(doc)

    def create_index(self, keys: str, **kwargs: object) -> str:
        return keys


class TestQueryTracing:
    def test_success_path_writes_trace_document(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.observability import tracer

        fake = _FakeTraceCollection()
        monkeypatch.setattr(tracer, "get_collection", lambda name: fake)
        monkeypatch.setattr(pipeline, "get_cached_answer", lambda q: None)
        monkeypatch.setattr(pipeline, "set_cached_answer", lambda q, r: None)
        monkeypatch.setattr(pipeline, "hybrid_search", lambda q: CHUNKS)
        monkeypatch.setattr(pipeline, "get_reranker", lambda: _FakeReranker())
        monkeypatch.setattr(pipeline, "score_and_filter", lambda c: c)
        monkeypatch.setattr(
            pipeline,
            "generate_answer",
            lambda q, c: {
                "answer": "grounded [1] [2]",
                "context_chunks": c,
                "token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )

        result = answer_query("about first")

        assert result["answer"] == "grounded [1] [2]"
        assert len(fake.docs) == 1
        trace = fake.docs[0]
        assert trace["query"] == "about first"
        assert trace["success"] is True
        assert trace["error"] is None
        assert trace["duration_ms"] > 0
        assert {"cache", "retrieval", "rerank", "confidence", "generation", "citations"} <= set(
            trace["stages"]
        )
        assert trace["stages"]["generation"]["token_usage"]["total_tokens"] == 10

    def test_failure_path_traces_error_and_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.observability import tracer

        fake = _FakeTraceCollection()
        monkeypatch.setattr(tracer, "get_collection", lambda name: fake)
        monkeypatch.setattr(pipeline, "get_cached_answer", lambda q: None)

        def _boom(q: str) -> list:
            raise ValueError("retrieval exploded")

        monkeypatch.setattr(pipeline, "hybrid_search", _boom)

        with pytest.raises(ValueError, match="retrieval exploded"):
            answer_query("about first")

        assert len(fake.docs) == 1
        trace = fake.docs[0]
        assert trace["query"] == "about first"
        assert trace["success"] is False
        assert trace["error"] == "retrieval exploded"
        assert "cache" in trace["stages"]