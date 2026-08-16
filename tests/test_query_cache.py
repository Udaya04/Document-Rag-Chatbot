from __future__ import annotations

import pytest

from src.caching import query_cache
from src.caching.query_cache import (
    CACHE_TTL_SECONDS,
    _ensure_ttl_index,
    get_cache_key,
    get_cached_answer,
    set_cached_answer,
)


class _FakeCacheCollection:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.indexes: list[dict] = []

    def create_index(self, keys: str, **kwargs: object) -> str:
        self.indexes.append({"key": keys, **kwargs})
        return keys

    def replace_one(self, query: dict, doc: dict, upsert: bool = False) -> None:
        self.docs[query["_id"]] = doc

    def find_one(self, query: dict) -> dict | None:
        return self.docs.get(query["_id"])

    def list_indexes(self) -> list[dict]:
        return list(self.indexes)


def _fake_collection(monkeypatch: pytest.MonkeyPatch) -> _FakeCacheCollection:
    fake = _FakeCacheCollection()
    monkeypatch.setattr(query_cache, "get_collection", lambda name: fake)
    return fake


class TestCacheKey:
    def test_case_and_whitespace_ignored(self) -> None:
        assert get_cache_key("MongoDB Atlas?") == get_cache_key(" mongodb atlas? ")
        assert get_cache_key("What is this?") == get_cache_key("   what is this?   ")

    def test_genuinely_different_queries_differ(self) -> None:
        assert get_cache_key("What is MongoDB Atlas?") != get_cache_key("What is France?")
        assert get_cache_key("What is Atlas Vector Search?") != get_cache_key("What is MongoDB Atlas?")


class TestRoundTrip:
    def test_set_then_get_returns_same_result_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_collection(monkeypatch)
        result = {
            "answer": "A grounded [1] answer.",
            "citations": [{"marker": 1, "chunk_id": "c1"}],
            "context_chunks": [{"chunk_id": "c1", "text": "first"}],
        }

        set_cached_answer("What is MongoDB Atlas?", result)
        got = get_cached_answer("what is mongodb atlas? ")

        assert got == result

    def test_never_cached_query_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_collection(monkeypatch)
        assert get_cached_answer("never been asked") is None


class TestTtlIndex:
    def test_ttl_index_created_on_created_at(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _fake_collection(monkeypatch)

        _ensure_ttl_index(fake)

        indexes = fake.list_indexes()
        assert any(
            idx["key"] == "created_at"
            and idx["expireAfterSeconds"] == CACHE_TTL_SECONDS
            for idx in indexes
        ), f"expected a TTL index on created_at with expireAfterSeconds={CACHE_TTL_SECONDS}, got {indexes}"