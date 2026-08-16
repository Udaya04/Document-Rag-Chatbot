from __future__ import annotations

from datetime import datetime

import pytest

from src.observability import tracer
from src.observability.tracer import (
    TRACE_TTL_SECONDS,
    _ensure_trace_ttl_index,
    finalize_trace,
    record_stage,
    start_trace,
)


class _FakeTraceCollection:
    def __init__(self) -> None:
        self.docs: list[dict] = []
        self.indexes: list[dict] = []

    def insert_one(self, doc: dict) -> None:
        self.docs.append(doc)

    def create_index(self, keys: str, **kwargs: object) -> str:
        self.indexes.append({"key": keys, **kwargs})
        return keys

    def list_indexes(self) -> list[dict]:
        return list(self.indexes)


def _fake_collection(monkeypatch: pytest.MonkeyPatch) -> _FakeTraceCollection:
    fake = _FakeTraceCollection()
    monkeypatch.setattr(tracer, "get_collection", lambda name: fake)
    return fake


class TestStartTrace:
    def test_returns_correct_initial_shape(self) -> None:
        trace = start_trace("What is MongoDB Atlas?")
        assert trace["query"] == "What is MongoDB Atlas?"
        assert isinstance(trace["started_at"], datetime)
        assert trace["stages"] == {}


class TestRecordStage:
    def test_adds_and_updates_without_clobbering(self) -> None:
        trace = start_trace("q")
        record_stage(trace, "cache", hit=False)
        record_stage(trace, "retrieval", chunks=[{"chunk_id": "c1", "score": 1.0}])
        record_stage(trace, "cache", hit=True)

        assert trace["stages"]["cache"] == {"hit": True}
        assert trace["stages"]["retrieval"] == {"chunks": [{"chunk_id": "c1", "score": 1.0}]}
        assert set(trace["stages"]) == {"cache", "retrieval"}


class TestFinalizeTrace:
    def test_success_case_persists_duration_and_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _fake_collection(monkeypatch)
        trace = start_trace("q")

        finalize_trace(trace, success=True)

        assert len(fake.docs) == 1
        doc = fake.docs[0]
        assert doc is trace
        assert doc["success"] is True
        assert doc["error"] is None
        assert doc["duration_ms"] > 0
        assert doc["started_at"] is trace["started_at"]

    def test_failure_case_passes_error_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _fake_collection(monkeypatch)
        trace = start_trace("q")

        finalize_trace(trace, success=False, error="retrieval exploded")

        doc = fake.docs[0]
        assert doc["success"] is False
        assert doc["error"] == "retrieval exploded"
        assert doc["duration_ms"] > 0


class TestTraceTtlIndex:
    def test_ttl_index_created_on_started_at(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _fake_collection(monkeypatch)
        monkeypatch.setattr(tracer, "_ttl_index_ensured", False)

        _ensure_trace_ttl_index(fake)

        indexes = fake.list_indexes()
        assert any(
            idx["key"] == "started_at" and idx["expireAfterSeconds"] == TRACE_TTL_SECONDS
            for idx in indexes
        ), f"expected a TTL index on started_at with expireAfterSeconds={TRACE_TTL_SECONDS}, got {indexes}"