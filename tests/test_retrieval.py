from __future__ import annotations

from typing import Any

import pytest

from src.ingestion.normalize import Chunk, DocumentMetadata
from src.retrieval import embedder, mongo_search, mongo_vector


def _chunk(embedding: list[float] | None = None) -> Chunk:
    return Chunk(
        chunk_id="c:0",
        parent_doc_id="doc-1",
        chunk_index=0,
        text="hello",
        metadata=DocumentMetadata(source_file="a.txt", file_type="txt"),
        embedding=embedding,
    )


class TestChunkEmbedding:
    def test_defaults_none(self) -> None:
        assert _chunk().embedding is None

    def test_settable(self) -> None:
        assert _chunk([1.0, 2.0, 3.0]).embedding == [1.0, 2.0, 3.0]


@pytest.fixture(autouse=True)
def _reset_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder.Embedder._instance = None

    class FakeST:
        def __init__(self, name: str) -> None:
            self._name = name

        def get_embedding_dimension(self) -> int:
            return 384

        def encode(self, texts: list[str], **_: Any) -> list[list[float]]:
            return [[float(len(t))] * 3 for t in texts]

    monkeypatch.setattr(embedder, "SentenceTransformer", FakeST)
    yield
    embedder.Embedder._instance = None


class TestEmbedder:
    def test_singleton(self) -> None:
        assert embedder.Embedder() is embedder.Embedder()

    def test_dimensions(self) -> None:
        assert embedder.Embedder().dimensions == 384

    def test_embed_text(self) -> None:
        assert embedder.Embedder().embed_text("hi") == [2.0, 2.0, 2.0]

    def test_embed_batch(self) -> None:
        assert embedder.Embedder().embed_batch(["a", "bb"]) == [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]

    def test_embed_batch_empty(self) -> None:
        assert embedder.Embedder().embed_batch([]) == []


class _FakeCollection:
    def __init__(self) -> None:
        self.pipelines: list[list[dict[str, Any]]] = []

    def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.pipelines.append(pipeline)
        return []


class _FakeDB(dict):
    def __init__(self) -> None:
        super().__init__()
        self.chunks = _FakeCollection()

    def __getitem__(self, key: str) -> _FakeCollection:
        return self.chunks


class TestKeywordSearch:
    def test_builds_search_pipeline(self) -> None:
        fdb = _FakeDB()
        mongo_search.keyword_search("hello world", limit=5, db=fdb)
        pipe = fdb.chunks.pipelines[0]
        assert pipe[0]["$search"]["index"] == "default"
        assert pipe[0]["$search"]["text"] == {"query": "hello world", "path": "text"}
        assert pipe[1] == {"$limit": 5}
        assert pipe[2]["$project"]["score"] == {"$meta": "searchScore"}
        assert set(pipe[2]["$project"]) >= {"_id", "chunk_id", "text", "parent_doc_id"}


class TestVectorSearch:
    def test_builds_vector_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeEmb:
            def embed_text(self, query: str) -> list[float]:
                return [0.1] * 384

        monkeypatch.setattr(mongo_vector, "get_embedder", lambda: FakeEmb())
        fdb = _FakeDB()
        mongo_vector.vector_search("hello", limit=7, db=fdb)
        pipe = fdb.chunks.pipelines[0]
        assert pipe[0]["$vectorSearch"]["index"] == "vector_index"
        assert pipe[0]["$vectorSearch"]["path"] == "embedding"
        assert pipe[0]["$vectorSearch"]["queryVector"] == [0.1] * 384
        assert pipe[0]["$vectorSearch"]["limit"] == 7
        assert pipe[1]["$project"]["score"] == {"$meta": "vectorSearchScore"}
