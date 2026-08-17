from __future__ import annotations

import pytest
from datetime import UTC

from pymongo.errors import DuplicateKeyError, PyMongoError

from src.ingestion.chunker import RecursiveTokenChunker, chunk_document
from src.ingestion.loaders import (
    load_html,
    load_md,
    load_pdf,
    load_txt,
    process_raw_file,
)
from src.ingestion.normalize import (
    DocumentMetadata,
    NormalizedDocument,
    UnsupportedFileTypeError,
    generate_doc_id,
)
from src.ingestion.pipeline import IngestionPipeline


class TestGenerateDocId:
    def test_deterministic(self) -> None:
        a = generate_doc_id("hello world", "a.txt")
        b = generate_doc_id("hello world", "a.txt")
        assert a == b

    def test_unique_for_different_content(self) -> None:
        assert generate_doc_id("one", "a.txt") != generate_doc_id("two", "a.txt")

    def test_unique_for_different_source(self) -> None:
        assert generate_doc_id("same", "a.txt") != generate_doc_id("same", "b.txt")

    def test_is_hex_sha256(self) -> None:
        assert len(generate_doc_id("x", "y")) == 64
        int(generate_doc_id("x", "y"), 16)


class TestDocumentMetadata:
    def test_defaults(self) -> None:
        md = DocumentMetadata(source_file="a.txt", file_type="txt")
        assert md.trust_score == 1.0
        assert md.version == 1
        assert md.custom == {}
        assert md.processed_at.tzinfo is not None
        assert md.processed_at.utcoffset() == UTC.utcoffset(None)


class TestChunker:
    def test_chunk_id_continuity(self) -> None:
        doc = NormalizedDocument(
            doc_id="doc-1",
            content=" ".join(["word"] * 3000),
            metadata=DocumentMetadata(source_file="a.txt", file_type="txt"),
        )
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.parent_doc_id == "doc-1" for c in chunks)
        assert chunks[0].chunk_id == "doc-1:0"

    def test_chunk_token_limit_respected(self) -> None:
        doc = NormalizedDocument(
            doc_id="doc-1",
            content=" ".join(["word"] * 3000),
            metadata=DocumentMetadata(source_file="a.txt", file_type="txt"),
        )
        chunker = RecursiveTokenChunker(chunk_size=100, overlap=10)
        texts = chunker.chunk_text(doc.content)
        for text in texts:
            assert len(chunker._encoding.encode(text)) <= 100

    def test_metadata_inherited(self) -> None:
        doc = NormalizedDocument(
            doc_id="doc-1",
            content=" ".join(["word"] * 2000),
            metadata=DocumentMetadata(
                source_file="a.txt",
                file_type="txt",
                custom={"title": "x"},
            ),
        )
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        assert all(c.metadata.source_file == "a.txt" for c in chunks)
        assert all(c.metadata.custom["title"] == "x" for c in chunks)
        chunks[0].metadata.custom["title"] = "mutated"
        assert chunks[1].metadata.custom["title"] == "x"

    def test_empty_content(self) -> None:
        doc = NormalizedDocument(
            doc_id="doc-1",
            content="",
            metadata=DocumentMetadata(source_file="a.txt", file_type="txt"),
        )
        assert chunk_document(doc) == []

    def test_short_content_single_chunk(self) -> None:
        doc = NormalizedDocument(
            doc_id="doc-1",
            content="short text",
            metadata=DocumentMetadata(source_file="a.txt", file_type="txt"),
        )
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "short text"

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            RecursiveTokenChunker(chunk_size=0)
        with pytest.raises(ValueError):
            RecursiveTokenChunker(overlap=-1)
        with pytest.raises(ValueError):
            RecursiveTokenChunker(chunk_size=10, overlap=10)


class TestLoaders:
    def test_load_txt(self, tmp_path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("hello world", encoding="utf-8")
        doc = load_txt(p)
        assert doc.content == "hello world"
        assert doc.metadata.file_type == "txt"
        assert doc.metadata.source_file == str(p)

    def test_load_html_title_and_tags_stripped(self, tmp_path) -> None:
        p = tmp_path / "a.html"
        p.write_text(
            "<html><head><title>My Title</title></head>"
            "<body><script>bad()</script><p>Hello <b>world</b></p></body></html>",
            encoding="utf-8",
        )
        doc = load_html(p)
        assert doc.metadata.custom["title"] == "My Title"
        assert "bad()" not in doc.content
        assert "Hello" in doc.content and "world" in doc.content

    def test_load_md(self, tmp_path) -> None:
        p = tmp_path / "a.md"
        p.write_text("# Heading\n\nSome **bold** text.", encoding="utf-8")
        doc = load_md(p)
        assert "Heading" in doc.content
        assert "bold" in doc.content
        assert doc.metadata.file_type == "md"

    def test_load_pdf(self, tmp_path) -> None:
        import pymupdf

        p = tmp_path / "a.pdf"
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Hello PDF world")
        pdf.save(str(p))
        pdf.close()
        doc = load_pdf(p)
        assert "Hello PDF world" in doc.content
        assert doc.metadata.custom["page_count"] == 1

    def test_process_raw_file_router(self, tmp_path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("content", encoding="utf-8")
        assert process_raw_file(p).metadata.file_type == "txt"

    def test_unsupported_extension(self, tmp_path) -> None:
        p = tmp_path / "a.xyz"
        p.write_text("content", encoding="utf-8")
        with pytest.raises(UnsupportedFileTypeError):
            process_raw_file(p)

    def test_missing_file(self, tmp_path) -> None:
        p = tmp_path / "missing.txt"
        from src.ingestion.normalize import FileProcessingError

        with pytest.raises(FileProcessingError):
            process_raw_file(p)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.call_count = 0
        self.batch_sizes: list[int] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.batch_sizes.append(len(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeCollection:
    def __init__(self, data=None) -> None:
        self.data = data if data is not None else {}
        self.calls: list[tuple[str, tuple]] = []
        self.raise_on_next: dict[str, Exception] = {}

    def _record(self, method: str, *args) -> None:
        self.calls.append((method, args))

    def create_index(self, *args, **kwargs) -> None:
        self._record("create_index", args)

    def find_one(self, query, *args, **kwargs):
        self._record("find_one", query)
        return self.data.get(query.get("_id"))

    def count_documents(self, query, *args, **kwargs) -> int:
        self._record("count_documents", query)
        pid = query.get("parent_doc_id")
        if pid is None:
            return len(self.data)
        return sum(1 for c in self.data.values() if c.get("parent_doc_id") == pid)

    def find(self, query, *args, **kwargs):
        self._record("find", query)
        if "$in" in query.get("_id", {}):
            ids = query["_id"]["$in"]
            return [{"_id": i} for i in ids if i in self.data]
        if "$in" in query.get("parent_doc_id", {}):
            ids = query["parent_doc_id"]["$in"]
            return [
                {"parent_doc_id": c["parent_doc_id"]}
                for c in self.data.values()
                if c.get("parent_doc_id") in ids
            ]
        return []

    def delete_one(self, query) -> None:
        self._record("delete_one", query)
        self.data.pop(query.get("_id"), None)

    def delete_many(self, query) -> None:
        self._record("delete_many", query)
        for doc_id in query.get("_id", {}).get("$in", []):
            self.data.pop(doc_id, None)

    def insert_one(self, doc) -> None:
        self._record("insert_one", doc)
        if "insert_one" in self.raise_on_next:
            raise self.raise_on_next.pop("insert_one")
        if doc["_id"] in self.data:
            raise DuplicateKeyError(f"duplicate {doc['_id']}")
        self.data[doc["_id"]] = doc

    def insert_many(self, docs, **kwargs) -> None:
        self._record("insert_many", docs)
        if "insert_many" in self.raise_on_next:
            raise self.raise_on_next.pop("insert_many")
        for doc in docs:
            self.data[doc["chunk_id"]] = doc


class _FakeDB:
    def __init__(self, docs_data=None, chunks_data=None) -> None:
        self.docs = _FakeCollection(docs_data or {})
        self.chunks = _FakeCollection(chunks_data or {})

    def __getitem__(self, name):
        return {"docs": self.docs, "chunks": self.chunks}[name]


def _write_txt(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestIngestionPipeline:
    def _make_pipeline(self):
        db = _FakeDB()
        pipeline = IngestionPipeline(db=db, embed=False)
        pipeline._embedder = _FakeEmbedder()
        return pipeline, db

    def test_directory_batches_dedup_and_embed(self, tmp_path) -> None:
        pipeline, db = self._make_pipeline()
        for i in range(3):
            _write_txt(tmp_path, f"a{i}.txt", f"Content {i} " * 20)

        stats = pipeline.process_directory(str(tmp_path), batch_size=2)

        assert stats == {"ingested": 3, "skipped": 0, "failed": 0}
        in_finds = [
            call for call in db.docs.calls
            if call[0] == "find" and "$in" in call[1][0].get("_id", {})
        ]
        assert len(in_finds) == 2  # one batched $in dedup query per window
        assert [call for call in db.docs.calls if call[0] == "find_one"] == []
        assert pipeline._embedder.call_count == 2  # one embed_batch per window
        assert pipeline._embedder.batch_sizes == [2, 1]
        assert sum(1 for c in db.docs.calls if c[0] == "insert_one") == 3
        assert sum(1 for c in db.chunks.calls if c[0] == "insert_many") == 3

    def test_directory_reprocesses_orphan_docs(self, tmp_path) -> None:
        pipeline, db = self._make_pipeline()
        path = _write_txt(tmp_path, "o.txt", "Orphan content " * 10)
        doc_id = generate_doc_id(path.read_bytes().decode("latin-1"), str(path))
        db.docs.data[doc_id] = {"_id": doc_id}  # crash left doc without chunks

        stats = pipeline.process_directory(str(tmp_path), batch_size=10)

        assert stats == {"ingested": 1, "skipped": 0, "failed": 0}
        assert ("delete_many", ({"_id": {"$in": [doc_id]}},)) in db.docs.calls
        assert doc_id in db.docs.data
        assert any(c.get("parent_doc_id") == doc_id for c in db.chunks.data.values())

    def test_directory_skips_duplicate_files(self, tmp_path) -> None:
        pipeline, db = self._make_pipeline()
        _write_txt(tmp_path, "d.txt", "Same content " * 10)

        first = pipeline.process_directory(str(tmp_path), batch_size=10)
        second = pipeline.process_directory(str(tmp_path), batch_size=10)

        assert first == {"ingested": 1, "skipped": 0, "failed": 0}
        assert second == {"ingested": 0, "skipped": 1, "failed": 0}

    def test_directory_counts_unprocessable_as_failed(self, tmp_path) -> None:
        pipeline, db = self._make_pipeline()
        _write_txt(tmp_path, "ok.txt", "Fine content " * 10)
        (tmp_path / "bad.xyz").write_text("nope", encoding="utf-8")

        stats = pipeline.process_directory(str(tmp_path), batch_size=10)

        assert stats == {"ingested": 1, "skipped": 0, "failed": 1}

    def test_insert_race_skips_not_fails(self, tmp_path) -> None:
        pipeline, db = self._make_pipeline()
        path = _write_txt(tmp_path, "race.txt", "Race content " * 10)
        doc = process_raw_file(path)
        chunks = chunk_document(doc)
        db.docs.raise_on_next["insert_one"] = DuplicateKeyError("dup")

        result = pipeline._insert_doc_with_chunks(path, doc.doc_id, doc, chunks)

        assert result == "skipped"
        assert sum(1 for c in db.chunks.calls if c[0] == "insert_many") == 0

    def test_chunk_insert_failure_rolls_back_doc(self, tmp_path) -> None:
        pipeline, db = self._make_pipeline()
        path = _write_txt(tmp_path, "rb.txt", "Rollback content " * 10)
        doc = process_raw_file(path)
        chunks = chunk_document(doc)
        db.chunks.raise_on_next["insert_many"] = PyMongoError("boom")

        with pytest.raises(PyMongoError):
            pipeline._insert_doc_with_chunks(path, doc.doc_id, doc, chunks)

        assert doc.doc_id not in db.docs.data
        assert any(c[0] == "delete_one" for c in db.docs.calls)
