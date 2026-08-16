from __future__ import annotations

import pytest
from datetime import UTC

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
