from __future__ import annotations

import logging
from pathlib import Path

import pymupdf
import markdown as markdown_lib
from bs4 import BeautifulSoup

from src.ingestion.normalize import (
    DocumentMetadata,
    FileProcessingError,
    NormalizedDocument,
    UnsupportedFileTypeError,
    generate_doc_id,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({"txt", "pdf", "html", "htm", "md"})

_MARKDOWN_EXTENSIONS = ["fenced_code", "tables", "codehilite"]
_MD_BLACKLIST = {"script", "style", "pre", "code"}


def _raw_for_hash(data: bytes) -> str:
    return data.decode("latin-1")


def _build_document(
    path: Path,
    raw: bytes,
    content: str,
    file_type: str,
    custom: dict | None = None,
) -> NormalizedDocument:
    doc_id = generate_doc_id(_raw_for_hash(raw), str(path))
    return NormalizedDocument(
        doc_id=doc_id,
        content=content,
        metadata=DocumentMetadata(
            source_file=str(path),
            file_type=file_type,
            custom=custom or {},
        ),
    )


def load_txt(path: Path) -> NormalizedDocument:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FileProcessingError(str(path), f"cannot read file: {exc}") from exc
    content = raw.decode("utf-8", errors="replace")
    return _build_document(path, raw, content, "txt")


def load_pdf(path: Path) -> NormalizedDocument:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FileProcessingError(str(path), f"cannot read file: {exc}") from exc
    try:
        doc = pymupdf.open(str(path))
        pages = [page.get_text() for page in doc]
        page_count = len(pages)
        doc.close()
    except pymupdf.FileDataError as exc:
        raise FileProcessingError(str(path), f"corrupt or unreadable PDF: {exc}") from exc
    except pymupdf.EmptyFileError as exc:
        raise FileProcessingError(str(path), f"empty PDF: {exc}") from exc
    except Exception as exc:
        raise FileProcessingError(str(path), f"PDF parse failed: {exc}") from exc

    content = "\n\n".join(pages).strip()
    return _build_document(
        path,
        raw,
        content,
        "pdf",
        {"page_count": page_count},
    )


def load_html(path: Path) -> NormalizedDocument:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FileProcessingError(str(path), f"cannot read file: {exc}") from exc
    try:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        content = soup.get_text(separator="\n")
        content = "\n".join(line.strip() for line in content.splitlines()).strip()
    except Exception as exc:
        raise FileProcessingError(str(path), f"HTML parse failed: {exc}") from exc

    custom: dict = {}
    if title:
        custom["title"] = title
    return _build_document(path, raw, content, "html", custom)


def load_md(path: Path) -> NormalizedDocument:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FileProcessingError(str(path), f"cannot read file: {exc}") from exc
    try:
        text = raw.decode("utf-8", errors="replace")
        html = markdown_lib.markdown(text, extensions=_MARKDOWN_EXTENSIONS)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_MD_BLACKLIST):
            tag.decompose()
        content = soup.get_text(separator="\n")
        content = "\n".join(line.strip() for line in content.splitlines()).strip()
    except Exception as exc:
        raise FileProcessingError(str(path), f"markdown parse failed: {exc}") from exc

    return _build_document(path, raw, content, "md")


_LOADERS = {
    "txt": load_txt,
    "pdf": load_pdf,
    "html": load_html,
    "htm": load_html,
    "md": load_md,
}


def process_raw_file(file_path: Path) -> NormalizedDocument:
    path = Path(file_path)
    if not path.is_file():
        raise FileProcessingError(str(path), "path is not a file")
    ext = path.suffix.lower().lstrip(".")
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileTypeError(str(path), ext)
    logger.debug("Loading %s via %s loader", path, ext)
    return loader(path)
