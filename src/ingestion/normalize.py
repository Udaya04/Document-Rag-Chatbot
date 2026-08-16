from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source_file: str
    file_type: str
    processed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trust_score: float = 1.0
    version: int = 1
    custom: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    doc_id: str
    content: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    chunk_id: str
    parent_doc_id: str
    chunk_index: int
    text: str
    metadata: DocumentMetadata
    embedding: list[float] | None = None


def generate_doc_id(content: str, source_file: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(source_file.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


class FileProcessingError(Exception):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to process '{path}': {reason}")


class UnsupportedFileTypeError(FileProcessingError):
    def __init__(self, path: str, file_type: str) -> None:
        super().__init__(path, f"unsupported file type: {file_type!r}")
