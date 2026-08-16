from __future__ import annotations

import logging
from pathlib import Path

from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from src.db.client import get_db
from src.ingestion.chunker import chunk_document
from src.ingestion.loaders import process_raw_file
from src.ingestion.normalize import FileProcessingError
from src.ingestion.normalize import generate_doc_id
from src.retrieval.embedder import get_embedder

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, db: Database | None = None, embed: bool = True) -> None:
        self._db = db if db is not None else get_db()
        self._docs = self._db["docs"]
        self._chunks = self._db["chunks"]
        self._embedder = get_embedder() if embed else None
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        try:
            self._docs.create_index("doc_id", unique=True)
            self._chunks.create_index("parent_doc_id")
        except PyMongoError as exc:
            logger.warning("Could not create indexes: %s", exc)

    def _is_duplicate(self, doc_id: str) -> bool:
        return self._docs.find_one({"doc_id": doc_id}, {"_id": 1}) is not None

    def _raw_for_hash(self, path: Path) -> str:
        return path.read_bytes().decode("latin-1")

    def process_file(self, file_path: Path) -> str:
        path = Path(file_path)
        doc_id = generate_doc_id(self._raw_for_hash(path), str(path))

        if self._is_duplicate(doc_id):
            logger.info("Skipping duplicate: %s (doc_id=%s)", path, doc_id)
            return "skipped"

        doc = process_raw_file(path)
        chunks = chunk_document(doc)

        if self._embedder is not None:
            texts = [chunk.text for chunk in chunks]
            embeddings = self._embedder.embed_batch(texts)
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding

        doc_payload = doc.model_dump(mode="python")
        doc_payload["_id"] = doc_id
        try:
            self._docs.insert_one(doc_payload)
        except DuplicateKeyError:
            logger.warning("Duplicate key on insert (race): %s", doc_id)
            return "skipped"
        except PyMongoError as exc:
            logger.error("Failed to insert document %s: %s", doc_id, exc)
            raise

        try:
            self._chunks.insert_many(
                [chunk.model_dump(mode="python") for chunk in chunks],
                ordered=False,
            )
        except PyMongoError as exc:
            logger.error("Chunk insert failed for %s, rolling back document: %s", doc_id, exc)
            self._docs.delete_one({"_id": doc_id})
            raise

        logger.info("Ingested %s: %d chunks (doc_id=%s)", path, len(chunks), doc_id)
        return "ingested"

    def process_directory(self, dir_path: str) -> dict[str, int]:
        root = Path(dir_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {root}")

        stats = {"ingested": 0, "skipped": 0, "failed": 0}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            try:
                result = self.process_file(path)
                stats[result] += 1
            except FileProcessingError as exc:
                logger.error("Skipping unprocessable file %s: %s", path, exc.reason)
                stats["failed"] += 1
            except PyMongoError as exc:
                logger.error("Mongo error for %s: %s", path, exc)
                stats["failed"] += 1
            except Exception as exc:
                logger.exception("Unexpected error for %s: %s", path, exc)
                stats["failed"] += 1
        return stats
