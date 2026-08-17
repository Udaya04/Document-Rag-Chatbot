"""Document ingestion: raw files -> normalized docs -> chunks -> Mongo (+ embeddings)."""

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

_DEFAULT_BATCH_SIZE = 200


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
        """True only if the doc exists AND has chunks; self-heals crash orphans."""
        if self._docs.find_one({"_id": doc_id}, {"_id": 1}) is None:
            return False
        if self._chunks.count_documents({"parent_doc_id": doc_id}) == 0:
            logger.warning("Removing orphan doc without chunks: %s", doc_id)
            self._docs.delete_one({"_id": doc_id})
            return False
        return True

    def _raw_for_hash(self, path: Path) -> str:
        return path.read_bytes().decode("latin-1")

    def _insert_doc_with_chunks(
        self,
        path: Path,
        doc_id: str,
        doc,
        chunks,
    ) -> str:
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

    def _existing_ids(self, doc_ids: list[str]) -> set[str]:
        try:
            return {doc["_id"] for doc in self._docs.find({"_id": {"$in": doc_ids}}, {"_id": 1})}
        except PyMongoError as exc:
            logger.error("Batched dedup query failed, falling back per-file: %s", exc)
            return set()

    def _parent_ids_with_chunks(self, doc_ids: list[str]) -> set[str] | None:
        try:
            return {
                chunk["parent_doc_id"]
                for chunk in self._chunks.find(
                    {"parent_doc_id": {"$in": doc_ids}},
                    {"parent_doc_id": 1},
                )
            }
        except PyMongoError as exc:
            logger.error("Orphan-check query failed: %s", exc)
            return None

    def _process_window(
        self,
        window: list[Path],
        stats: dict[str, int],
    ) -> None:
        loaded: list[tuple[Path, object]] = []
        for path in window:
            try:
                doc = process_raw_file(path)
            except FileProcessingError as exc:
                logger.error("Skipping unprocessable file %s: %s", path, exc.reason)
                stats["failed"] += 1
                continue
            loaded.append((path, doc))

        if not loaded:
            return

        doc_ids = [doc.doc_id for _, doc in loaded]
        existing = self._existing_ids(doc_ids)

        has_chunks = self._parent_ids_with_chunks(doc_ids)
        if has_chunks is not None:
            orphans = existing - has_chunks
            if orphans:
                try:
                    self._docs.delete_many({"_id": {"$in": list(orphans)}})
                except PyMongoError as exc:
                    logger.error("Failed to delete orphan docs: %s", exc)
                    raise
                existing -= orphans

        to_insert: list[tuple[Path, object, list]] = []
        for path, doc in loaded:
            if doc.doc_id in existing:
                logger.info("Skipping duplicate: %s (doc_id=%s)", path, doc.doc_id)
                stats["skipped"] += 1
                continue
            try:
                chunks = chunk_document(doc)
            except Exception as exc:
                logger.exception("Chunking failed for %s: %s", path, exc)
                stats["failed"] += 1
                continue
            to_insert.append((path, doc, chunks))

        if to_insert and self._embedder is not None:
            flat_chunks = [chunk for (_, _, chunks) in to_insert for chunk in chunks]
            texts = [chunk.text for chunk in flat_chunks]
            embeddings = self._embedder.embed_batch(texts)
            for chunk, embedding in zip(flat_chunks, embeddings, strict=True):
                chunk.embedding = embedding

        for path, doc, chunks in to_insert:
            try:
                result = self._insert_doc_with_chunks(path, doc.doc_id, doc, chunks)
            except PyMongoError as exc:
                logger.error("Mongo error for %s: %s", path, exc)
                stats["failed"] += 1
                continue
            except Exception as exc:
                logger.exception("Unexpected error for %s: %s", path, exc)
                stats["failed"] += 1
                continue
            stats[result] += 1

    def process_file(self, file_path: str) -> str:
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

        return self._insert_doc_with_chunks(path, doc_id, doc, chunks)

    def process_directory(
        self,
        dir_path: str,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> dict[str, int]:
        root = Path(dir_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {root}")

        files = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.name.startswith(".")
        ]
        total = len(files)
        stats = {"ingested": 0, "skipped": 0, "failed": 0}

        for start in range(0, total, batch_size):
            window = files[start : start + batch_size]
            self._process_window(window, stats)
            processed = min(start + batch_size, total)
            logger.info(
                "Progress: files processed=%d/%d ingested=%d skipped=%d failed=%d",
                processed,
                total,
                stats["ingested"],
                stats["skipped"],
                stats["failed"],
            )
        return stats