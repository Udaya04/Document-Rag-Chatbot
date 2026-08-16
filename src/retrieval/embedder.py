from __future__ import annotations

import logging
from threading import Lock

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder:
    _instance: Embedder | None = None
    _lock = Lock()

    def __new__(cls) -> Embedder:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._model = None
                cls._instance = instance
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_model", None) is None:
            logger.info("Loading embedding model %s", _MODEL_NAME)
            self._model = SentenceTransformer(_MODEL_NAME)
            logger.info("Embedding model loaded (dims=%d)", self.dimensions)

    @property
    def dimensions(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return [float(value) for value in embedding]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
        )
        return [[float(value) for value in row] for row in embeddings]


def get_embedder() -> Embedder:
    return Embedder()
