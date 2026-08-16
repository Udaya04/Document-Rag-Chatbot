"""Cross-encoder reranking of retrieval candidates."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    _instance: Reranker | None = None
    _lock = Lock()

    def __new__(cls) -> Reranker:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._model = None
                cls._instance = instance
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_model", None) is None:
            logger.info("Loading cross-encoder model %s", _MODEL_NAME)
            self._model = CrossEncoder(_MODEL_NAME)
            logger.info("Cross-encoder model loaded")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Score candidates against the query and return the top ``top_k``.

        Each candidate's ``text`` field is paired with the query and scored with
        the cross-encoder. Returned dicts are copies with ``rerank_score``
        attached; the input list is never mutated.

        Args:
            query: Query text to score candidates against.
            candidates: Candidate dicts, each expected to have a ``text`` field.
            top_k: Number of best-scoring candidates to return.

        Returns:
            Up to ``top_k`` copied candidates sorted by ``rerank_score``
            descending.
        """
        if not candidates:
            return []

        texts = [str(candidate.get("text", "")) for candidate in candidates]
        scores = self._model.predict(
            [(query, text) for text in texts],
            batch_size=32,
            show_progress_bar=False,
        )

        scored: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores, strict=True):
            copy = dict(candidate)
            copy["rerank_score"] = float(score)
            scored.append(copy)

        scored.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
        return scored[:top_k]


def get_reranker() -> Reranker:
    return Reranker()