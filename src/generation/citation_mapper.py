"""Map [n] citation markers in generated text back to their source chunks."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"[\[【](\d+)[^\]】]*[\]】]")


def extract_citations(answer_text: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse [n] markers out of the answer into citation dicts.

    Markers are 1-indexed against the chunk list. Each unique valid marker
    resolves to one citation. Out-of-range markers are logged and dropped
    (never crash); no markers yields an empty list.
    """
    citations: list[dict[str, Any]] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer_text):
        marker = int(match.group(1))
        if marker < 1 or marker > len(chunks):
            logger.warning(
                "Citation marker [%d] out of range (%d chunks); dropping",
                marker,
                len(chunks),
            )
            continue
        if marker in seen:
            continue
        seen.add(marker)
        chunk = chunks[marker - 1]
        citations.append(
            {
                "marker": marker,
                "chunk_id": chunk.get("chunk_id"),
                "parent_doc_id": chunk.get("parent_doc_id"),
                "source": chunk.get("source"),
            }
        )
    return citations


def map_citations(answer_text: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the answer unchanged plus its resolved citation list."""
    return {
        "answer": answer_text,
        "citations": extract_citations(answer_text, chunks),
    }