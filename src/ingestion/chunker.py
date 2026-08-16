from __future__ import annotations

from collections import deque

import tiktoken
from tiktoken import Encoding

from src.ingestion.normalize import Chunk, NormalizedDocument

_encoders: dict[str, Encoding] = {}


def _get_encoding(name: str) -> Encoding:
    enc = _encoders.get(name)
    if enc is None:
        enc = tiktoken.get_encoding(name)
        _encoders[name] = enc
    return enc


class RecursiveTokenChunker:
    _SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size")
        self._encoding = _get_encoding(encoding_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_recursive(self, text: str, seps: list[str], max_tokens: int) -> list[str]:
        if not text:
            return []
        if len(self._encoding.encode(text)) <= max_tokens:
            return [text]
        if not seps:
            tokens = self._encoding.encode(text)
            return [
                self._encoding.decode(tokens[i : i + max_tokens])
                for i in range(0, len(tokens), max_tokens)
            ]
        sep = seps[0]
        pieces: list[str] = []
        for part in text.split(sep):
            if not part:
                continue
            if len(self._encoding.encode(part)) <= max_tokens:
                pieces.append(part)
            else:
                pieces.extend(self._split_recursive(part, seps[1:], max_tokens))
        return pieces

    def _pack(
        self, pieces: list[str], chunk_size: int, overlap: int
    ) -> list[str]:
        enc = self._encoding
        chunks: list[str] = []
        buffer: list[int] = []
        queue: deque[str] = deque(pieces)

        while queue:
            piece = queue.popleft()
            piece_tokens = enc.encode(piece)
            if not piece_tokens:
                continue
            if len(buffer) + len(piece_tokens) <= chunk_size:
                buffer.extend(piece_tokens)
                continue

            if buffer:
                chunks.append(enc.decode(buffer))
            overlap_tokens = buffer[-overlap:] if overlap and buffer else []
            room = chunk_size - len(overlap_tokens)
            if room <= 0:
                queue.appendleft(piece)
                buffer = []
                continue
            buffer = overlap_tokens + piece_tokens[:room]
            leftover = piece_tokens[room:]
            if leftover:
                queue.appendleft(enc.decode(leftover))

        if buffer:
            chunks.append(enc.decode(buffer))
        return [chunk for chunk in chunks if chunk.strip()]

    def chunk_text(self, text: str) -> list[str]:
        pieces = self._split_recursive(text, list(self._SEPARATORS), self.chunk_size)
        return self._pack(pieces, self.chunk_size, self.overlap)


def chunk_document(
    doc: NormalizedDocument,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    chunker = RecursiveTokenChunker(
        chunk_size=chunk_size,
        overlap=overlap,
    )
    texts = chunker.chunk_text(doc.content)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}:{index}",
            parent_doc_id=doc.doc_id,
            chunk_index=index,
            text=text,
            metadata=doc.metadata.model_copy(deep=True),
        )
        for index, text in enumerate(texts)
    ]
