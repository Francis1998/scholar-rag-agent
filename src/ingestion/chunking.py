"""Document chunking utilities."""

import hashlib

from retrieval.models import Chunk, Document


def stable_id(value: str, prefix: str) -> str:
    """Return a stable short identifier for a value."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


class TextChunker:
    """Split normalized documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        """Create a text chunker."""
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Return chunks for a document."""
        text = " ".join(document.text.split())
        if not text:
            return []
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]
            chunk_id = stable_id(f"{document.document_id}:{index}:{chunk_text}", "chunk")
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    title=document.title,
                    text=chunk_text,
                    source=document.source,
                    metadata={**document.metadata, "chunk_index": str(index)},
                )
            )
            if end == len(text):
                break
            start = max(end - self._overlap, start + 1)
            index += 1
        return chunks
