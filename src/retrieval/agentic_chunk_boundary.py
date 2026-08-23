"""Deterministic, semantic-boundary-aware text splitting for retrieval chunks."""

import hashlib
import re
from collections.abc import Callable

from retrieval.models import Chunk, Document

_HEADING_LINE = re.compile(r"(?m)^ {0,3}#{1,6}[ \t]+\S.*$")
_BLANK_LINE = re.compile(r"\n\s*\n+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _stable_chunk_id(document_id: str, index: int, text: str) -> str:
    """Return a stable short chunk id derived from document id, index, and text."""
    digest = hashlib.sha256(f"{document_id}:{index}:{text}".encode()).hexdigest()[:16]
    return f"chunk-{digest}"


def _merge_join(first: str, second: str) -> str:
    """Join two adjacent chunks using a separator that matches their shape.

    Multi-line chunks (paragraphs, heading sections) are rejoined with a
    blank line so structure is preserved; single-line chunks (sentences,
    words) are rejoined with a single space.
    """
    if "\n" in first or "\n" in second or _HEADING_LINE.match(second.lstrip()):
        return f"{first}\n\n{second}"
    return f"{first} {second}"


class AgenticChunkBoundarySplitter:
    """Split long text on the most semantic boundary that keeps it within bounds.

    Inspired by LlamaIndex's sentence-window parsing, Chonkie's recursive
    chunkers, and LangChain's ``MarkdownHeaderTextSplitter``, this splitter
    tries boundaries from most to least semantic: markdown headings, then
    blank-line paragraph breaks, then sentence boundaries, and only falls
    back to word- or character-level cuts when a single sentence itself
    exceeds ``max_chars``. Adjacent chunks smaller than ``min_chars`` are
    then merged with a neighbor when the merge still fits under
    ``max_chars``. Despite the "agentic" name, no LLM or network call is
    used: this is pure, deterministic text splitting meant to feed
    agent-ready retrieval chunks for **GPT-5.5**, **Claude Sonnet 4.6**,
    **Gemini 3.x**, or **Kimi K2** downstream synthesis.
    """

    def __init__(self, max_chars: int = 1200, min_chars: int = 200) -> None:
        """Create a boundary-aware text splitter.

        Args:
            max_chars: Maximum number of characters each returned chunk may
                contain. Enforced exactly; only the terminal character-level
                fallback can require slicing mid-word.
            min_chars: Minimum number of characters a returned chunk should
                contain when it can be merged with a neighbor without
                exceeding ``max_chars``. An isolated chunk (for example the
                only chunk in a short document, or one that cannot merge
                without exceeding ``max_chars``) may still fall below this
                bound; enforcement is best-effort, not absolute.

        Raises:
            ValueError: If ``max_chars`` is not a positive integer, if
                ``min_chars`` is negative, or if ``min_chars`` is not smaller
                than ``max_chars``.
        """
        if max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        if min_chars < 0:
            raise ValueError("min_chars must not be negative")
        if min_chars >= max_chars:
            raise ValueError("min_chars must be smaller than max_chars")
        self._max_chars = max_chars
        self._min_chars = min_chars

    def split(self, text: str) -> list[str]:
        """Return ``text`` split into boundary-aware chunks respecting bounds.

        Markdown heading sections that already fit within ``max_chars`` are
        kept whole, including their heading line. Oversized sections are
        split on blank-line paragraph breaks, then sentence boundaries, then
        words, then raw characters, greedily repacking each level's pieces
        up to ``max_chars`` before recursing further. Adjacent chunks under
        ``min_chars`` are then merged where the merge still fits. Blank
        input returns an empty list.
        """
        if not text.strip():
            return []
        chunks: list[str] = []
        for section in self._split_by_headings(text):
            stripped = section.strip()
            if stripped:
                chunks.extend(self._pack_section(stripped))
        return self._merge_small_chunks(chunks)

    def chunk(self, document: Document) -> list[Chunk]:
        """Return retrieval :class:`Chunk` objects for ``document``.

        Chunk ids are derived deterministically from the document id, chunk
        index, and chunk text, matching the convention used by
        :class:`ingestion.chunking.TextChunker`.
        """
        chunks: list[Chunk] = []
        for index, chunk_text in enumerate(self.split(document.text)):
            chunk_id = _stable_chunk_id(document.document_id, index, chunk_text)
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
        return chunks

    @staticmethod
    def _split_by_headings(text: str) -> list[str]:
        """Return text sections anchored at each markdown heading line."""
        matches = list(_HEADING_LINE.finditer(text))
        if not matches:
            return [text]
        sections: list[str] = []
        if matches[0].start() > 0:
            sections.append(text[: matches[0].start()])
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections.append(text[match.start() : end])
        return sections

    def _pack_section(self, section: str) -> list[str]:
        if len(section) <= self._max_chars:
            return [section]
        paragraphs = [
            " ".join(paragraph.split())
            for paragraph in _BLANK_LINE.split(section)
            if paragraph.strip()
        ]
        return self._pack_units(paragraphs, "\n\n", self._split_paragraph)

    def _split_paragraph(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self._max_chars:
            return [paragraph]
        sentences = [s for s in _SENTENCE_BOUNDARY.split(paragraph) if s]
        if len(sentences) <= 1:
            return self._split_words(paragraph)
        return self._pack_units(sentences, " ", self._split_words)

    def _split_words(self, sentence: str) -> list[str]:
        if len(sentence) <= self._max_chars:
            return [sentence]
        words = sentence.split(" ")
        if len(words) <= 1:
            return self._split_chars(sentence)
        return self._pack_units(words, " ", self._split_chars)

    def _split_chars(self, token: str) -> list[str]:
        if len(token) <= self._max_chars:
            return [token]
        return [token[i : i + self._max_chars] for i in range(0, len(token), self._max_chars)]

    def _pack_units(
        self,
        units: list[str],
        separator: str,
        split_unit: Callable[[str], list[str]],
    ) -> list[str]:
        chunks: list[str] = []
        current = ""
        for unit in units:
            for piece in split_unit(unit):
                candidate = f"{current}{separator}{piece}" if current else piece
                if len(candidate) <= self._max_chars:
                    current = candidate
                    continue
                if current:
                    chunks.append(current)
                current = piece
        if current:
            chunks.append(current)
        return chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks under ``min_chars`` into a neighbor when it still fits.

        Merging prefers the following chunk first (keeping earlier chunks at
        full size) and falls back to the previous chunk. A chunk that cannot
        be merged without exceeding ``max_chars`` is kept as-is.
        """
        if self._min_chars <= 0 or len(chunks) <= 1:
            return chunks
        merged: list[str] = []
        index = 0
        while index < len(chunks):
            current = chunks[index]
            if len(current) < self._min_chars and index + 1 < len(chunks):
                candidate = _merge_join(current, chunks[index + 1])
                if len(candidate) <= self._max_chars:
                    merged.append(candidate)
                    index += 2
                    continue
            if len(current) < self._min_chars and merged:
                candidate = _merge_join(merged[-1], current)
                if len(candidate) <= self._max_chars:
                    merged[-1] = candidate
                    index += 1
                    continue
            merged.append(current)
            index += 1
        return merged
