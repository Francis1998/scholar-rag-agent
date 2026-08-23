"""Deterministic semantic-boundary text chunking for agent-ready retrieval."""

import re
from collections.abc import Callable

from ingestion.chunking import stable_id
from retrieval.models import Chunk, Document

_HEADING_LINE = re.compile(r"(?m)^ {0,3}#{1,6}[ \t]+\S.*$")
_BLANK_LINE = re.compile(r"\n\s*\n+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class AgenticChunkBoundarySplitter:
    """Split long text on the most semantic boundary that keeps it under a limit.

    Inspired by LlamaIndex's sentence-window parsing, Chonkie's recursive
    chunkers, and LangChain's ``MarkdownHeaderTextSplitter``, this splitter
    tries boundaries from most to least semantic: markdown headings, then
    blank-line paragraph breaks, then sentence boundaries, and only falls
    back to word- or character-level cuts when a single sentence itself
    exceeds ``max_chars``. Despite the "agentic" name, no LLM or network call
    is used: it is pure, deterministic text splitting meant to feed
    agent-ready retrieval chunks.
    """

    def __init__(self, max_chars: int = 800) -> None:
        """Create a boundary-aware text splitter.

        Args:
            max_chars: Maximum number of characters each returned chunk may
                contain. Enforced exactly; only the terminal character-level
                fallback can require slicing mid-word.

        Raises:
            ValueError: If ``max_chars`` is not a positive integer.
        """
        if max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        self._max_chars = max_chars

    def split(self, text: str) -> list[str]:
        """Return ``text`` split into chunks no longer than ``max_chars``.

        Markdown heading sections that already fit within ``max_chars`` are
        kept whole, including their heading line. Oversized sections are
        split on blank-line paragraph breaks, then sentence boundaries, then
        words, then raw characters, greedily repacking each level's pieces up
        to ``max_chars`` before recursing further. Blank input returns an
        empty list.
        """
        if not text.strip():
            return []
        chunks: list[str] = []
        for section in self._split_by_headings(text):
            stripped = section.strip()
            if stripped:
                chunks.extend(self._pack_section(stripped))
        return chunks

    def chunk(self, document: Document) -> list[Chunk]:
        """Return retrieval :class:`Chunk` objects for ``document``.

        Chunk ids are derived deterministically from the document id, chunk
        index, and chunk text, matching the convention used by
        :class:`ingestion.chunking.TextChunker`.
        """
        chunks: list[Chunk] = []
        for index, chunk_text in enumerate(self.split(document.text)):
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
