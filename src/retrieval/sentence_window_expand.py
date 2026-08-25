"""Expand retrieved chunk text with neighboring sentences from full document text."""

import re

from retrieval.models import SearchResult

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class SentenceWindowExpander:
    """Expand each hit with ±N neighboring sentences from full document text.

    Inspired by LlamaIndex ``SentenceWindowNodeParser`` / sentence-window
    retrieval. Unlike :class:`ParentDocumentExpander`, which swaps a child hit
    for an entire parent document from an external store, this expander only
    widens the local sentence window around the retrieved span when
    ``document_text`` or ``full_text`` is present in chunk metadata. Unlike
    :class:`ContextualCompressor`, which *shrinks* text to query-relevant
    sentences, this stage *adds* neighboring context. Input objects are never
    mutated; copies are always returned.
    """

    def __init__(self, window_sentences: int = 1) -> None:
        """Create a sentence-window expander.

        Args:
            window_sentences: Number of sentences to include on each side of
                the retrieved span. Inclusive bound ``0`` leaves text unchanged
                when a full-document source is available; the maximum is ``10``.

        Raises:
            ValueError: If ``window_sentences`` is not an integer in ``0..10``.
        """
        if (
            not isinstance(window_sentences, int)
            or isinstance(window_sentences, bool)
            or not 0 <= window_sentences <= 10
        ):
            raise ValueError("window_sentences must be an integer within 0..10")
        self._window_sentences = window_sentences

    def expand(self, results: list[SearchResult]) -> list[SearchResult]:
        """Return copies of ``results`` with optionally widened chunk text.

        For each result, read full document text from chunk metadata key
        ``document_text``, falling back to ``full_text``. When neither is set,
        or the chunk text cannot be located inside the document, the returned
        copy keeps the original chunk text. Score, retriever, and path are
        preserved; ``retriever`` is set to ``sentence_window`` and the prior
        retriever is appended to ``path`` only when the text actually expands.
        """
        expanded: list[SearchResult] = []
        for result in results:
            new_text = self._expand_text(result.chunk.text, result.chunk.metadata)
            if new_text == result.chunk.text:
                expanded.append(
                    SearchResult(
                        chunk=result.chunk.model_copy(deep=True),
                        score=result.score,
                        retriever=result.retriever,
                        path=list(result.path),
                    )
                )
                continue
            expanded.append(
                SearchResult(
                    chunk=result.chunk.model_copy(deep=True, update={"text": new_text}),
                    score=result.score,
                    retriever="sentence_window",
                    path=[*result.path, result.retriever],
                )
            )
        return expanded

    def _expand_text(self, chunk_text: str, metadata: dict[str, str]) -> str:
        document_text = metadata.get("document_text") or metadata.get("full_text") or ""
        if not document_text.strip() or self._window_sentences == 0:
            return chunk_text

        sentences = self._split_sentences(document_text)
        if not sentences:
            return chunk_text

        span = self._locate_span(chunk_text, document_text, sentences)
        if span is None:
            return chunk_text

        start, end = span
        window_start = max(0, start - self._window_sentences)
        window_end = min(len(sentences), end + self._window_sentences)
        return " ".join(sentences[window_start:window_end])

    @classmethod
    def _locate_span(
        cls,
        chunk_text: str,
        document_text: str,
        sentences: list[str],
    ) -> tuple[int, int] | None:
        """Return half-open sentence indices covering ``chunk_text`` in ``document_text``."""
        needle = chunk_text.strip()
        if not needle:
            return None

        # Prefer exact substring location in the full document.
        index = document_text.find(needle)
        if index >= 0:
            end_index = index + len(needle)
            return cls._span_for_char_range(document_text, sentences, index, end_index)

        # Fall back to locating the chunk's own sentences inside the document.
        chunk_sentences = cls._split_sentences(needle)
        if not chunk_sentences:
            return None
        for start in range(len(sentences) - len(chunk_sentences) + 1):
            window = sentences[start : start + len(chunk_sentences)]
            if window == chunk_sentences:
                return start, start + len(chunk_sentences)
        return None

    @staticmethod
    def _span_for_char_range(
        document_text: str,
        sentences: list[str],
        start_char: int,
        end_char: int,
    ) -> tuple[int, int] | None:
        # Rebuild sentence character offsets by scanning the document in order.
        cursor = 0
        offsets: list[tuple[int, int]] = []
        for sentence in sentences:
            found = document_text.find(sentence, cursor)
            if found < 0:
                return None
            offsets.append((found, found + len(sentence)))
            cursor = found + len(sentence)

        start_i: int | None = None
        end_i: int | None = None
        for index, (sent_start, sent_end) in enumerate(offsets):
            overlaps = sent_end > start_char and sent_start < end_char
            if overlaps:
                if start_i is None:
                    start_i = index
                end_i = index + 1
        if start_i is None or end_i is None:
            return None
        return start_i, end_i

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        return [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(normalized)
            if sentence.strip()
        ]
