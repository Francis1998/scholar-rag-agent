"""Deterministic query-term alignment to retrieved evidence text."""

import re
from collections.abc import Collection
from dataclasses import dataclass

from retrieval.models import SearchResult
from retrieval.sparse import STOPWORDS

_TERM_PATTERN = re.compile(r"(?<!\w)\w+(?:[-']\w+)*(?!\w)", re.UNICODE)


@dataclass(frozen=True)
class EvidenceSpan:
    """Half-open character span for one normalized query term."""

    start: int
    end: int
    term: str


@dataclass(frozen=True)
class EvidenceSpanAlignment:
    """Evidence spans associated with one retrieved result."""

    result: SearchResult
    spans: tuple[EvidenceSpan, ...]


class EvidenceSpanAligner:
    """Locate exact query terms in result text without changing the text."""

    def __init__(self, stopwords: Collection[str] | None = None) -> None:
        """Create an aligner with shared or caller-provided stopwords."""
        words = STOPWORDS if stopwords is None else stopwords
        self._stopwords = frozenset(word.casefold() for word in words)

    def align_text(self, query: str, text: str) -> tuple[EvidenceSpan, ...]:
        """Return ordered, repeated term spans into the original ``text``.

        Matching is case-insensitive and token-exact. Offsets use Python's
        half-open string convention, so ``text[span.start:span.end]`` recovers
        the evidence exactly as written.
        """
        query_terms = {
            match.group(0).casefold()
            for match in _TERM_PATTERN.finditer(query)
            if match.group(0).casefold() not in self._stopwords
        }
        if not query_terms:
            return ()

        return tuple(
            EvidenceSpan(
                start=match.start(),
                end=match.end(),
                term=normalized,
            )
            for match in _TERM_PATTERN.finditer(text)
            if (normalized := match.group(0).casefold()) in query_terms
        )

    def align(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[EvidenceSpanAlignment]:
        """Return one alignment per result, preserving retrieval order."""
        return [
            EvidenceSpanAlignment(
                result=result,
                spans=self.align_text(query, result.chunk.text),
            )
            for result in results
        ]
