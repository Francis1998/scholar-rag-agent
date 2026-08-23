"""Deterministic lexical groundedness scoring for inline citation markers."""

import math
import re
from dataclasses import dataclass

from retrieval.models import SearchResult
from retrieval.sparse import meaningful_terms

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_BRACKET_CITATION = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_AUTHOR_YEAR_CITATION = re.compile(
    r"\(\s*[A-Z][\w'-]+"
    r"(?:\s+(?:and|&)\s+[A-Z][\w'-]+|\s+et\s+al\.?)?"
    r"\s*,\s*\d{4}[a-z]?\s*\)"
)
_AUTHOR_YEAR_CONNECTOR = re.compile(r"\s+(?:et\s+al\.?|and|&)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class CitationMention:
    """One citation marker found in a sentence and its resolved support."""

    marker: str
    sentence: str
    candidate_chunk_ids: tuple[str, ...]
    grounded: bool
    overlap_score: float


@dataclass(frozen=True)
class CitationGroundednessReport:
    """Per-citation grounding decisions and aggregate groundedness."""

    citations: tuple[CitationMention, ...]
    groundedness: float
    grounded_count: int
    ungrounded_count: int


class CitationGroundednessScorer:
    """Score whether inline citation markers lexically match cited evidence.

    Inspired by RAGAS ``context_precision`` and TruLens groundedness metrics,
    but scoped to explicit citation markers rather than whole claim
    sentences. ``[n]`` numeric markers resolve to the ``n``-th retrieved
    result (1-indexed); ``(Author, Year)`` markers resolve by matching the
    ``authors`` and ``year`` chunk metadata populated by ingestion
    connectors. This is distinct from :class:`ClaimVerificationGate`, which
    splits every sentence into a claim and checks lexical overlap against
    *any* retrieved chunk regardless of which source, if any, the sentence
    actually cites.
    """

    def __init__(self, overlap_threshold: float = 0.3) -> None:
        """Create a citation groundedness scorer.

        Args:
            overlap_threshold: Inclusive fraction of sentence terms that must
                overlap the cited chunk's title and text for a citation to
                count as grounded.

        Raises:
            ValueError: If ``overlap_threshold`` is non-finite or outside
                ``[0.0, 1.0]``.
        """
        if not math.isfinite(overlap_threshold) or not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be a finite number within [0.0, 1.0]")
        self._overlap_threshold = overlap_threshold

    def score(self, answer: str, results: list[SearchResult]) -> CitationGroundednessReport:
        """Return per-citation grounding decisions and overall groundedness.

        Sentences are split on ``.``, ``!``, ``?``, or newlines. Every ``[n]``
        or ``(Author, Year)`` marker in a sentence is resolved to zero or more
        candidate chunks, and lexical overlap between the marker-stripped
        sentence and each candidate's title/text is measured. A citation is
        ``grounded`` when it resolves to at least one candidate, the sentence
        has content terms, and the best overlap meets ``overlap_threshold``.
        Sentences without citation markers do not contribute to the report.
        """
        mentions: list[CitationMention] = []
        for sentence in self._split_sentences(answer):
            for marker, candidate_ids in self._find_markers(sentence, results):
                mentions.append(self._score_marker(marker, sentence, candidate_ids, results))

        if not mentions:
            return CitationGroundednessReport(
                citations=(),
                groundedness=0.0,
                grounded_count=0,
                ungrounded_count=0,
            )

        grounded_count = sum(1 for mention in mentions if mention.grounded)
        ungrounded_count = len(mentions) - grounded_count
        return CitationGroundednessReport(
            citations=tuple(mentions),
            groundedness=grounded_count / len(mentions),
            grounded_count=grounded_count,
            ungrounded_count=ungrounded_count,
        )

    def _score_marker(
        self,
        marker: str,
        sentence: str,
        candidate_ids: tuple[str, ...],
        results: list[SearchResult],
    ) -> CitationMention:
        sentence_terms = meaningful_terms(self._strip_markers(sentence))
        chunks_by_id = {result.chunk.chunk_id: result for result in results}
        best_score = 0.0
        if sentence_terms:
            for chunk_id in candidate_ids:
                result = chunks_by_id.get(chunk_id)
                if result is None:
                    continue
                chunk_terms = meaningful_terms(f"{result.chunk.title} {result.chunk.text}")
                overlap = len(sentence_terms & chunk_terms) / len(sentence_terms)
                best_score = max(best_score, overlap)
        grounded = (
            bool(candidate_ids) and bool(sentence_terms) and best_score >= self._overlap_threshold
        )
        return CitationMention(
            marker=marker,
            sentence=sentence,
            candidate_chunk_ids=candidate_ids,
            grounded=grounded,
            overlap_score=best_score,
        )

    @staticmethod
    def _find_markers(
        sentence: str, results: list[SearchResult]
    ) -> list[tuple[str, tuple[str, ...]]]:
        markers: list[tuple[str, tuple[str, ...]]] = []
        for match in _BRACKET_CITATION.finditer(sentence):
            indices = [
                int(piece.strip())
                for piece in match.group(0).strip("[]").split(",")
                if piece.strip()
            ]
            chunk_ids = tuple(
                results[index - 1].chunk.chunk_id for index in indices if 1 <= index <= len(results)
            )
            markers.append((match.group(0), chunk_ids))
        for match in _AUTHOR_YEAR_CITATION.finditer(sentence):
            author, year = CitationGroundednessScorer._split_author_year(match.group(0))
            chunk_ids = tuple(
                result.chunk.chunk_id
                for result in results
                if CitationGroundednessScorer._matches_author_year(result, author, year)
            )
            markers.append((match.group(0), chunk_ids))
        return markers

    @staticmethod
    def _split_author_year(marker: str) -> tuple[str, str]:
        inner = marker.strip("()")
        author, _, year = inner.rpartition(",")
        return author.strip(), year.strip()

    @staticmethod
    def _matches_author_year(result: SearchResult, author: str, year: str) -> bool:
        result_year = result.chunk.metadata.get("year", "").strip()
        if len(result_year) < 4 or result_year[:4] != year[:4]:
            return False
        authors_field = result.chunk.metadata.get("authors", "")
        if not authors_field:
            return False
        surname = _AUTHOR_YEAR_CONNECTOR.split(author, maxsplit=1)[0].strip()
        return bool(surname) and surname.casefold() in authors_field.casefold()

    @staticmethod
    def _strip_markers(sentence: str) -> str:
        without_brackets = _BRACKET_CITATION.sub(" ", sentence)
        return _AUTHOR_YEAR_CITATION.sub(" ", without_brackets)

    @staticmethod
    def _split_sentences(answer: str) -> list[str]:
        normalized = " ".join(answer.split())
        if not normalized:
            return []
        return [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(normalized)
            if sentence.strip()
        ]
