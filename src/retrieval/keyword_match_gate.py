"""Gate retrieval hits by query-keyword coverage fraction."""

import math
import re

from retrieval.models import SearchResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class KeywordMatchGate:
    """Keep hits covering at least ``min_coverage`` of query tokens.

    Inspired by Elasticsearch/Haystack ``minimum_should_match`` keyword
    filters. Coverage is ``|query ∩ doc| / |query|`` (empty query → keep all).
    Distinct from :class:`~retrieval.term_coverage_boost.TermCoverageBooster`
    (soft boost) and :class:`~retrieval.cross_encoder_gate.CrossEncoderGate`.
    Inputs are not mutated. Local postprocessor for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, min_coverage: float = 0.5) -> None:
        """Create a keyword-match gate.

        Args:
            min_coverage: Inclusive coverage floor in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``min_coverage`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(min_coverage) or not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be a finite number within [0.0, 1.0]")
        self._min_coverage = min_coverage

    def gate(
        self,
        results: list[SearchResult],
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results whose keyword coverage meets ``min_coverage``."""
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        query_tokens = self._tokens(query)
        kept: list[SearchResult] = []
        for result in results:
            coverage = self._coverage(query_tokens, self._tokens(result.chunk.text))
            if coverage < self._min_coverage:
                continue
            kept.append(
                SearchResult(
                    chunk=result.chunk,
                    score=coverage,
                    retriever="keyword_match_gate",
                    path=[*result.path, result.retriever],
                )
            )
            if len(kept) >= limit:
                break
        return kept

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_RE.findall(text)}

    @staticmethod
    def _coverage(query_tokens: set[str], doc_tokens: set[str]) -> float:
        if not query_tokens:
            return 1.0
        return len(query_tokens & doc_tokens) / len(query_tokens)
