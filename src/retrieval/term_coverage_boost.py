"""Boost retrieval hits by query-term coverage fraction."""

import math
import re

from retrieval.models import SearchResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class TermCoverageBooster:
    """Soft-boost results by the fraction of query tokens present in the chunk.

    Inspired by Haystack/Elasticsearch ``minimum_should_match`` coverage boosts
    and LlamaIndex keyword postprocessors. Blends prior relevance with coverage:

    ```text
    new_score = (1 - alpha) * old + alpha * coverage
    ```

    where ``coverage = |query ∩ doc| / |query|`` (empty query → ``1.0``).
    Distinct from :class:`~retrieval.lexical_overlap_boost.LexicalOverlapBooster`
    (symmetric Jaccard) and :class:`~retrieval.cross_encoder_gate.CrossEncoderGate`
    (hard drop). Inputs are not mutated. Local postprocessor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, alpha: float = 0.4) -> None:
        """Create a term-coverage booster.

        Args:
            alpha: Coverage blend weight in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``alpha`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        self._alpha = alpha

    def boost(
        self,
        results: list[SearchResult],
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored by query-term coverage and re-sorted."""
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        query_tokens = self._tokens(query)
        rescored: list[SearchResult] = []
        for result in results:
            coverage = self._coverage(query_tokens, self._tokens(result.chunk.text))
            new_score = (1.0 - self._alpha) * result.score + self._alpha * coverage
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=new_score,
                    retriever="term_coverage_boost",
                    path=[*result.path, result.retriever],
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:limit]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_RE.findall(text)}

    @staticmethod
    def _coverage(query_tokens: set[str], doc_tokens: set[str]) -> float:
        if not query_tokens:
            return 1.0
        return len(query_tokens & doc_tokens) / len(query_tokens)
