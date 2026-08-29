"""Soft-boost retrieval hits by preferred author-count bands."""

import math
import re

from retrieval.models import SearchResult

_AUTHOR_KEYS = ("author_count", "num_authors", "authors")
_AUTHOR_SPLIT = re.compile(r"\s*;\s*|\s*,\s*|\s+and\s+", re.IGNORECASE)


class AuthorCountBooster:
    """Blend relevance with a soft preference for mid-sized author lists.

    Inspired by bibliometric priors used in Haystack/LlamaIndex metadata
    boosts. Author count is read from ``author_count`` / ``num_authors``
    (int-like) or inferred by splitting ``authors`` on commas/semicolons/
    ``and``. Counts in ``[min_authors, max_authors]`` (default 2..12) score
    ``1.0``; others score ``0.2``. Local postprocessor (not a DOI connector)
    for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        min_authors: int = 2,
        max_authors: int = 12,
    ) -> None:
        """Create an author-count booster.

        Args:
            alpha: Weight for the author-count signal in ``[0.0, 1.0]``.
            min_authors: Inclusive lower bound of the preferred band.
            max_authors: Inclusive upper bound of the preferred band.

        Raises:
            ValueError: If bounds/alpha are invalid.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        if min_authors < 1 or max_authors < min_authors:
            raise ValueError("author bounds must satisfy 1 <= min_authors <= max_authors")
        self._alpha = alpha
        self._min_authors = min_authors
        self._max_authors = max_authors

    def boost(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored by author-count preference."""
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []
        rescored: list[SearchResult] = []
        for result in results:
            count = self._author_count(result.chunk.metadata)
            band = (
                1.0
                if count is not None and self._min_authors <= count <= self._max_authors
                else 0.2
            )
            score = (1.0 - self._alpha) * result.score + self._alpha * band
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="author_count_boost",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _author_count(self, metadata: dict[str, str]) -> int | None:
        for key in ("author_count", "num_authors"):
            raw = metadata.get(key, "").strip()
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError:
                continue
            if value >= 0:
                return value
        authors = metadata.get("authors", "").strip()
        if not authors:
            return None
        parts = [part for part in _AUTHOR_SPLIT.split(authors) if part.strip()]
        return len(parts) if parts else None
