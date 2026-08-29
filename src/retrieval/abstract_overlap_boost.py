"""Boost retrieval hits by query↔abstract token overlap."""

import math
import re

from retrieval.models import SearchResult

_ABSTRACT_KEYS = ("abstract", "summary", "abstract_text")
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}", re.IGNORECASE)


class AbstractOverlapBooster:
    """Blend relevance with query/abstract lexical overlap.

    Inspired by LlamaIndex/Haystack keyword-overlap postprocessors. Reads
    abstract text from ``chunk.metadata`` keys ``abstract``, ``summary``, or
    ``abstract_text``. Overlap is Jaccard over casefolded alphanumeric tokens
    of length >= 2. Blended score:

    ```text
    new_score = (1 - alpha) * old + alpha * jaccard
    ```

    Results are re-sorted descending (stable). Inputs are not mutated. Local
    retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create an abstract-overlap booster.

        Args:
            alpha: Weight for the overlap signal in ``[0.0, 1.0]``.

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
        """Return results re-scored by abstract overlap with ``query``."""
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []
        query_tokens = self._tokens(query)
        rescored: list[SearchResult] = []
        for result in results:
            overlap = self._jaccard(query_tokens, self._abstract_tokens(result.chunk.metadata))
            score = (1.0 - self._alpha) * result.score + self._alpha * overlap
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="abstract_overlap_boost",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _abstract_tokens(self, metadata: dict[str, str]) -> set[str]:
        for key in _ABSTRACT_KEYS:
            raw = metadata.get(key, "")
            if raw.strip():
                return self._tokens(raw)
        return set()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        return len(left & right) / len(union)
