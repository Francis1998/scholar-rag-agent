"""Boost retrieval hits by intra-chunk coherence with query continuity."""

import math
import re

from retrieval.models import SearchResult
from retrieval.sparse import tokenize

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class CoherenceBooster:
    """Blend relevance with sentence-neighbor and query-term continuity.

    Inspired by LlamaIndex/Haystack coherence-aware rerankers that prefer
    passages whose adjacent sentences share vocabulary and whose query terms
    appear consistently across the chunk (not a DOI connector). Coherence is
    the mean of:

    * **neighbor overlap** — mean Jaccard similarity between token sets of
      adjacent sentences (``0.0`` when fewer than two sentences);
    * **query continuity** — fraction of sentences that contain at least one
      query token (``0.0`` when the query has no tokens).

    Blended score:

    ```text
    new_score = (1 - alpha) * old + alpha * coherence
    ```

    Results are re-sorted descending (stable). Inputs are not mutated. Local
    retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create a coherence booster.

        Args:
            alpha: Weight for the coherence signal in ``[0.0, 1.0]``.

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
        """Return results re-scored by intra-chunk coherence with ``query``."""
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []
        query_tokens = set(tokenize(query))
        rescored: list[SearchResult] = []
        for result in results:
            coherence = self._coherence(result.chunk.text, query_tokens)
            score = (1.0 - self._alpha) * result.score + self._alpha * coherence
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="coherence_boost",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _coherence(self, text: str, query_tokens: set[str]) -> float:
        sentences = self._sentences(text)
        if not sentences:
            return 0.0
        token_sets = [set(tokenize(sentence)) for sentence in sentences]
        neighbor = self._neighbor_overlap(token_sets)
        continuity = self._query_continuity(token_sets, query_tokens)
        return 0.5 * neighbor + 0.5 * continuity

    @staticmethod
    def _sentences(text: str) -> list[str]:
        parts = _SENTENCE_SPLIT_RE.split(text.strip())
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _neighbor_overlap(token_sets: list[set[str]]) -> float:
        if len(token_sets) < 2:
            return 0.0
        overlaps = [
            CoherenceBooster._jaccard(token_sets[index], token_sets[index + 1])
            for index in range(len(token_sets) - 1)
        ]
        return sum(overlaps) / len(overlaps)

    @staticmethod
    def _query_continuity(token_sets: list[set[str]], query_tokens: set[str]) -> float:
        if not query_tokens or not token_sets:
            return 0.0
        hits = sum(1 for tokens in token_sets if tokens & query_tokens)
        return hits / len(token_sets)

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        return len(left & right) / len(union)
