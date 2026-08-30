"""Boost retrieval hits by query↔chunk entity token overlap."""

import math
import re

from retrieval.models import SearchResult

_ENTITY_RE = re.compile(
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}",
)


class EntityOverlapBooster:
    """Blend relevance with query/chunk entity-token Jaccard overlap.

    Inspired by entity-centric retrieval in GraphRAG/Haystack. Extracts
    capitalized multi-word phrases and acronym-like tokens (``[A-Z]{2,}``)
    from the query and ``chunk.text``. Overlap is Jaccard over extracted
    entity tokens. Blended score:

    ```text
    new_score = (1 - alpha) * old + alpha * jaccard
    ```

    Results are re-sorted descending (stable). Inputs are not mutated. Local
    retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create an entity-overlap booster.

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
        """Return results re-scored by entity overlap with ``query``."""
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []
        query_entities = self._entities(query)
        rescored: list[SearchResult] = []
        for result in results:
            overlap = self._jaccard(query_entities, self._entities(result.chunk.text))
            score = (1.0 - self._alpha) * result.score + self._alpha * overlap
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="entity_overlap_boost",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    @classmethod
    def _entities(cls, text: str) -> set[str]:
        return {match.group(0) for match in _ENTITY_RE.finditer(text)}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        return len(left & right) / len(union)
