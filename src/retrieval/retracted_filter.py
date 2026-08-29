"""Filter or soft-demote retracted scholarly hits."""

import math

from retrieval.models import SearchResult

_STATUS_KEYS = ("retraction_status", "status", "publication_status")
_FLAG_KEYS = ("retracted", "is_retracted")
_RETRACTED_MARKERS = frozenset({"retracted", "retraction", "withdrawn", "true", "1", "yes"})


class RetractedFilter:
    """Drop or soft-demote retracted works via local metadata flags.

    Inspired by Retraction Watch / OpenAlex retraction signals used in
    scholarly search. Reads ``retraction_status`` / ``status`` /
    ``publication_status`` or boolean-like ``retracted`` / ``is_retracted``.

    * ``mode="filter"`` (default): remove retracted rows when any non-retracted
      survivor exists; if every row is retracted, keep inputs unchanged.
    * ``mode="demote"``: blend scores with demote signal ``0.1`` vs ``1.0``.

    Local postprocessor (not a DOI connector) for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines.
    """

    def __init__(self, alpha: float = 0.5, mode: str = "filter") -> None:
        """Create a retracted filter.

        Args:
            alpha: Demote-mode blend weight in ``[0.0, 1.0]``.
            mode: ``filter`` or ``demote``.

        Raises:
            ValueError: If ``alpha`` or ``mode`` is invalid.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        if mode not in {"filter", "demote"}:
            raise ValueError("mode must be 'filter' or 'demote'")
        self._alpha = alpha
        self._mode = mode

    def filter(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results with retracted works filtered or demoted."""
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []
        if self._mode == "demote":
            return self._demote(results, limit)
        return self._filter(results, limit)

    def _filter(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        kept = [result for result in results if not self._is_retracted(result.chunk.metadata)]
        survivors = kept if kept else list(results)
        return survivors[:limit]

    def _demote(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        rescored: list[SearchResult] = []
        for result in results:
            signal = 0.1 if self._is_retracted(result.chunk.metadata) else 1.0
            score = (1.0 - self._alpha) * result.score + self._alpha * signal
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="retracted_filter",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _is_retracted(self, metadata: dict[str, str]) -> bool:
        for key in _FLAG_KEYS:
            raw = metadata.get(key, "").strip().casefold()
            if raw in _RETRACTED_MARKERS:
                return True
        for key in _STATUS_KEYS:
            raw = metadata.get(key, "").strip().casefold()
            if any(marker in raw for marker in ("retract", "withdrawn")):
                return True
        return False
