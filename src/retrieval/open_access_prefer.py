"""Prefer open-access retrieval hits via boost or soft filter."""

import math

from retrieval.models import SearchResult

_OA_KEYS = ("open_access", "is_oa", "oa")
_TRUTHY = frozenset({"true", "1", "yes", "y", "oa", "open", "open_access"})


class OpenAccessPreferencer:
    """Prefer open-access results with boost blending or soft filtering.

    Inspired by LlamaIndex/Haystack metadata postprocessors and Unpaywall /
    OpenAlex open-access preference in scholarly search — but implemented as a
    local retrieval postprocessor (not a DOI connector). Open-access is read
    from truthy ``chunk.metadata`` values under ``open_access``, ``is_oa``, or
    ``oa``.

    * ``mode="boost"`` (default): blend prior score with an OA signal of
      ``1.0`` / ``0.0`` using ``alpha``.
    * ``mode="filter"``: when at least one hit is OA, drop non-OA rows; if
      none are OA, keep every input row.

    Boost mode re-sorts stably by blended score and does not mutate inputs.
    Filter mode preserves original identity, score, and order.
    """

    def __init__(self, alpha: float = 0.3, mode: str = "boost") -> None:
        """Create an open-access preferencer.

        Args:
            alpha: Weight assigned to the OA signal in boost mode, in
                ``[0.0, 1.0]``. Ignored for filter mode scoring (filter does
                not re-score).
            mode: ``"boost"`` to blend scores, or ``"filter"`` to soft-filter
                non-OA rows when any OA hit exists.

        Raises:
            ValueError: If ``alpha`` is invalid or ``mode`` is unsupported.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        if mode not in {"boost", "filter"}:
            raise ValueError("mode must be 'boost' or 'filter'")
        self._alpha = alpha
        self._mode = mode

    def prefer(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results preferred toward open-access evidence.

        In boost mode, each result receives
        ``(1 - alpha) * old + alpha * oa_signal`` and is sorted descending
        (stable for ties). In filter mode, non-OA rows are dropped only when
        the batch contains at least one OA hit; otherwise the full input is
        kept. ``top_k`` truncates after preference; ``None`` keeps every
        survivor.
        """
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        if self._mode == "filter":
            return self._filter(results, limit)
        return self._boost(results, limit)

    def _boost(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        rescored: list[SearchResult] = []
        for result in results:
            oa_signal = 1.0 if self._is_open_access(result.chunk.metadata) else 0.0
            score = (1.0 - self._alpha) * result.score + self._alpha * oa_signal
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="open_access_prefer",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]

    def _filter(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        oa_hits = [result for result in results if self._is_open_access(result.chunk.metadata)]
        survivors = oa_hits if oa_hits else list(results)
        return survivors[:limit]

    @staticmethod
    def _is_open_access(metadata: dict[str, str]) -> bool:
        for key in _OA_KEYS:
            raw = metadata.get(key, "").strip().lower()
            if raw in _TRUTHY:
                return True
        return False
