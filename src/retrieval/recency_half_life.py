"""Boost retrieval hits by publication-year half-life decay."""

import math

from retrieval.models import SearchResult

_YEAR_KEYS = ("year", "publication_year", "published_year")


class RecencyHalfLifeBooster:
    """Blend relevance with exponential publication-year decay.

    Inspired by temporal decay ranking in Haystack/Elasticsearch. Reads year
    from ``chunk.metadata`` keys ``year``, ``publication_year``, or
    ``published_year``. Recency signal:

    ```text
    recency = 0.5 ** ((ref_year - year) / half_life_years)
    ```

    clamped to ``[0.0, 1.0]``. Missing year → ``recency = 0.0``. Blended score:

    ```text
    new_score = (1 - alpha) * old + alpha * recency
    ```

    Results are re-sorted descending (stable). Inputs are not mutated. Local
    retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(
        self,
        half_life_years: float = 5.0,
        ref_year: int = 2026,
        alpha: float = 0.3,
    ) -> None:
        """Create a recency half-life booster.

        Args:
            half_life_years: Years for the recency signal to halve.
            ref_year: Reference publication year for age calculations.
            alpha: Weight for the recency signal in ``[0.0, 1.0]``.

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        if not math.isfinite(half_life_years) or half_life_years <= 0:
            raise ValueError("half_life_years must be a positive finite number")
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        self._half_life_years = half_life_years
        self._ref_year = ref_year
        self._alpha = alpha

    def boost(
        self,
        results: list[SearchResult],
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored by publication-year half-life decay."""
        del query  # API parity with other boosters; recency is metadata-only.
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []
        rescored: list[SearchResult] = []
        for result in results:
            recency = self._recency_score(result.chunk.metadata)
            score = (1.0 - self._alpha) * result.score + self._alpha * recency
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="recency_half_life",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _recency_score(self, metadata: dict[str, str]) -> float:
        year = self._publication_year(metadata)
        if year is None:
            return 0.0
        exponent = (self._ref_year - year) / self._half_life_years
        recency = 0.5**exponent
        return float(min(max(recency, 0.0), 1.0))

    @staticmethod
    def _publication_year(metadata: dict[str, str]) -> int | None:
        for key in _YEAR_KEYS:
            raw = metadata.get(key, "").strip()
            if not raw:
                continue
            candidate = raw[:4] if len(raw) >= 4 and raw[:4].isdigit() else raw
            if candidate.isdigit() and len(candidate) == 4:
                try:
                    return int(candidate)
                except ValueError:
                    continue
        return None
