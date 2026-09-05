"""Gate that re-ranks results by exponential publication-age decay."""

import math
from datetime import UTC, date, datetime

from retrieval.models import SearchResult

_DATE_FIELDS = ("published_at", "year", "date")


class TimeDecayGate:
    """Multiply relevance by exponential age decay, then re-rank.

    Inspired by Haystack/Elasticsearch temporal decay postprocessors and
    LlamaIndex freshness rerankers. Reads publication time from
    ``metadata["published_at"]``, ``metadata["year"]``, or ``metadata["date"]``
    (priority order). Decay factor:

    ```text
    decay = 0.5 ** (age_days / half_life_days)
    new_score = old_score * decay
    ```

    Missing or malformed dates receive ``decay = 0.0``. Future dates are
    treated as age ``0``. Distinct from :class:`~retrieval.freshness.FreshnessBooster`
    (blended normalized relevance) and
    :class:`~retrieval.recency_half_life.RecencyHalfLifeBooster` (year-only
    alpha blend). Inputs are not mutated. Local postprocessor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(
        self,
        half_life_days: float = 365.0,
        as_of: date | datetime | None = None,
    ) -> None:
        """Create a time-decay gate.

        Args:
            half_life_days: Days for the decay multiplier to halve.
            as_of: Reference instant for age calculations. Pin for reproducibility.

        Raises:
            ValueError: If ``half_life_days`` is non-finite or not positive.
        """
        if not math.isfinite(half_life_days) or half_life_days <= 0:
            raise ValueError("half_life_days must be a positive finite number")
        self._half_life_days = half_life_days
        self._as_of = self._as_utc_datetime(as_of or datetime.now(UTC))

    def gate(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored by age decay and sorted descending."""
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        rescored: list[SearchResult] = []
        for result in results:
            published_at = self._publication_datetime(result.chunk.metadata)
            decay = self._decay_factor(published_at)
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=result.score * decay,
                    retriever="time_decay_gate",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _decay_factor(self, published_at: datetime | None) -> float:
        if published_at is None:
            return 0.0
        age_days = max((self._as_of - published_at).total_seconds() / 86_400.0, 0.0)
        return float(0.5 ** (age_days / self._half_life_days))

    @classmethod
    def _publication_datetime(cls, metadata: dict[str, str]) -> datetime | None:
        for field in _DATE_FIELDS:
            parsed = cls._parse_datetime(metadata.get(field, ""))
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _parse_datetime(cls, value: str) -> datetime | None:
        candidate = value.strip()
        if not candidate:
            return None
        if len(candidate) == 4 and candidate.isdigit():
            try:
                return datetime(int(candidate), 1, 1, tzinfo=UTC)
            except ValueError:
                return None
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed_date = date.fromisoformat(candidate)
            except ValueError:
                return None
            return cls._as_utc_datetime(parsed_date)
        return cls._as_utc_datetime(parsed)

    @staticmethod
    def _as_utc_datetime(value: date | datetime) -> datetime:
        if not isinstance(value, datetime):
            return datetime(value.year, value.month, value.day, tzinfo=UTC)
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
