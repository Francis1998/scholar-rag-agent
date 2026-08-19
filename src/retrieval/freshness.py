"""Deterministic relevance and publication-recency re-scoring."""

import math
from datetime import UTC, date, datetime

from retrieval.models import SearchResult

_DATE_FIELDS = ("published_at", "year", "date")


class FreshnessBooster:
    """Re-score retrieval results with exponential publication-date decay."""

    def __init__(
        self,
        half_life_days: float = 365.0,
        relevance_weight: float = 0.8,
        as_of: date | datetime | None = None,
    ) -> None:
        """Create a freshness booster.

        Args:
            half_life_days: Number of days after which the recency signal halves.
            relevance_weight: Weight assigned to normalized relevance in ``[0, 1]``.
            as_of: Reference date for age calculations. Pin this for reproducible runs.

        Raises:
            ValueError: If a scoring parameter is outside its valid range.
        """
        if not math.isfinite(half_life_days) or half_life_days <= 0:
            raise ValueError("half_life_days must be a positive finite number")
        if not math.isfinite(relevance_weight) or not 0.0 <= relevance_weight <= 1.0:
            raise ValueError("relevance_weight must be within [0.0, 1.0]")
        self._half_life_days = half_life_days
        self._relevance_weight = relevance_weight
        self._as_of = self._as_utc_datetime(as_of or datetime.now(UTC))

    def boost(self, results: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
        """Return results ordered by combined relevance and publication recency.

        Relevance scores are min-max normalized before blending. Publication dates
        are read, in priority order, from ``published_at``, ``year``, and ``date``.
        Missing or malformed dates receive no recency contribution. Ties preserve
        input order.
        """
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        relevance = self._normalized_relevance([result.score for result in results])
        rescored: list[SearchResult] = []
        for result, relevance_score in zip(results, relevance, strict=True):
            published_at = self._publication_datetime(result.chunk.metadata)
            recency_score = self._recency_score(published_at)
            score = (
                self._relevance_weight * relevance_score
                + (1.0 - self._relevance_weight) * recency_score
            )
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="freshness",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]

    def _recency_score(self, published_at: datetime | None) -> float:
        if published_at is None:
            return 0.0
        age_days = max((self._as_of - published_at).total_seconds() / 86_400.0, 0.0)
        return math.exp(-math.log(2.0) * age_days / self._half_life_days)

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

    @staticmethod
    def _normalized_relevance(scores: list[float]) -> list[float]:
        lowest = min(scores)
        highest = max(scores)
        span = highest - lowest
        if span == 0:
            return [1.0 for _ in scores]
        return [(score - lowest) / span for score in scores]
