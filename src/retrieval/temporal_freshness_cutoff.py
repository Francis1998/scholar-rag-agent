"""Deterministic hard age cutoff for stale retrieval evidence."""

import math
from datetime import UTC, date, datetime

from retrieval.models import SearchResult

_DATE_FIELDS = ("published_at", "year", "date")


class TemporalFreshnessCutoff:
    """Drop chunks older than a maximum age before they reach synthesis.

    Inspired by LlamaIndex's ``TemporalRetriever`` recency filter and
    Haystack-style metadata freshness filters. Unlike :class:`FreshnessBooster`,
    which continuously re-scores and re-ranks results by recency, this gate is
    a hard boolean cutoff: a result either passes unmodified or is dropped.
    Relative order and scores of surviving results are preserved.
    """

    def __init__(
        self,
        max_age_days: float,
        keep_undated: bool = True,
        as_of: date | datetime | None = None,
    ) -> None:
        """Create a temporal freshness cutoff.

        Args:
            max_age_days: Maximum inclusive age, in days, a dated chunk may
                have relative to ``as_of`` before it is dropped.
            keep_undated: Whether chunks with no parseable publication date
                are kept. Defaults to ``True`` since undated evidence should
                not be silently discarded by default.
            as_of: Reference date for age calculations. Pin this for
                reproducible runs; defaults to the current UTC time.

        Raises:
            ValueError: If ``max_age_days`` is non-finite or not positive.
        """
        if not math.isfinite(max_age_days) or max_age_days <= 0:
            raise ValueError("max_age_days must be a positive finite number")
        self._max_age_days = max_age_days
        self._keep_undated = keep_undated
        self._as_of = self._as_utc_datetime(as_of or datetime.now(UTC))

    def filter(self, results: list[SearchResult]) -> list[SearchResult]:
        """Return ``results`` with chunks older than ``max_age_days`` dropped.

        Publication dates are read, in priority order, from the
        ``published_at``, ``year``, and ``date`` chunk metadata fields.
        Results without a parseable date are kept or dropped according to
        ``keep_undated``. Input order and scores are preserved for kept
        results; no result object is mutated or copied.
        """
        kept: list[SearchResult] = []
        for result in results:
            published_at = self._publication_datetime(result.chunk.metadata)
            if published_at is None:
                if self._keep_undated:
                    kept.append(result)
                continue
            age_days = max((self._as_of - published_at).total_seconds() / 86_400.0, 0.0)
            if age_days <= self._max_age_days:
                kept.append(result)
        return kept

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
