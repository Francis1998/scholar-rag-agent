"""Gate that boosts or filters hits by source authority / venue tier."""

import math
from collections.abc import Mapping

from retrieval.models import SearchResult

_DEFAULT_TIERS: dict[str, float] = {
    "high": 1.0,
    "h": 1.0,
    "medium": 0.5,
    "med": 0.5,
    "m": 0.5,
    "low": 0.2,
    "l": 0.2,
}
_AUTHORITY_KEYS = ("source_authority", "authority", "venue_tier", "authority_tier")
_VENUE_KEYS = ("venue", "journal", "source")


class SourceAuthorityGate:
    """Boost or filter results by source authority / venue tier metadata.

    Inspired by LlamaIndex/Haystack metadata boost and filter postprocessors.
    Resolution order per chunk:

    1. Numeric ``source_authority`` / ``authority`` in ``[0.0, 1.0]``.
    2. Tier label ``high`` / ``medium`` / ``low`` from ``source_authority``,
       ``authority``, ``venue_tier``, or ``authority_tier``.
    3. Lookup ``venue`` / ``journal`` / ``source`` in the optional
       ``venue_tiers`` map (values are high/medium/low or numeric).
    4. Else unknown authority ``0.0``.

    When ``min_authority`` is set, hits below that floor are dropped. Surviving
    scores are blended:

    ```text
    new_score = (1 - alpha) * old + alpha * authority
    ```

    Distinct from :class:`~retrieval.authority_boost.AuthorityBooster` (soft
    venue_rank / impact_factor blend, no filter) and
    :class:`~retrieval.venue_tier_boost.VenueTierBooster` (tier1/2/3 prestige).
    Inputs are not mutated. Local postprocessor for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        min_authority: float | None = None,
        venue_tiers: Mapping[str, str | float] | None = None,
    ) -> None:
        """Create a source-authority gate.

        Args:
            alpha: Weight for the authority signal in ``[0.0, 1.0]``.
            min_authority: Optional inclusive floor; hits below are dropped.
            venue_tiers: Optional venue/journal name → high/medium/low or float.

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        if min_authority is not None and (
            not math.isfinite(min_authority) or not 0.0 <= min_authority <= 1.0
        ):
            raise ValueError("min_authority must be a finite number within [0.0, 1.0]")
        self._alpha = alpha
        self._min_authority = min_authority
        self._venue_tiers = self._normalize_venue_tiers(venue_tiers or {})

    def gate(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return authority-boosted results, optionally filtered and truncated."""
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        rescored: list[SearchResult] = []
        for result in results:
            authority = self._authority_score(result.chunk.metadata)
            if self._min_authority is not None and authority < self._min_authority:
                continue
            score = (1.0 - self._alpha) * result.score + self._alpha * authority
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="source_authority_gate",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _authority_score(self, metadata: dict[str, str]) -> float:
        for key in _AUTHORITY_KEYS:
            raw = metadata.get(key, "").strip()
            if not raw:
                continue
            parsed = self._parse_authority_token(raw)
            if parsed is not None:
                return parsed

        for key in _VENUE_KEYS:
            name = metadata.get(key, "").strip().casefold()
            if name and name in self._venue_tiers:
                return self._venue_tiers[name]
        return 0.0

    @staticmethod
    def _parse_authority_token(raw: str) -> float | None:
        label = raw.casefold()
        if label in _DEFAULT_TIERS:
            return _DEFAULT_TIERS[label]
        try:
            value = float(raw)
        except ValueError:
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value

    @classmethod
    def _normalize_venue_tiers(
        cls,
        venue_tiers: Mapping[str, str | float],
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for name, tier in venue_tiers.items():
            key = name.strip().casefold()
            if not key:
                continue
            if isinstance(tier, (int, float)):
                value = float(tier)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("venue_tiers numeric values must be within [0.0, 1.0]")
                normalized[key] = value
                continue
            parsed = cls._parse_authority_token(str(tier))
            if parsed is None:
                raise ValueError("venue_tiers values must be high/medium/low or [0.0, 1.0]")
            normalized[key] = parsed
        return normalized
