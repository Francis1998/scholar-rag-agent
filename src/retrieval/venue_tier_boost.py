"""Deterministic venue-tier re-scoring for hybrid retrieval results."""

import math
from collections.abc import Mapping

from retrieval.models import SearchResult

_TIER_SCORES = {
    "tier1": 1.0,
    "tier2": 0.7,
    "tier3": 0.4,
}
_UNKNOWN_SCORE = 0.2

# Small built-in prestige list (casefolded display names → tier label).
_BUILTIN_VENUE_TIERS: dict[str, str] = {
    "nature": "tier1",
    "science": "tier1",
    "cell": "tier1",
    "nejm": "tier1",
    "new england journal of medicine": "tier1",
    "lancet": "tier1",
    "the lancet": "tier1",
    "jama": "tier1",
    "journal of the american medical association": "tier1",
}


class VenueTierBooster:
    """Re-score results by blending relevance with venue prestige tiers.

    Inspired by LlamaIndex/Haystack metadata boost postprocessors that promote
    higher-trust sources. Venue signals are read from ``chunk.metadata`` keys
    ``venue``, ``journal``, or ``venue_tier``. Tier labels map to scores
    ``tier1=1.0``, ``tier2=0.7``, ``tier3=0.4``, and unknown ``0.2``. An
    optional constructor ``venue_tiers`` map overlays the built-in prestige
    list (Nature, Science, Cell, NEJM, Lancet, JAMA → tier1). The blended
    score is:

    ```text
    new_score = (1 - alpha) * old + alpha * tier_score
    ```

    Results are re-sorted by ``new_score`` descending; equal scores retain
    input order (stable sort). Input objects are not mutated.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        venue_tiers: Mapping[str, str] | None = None,
    ) -> None:
        """Create a venue-tier booster.

        Args:
            alpha: Weight assigned to the tier score in ``[0.0, 1.0]``. The
                complementary weight ``1 - alpha`` is applied to the previous
                score.
            venue_tiers: Optional mapping of venue/journal display names to
                ``tier1`` / ``tier2`` / ``tier3``. Keys are matched
                case-insensitively and overlay the built-in prestige list.

        Raises:
            ValueError: If ``alpha`` is invalid or a tier label is unknown.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        merged = dict(_BUILTIN_VENUE_TIERS)
        if venue_tiers:
            for name, tier in venue_tiers.items():
                label = tier.strip().lower()
                if label not in _TIER_SCORES:
                    raise ValueError("venue_tiers values must be 'tier1', 'tier2', or 'tier3'")
                merged[name.strip().casefold()] = label
        self._alpha = alpha
        self._venue_tiers = merged

    def boost(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored and ordered by blended venue tier.

        Resolution order per chunk:

        1. ``metadata["venue_tier"]`` when it is ``tier1`` / ``tier2`` /
           ``tier3``.
        2. Else look up ``metadata["venue"]`` or ``metadata["journal"]`` in the
           merged venue map.
        3. Else unknown score ``0.2``.

        ``top_k`` truncates after sorting; ``None`` keeps every result.
        """
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        rescored: list[SearchResult] = []
        for result in results:
            tier_score = self._tier_score(result.chunk.metadata)
            score = (1.0 - self._alpha) * result.score + self._alpha * tier_score
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="venue_tier_boost",
                    path=[*result.path, result.retriever],
                )
            )
        # Stable sort: Python's sorted is stable, so equal scores keep input order.
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]

    def _tier_score(self, metadata: dict[str, str]) -> float:
        explicit = metadata.get("venue_tier", "").strip().lower()
        if explicit in _TIER_SCORES:
            return _TIER_SCORES[explicit]

        for key in ("venue", "journal"):
            name = metadata.get(key, "").strip().casefold()
            if not name:
                continue
            tier = self._venue_tiers.get(name)
            if tier is not None:
                return _TIER_SCORES[tier]
        return _UNKNOWN_SCORE
