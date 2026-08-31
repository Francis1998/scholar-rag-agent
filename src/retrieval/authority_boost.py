"""Boost retrieval hits by soft authority signals in chunk metadata."""

import math

from retrieval.models import SearchResult

_TRUE_VALUES = frozenset({"1", "true", "yes", "y", "peer_reviewed", "peer-reviewed"})
_FALSE_VALUES = frozenset({"0", "false", "no", "n"})
_NEUTRAL = 0.5


class AuthorityBooster:
    """Blend relevance with soft authority metadata signals.

    Inspired by LlamaIndex/Haystack metadata boost postprocessors. Distinct
    from :class:`~retrieval.venue_tier_boost.VenueTierBooster` (which maps
    ``venue`` / ``journal`` / ``venue_tier`` prestige labels) and
    :class:`~retrieval.citation_count_boost.CitationCountBooster` (which uses
    batch-normalized ``citation_count`` / ``cited_by_count``). This booster
    reads a different key scheme:

    1. ``source_authority`` — float in ``[0.0, 1.0]`` used directly when present.
    2. Else soft-combine any of:
       * ``venue_rank`` — positive numeric rank where ``1`` is best
         (``score = max(0, 1 - (rank - 1) * 0.1)``, clamped to ``[0, 1]``);
       * ``is_peer_reviewed`` — truthy → ``1.0``, falsey → ``0.2``;
       * ``impact_factor`` — numeric buckets: ``>=10 → 1.0``, ``>=3 → 0.65``,
         ``>0 → 0.35``, else ignored; or labels ``high`` / ``medium`` / ``low``.
    3. Missing all signals → neutral ``0.5`` (does not demote unknowns).

    Blended score:

    ```text
    new_score = (1 - alpha) * old + alpha * authority
    ```

    Results are re-sorted descending (stable). Inputs are not mutated. Local
    retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create an authority booster.

        Args:
            alpha: Weight for the authority signal in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``alpha`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        self._alpha = alpha

    def boost(
        self,
        results: list[SearchResult],
        query: str = "",
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored by authority metadata signals."""
        del query  # API parity with text boosters; authority is metadata-only.
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []
        rescored: list[SearchResult] = []
        for result in results:
            authority = self._authority_score(result.chunk.metadata)
            score = (1.0 - self._alpha) * result.score + self._alpha * authority
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="authority_boost",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _authority_score(self, metadata: dict[str, str]) -> float:
        direct = self._parse_source_authority(metadata)
        if direct is not None:
            return direct

        signals: list[float] = []
        rank_score = self._venue_rank_score(metadata)
        if rank_score is not None:
            signals.append(rank_score)
        peer_score = self._peer_reviewed_score(metadata)
        if peer_score is not None:
            signals.append(peer_score)
        impact_score = self._impact_factor_score(metadata)
        if impact_score is not None:
            signals.append(impact_score)
        if not signals:
            return _NEUTRAL
        return sum(signals) / len(signals)

    @staticmethod
    def _parse_source_authority(metadata: dict[str, str]) -> float | None:
        raw = metadata.get("source_authority", "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value

    @staticmethod
    def _venue_rank_score(metadata: dict[str, str]) -> float | None:
        raw = metadata.get("venue_rank", "").strip()
        if not raw:
            return None
        try:
            rank = float(raw)
        except ValueError:
            return None
        if not math.isfinite(rank) or rank < 1.0:
            return None
        return float(min(max(1.0 - (rank - 1.0) * 0.1, 0.0), 1.0))

    @staticmethod
    def _peer_reviewed_score(metadata: dict[str, str]) -> float | None:
        raw = metadata.get("is_peer_reviewed", "").strip().casefold()
        if not raw:
            return None
        if raw in _TRUE_VALUES:
            return 1.0
        if raw in _FALSE_VALUES:
            return 0.2
        return None

    @staticmethod
    def _impact_factor_score(metadata: dict[str, str]) -> float | None:
        raw = metadata.get("impact_factor", "").strip()
        if not raw:
            return None
        label = raw.casefold()
        if label in {"high", "h"}:
            return 1.0
        if label in {"medium", "med", "m"}:
            return 0.65
        if label in {"low", "l"}:
            return 0.35
        try:
            value = float(raw)
        except ValueError:
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        if value >= 10.0:
            return 1.0
        if value >= 3.0:
            return 0.65
        return 0.35
