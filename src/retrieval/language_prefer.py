"""Prefer retrieval hits in configured language(s) via boost or soft filter."""

import math
from collections.abc import Iterable

from retrieval.models import SearchResult

_LANG_KEYS = ("language", "lang")


class LanguagePreferencer:
    """Prefer results matching preferred language(s) with boost or soft filter.

    Inspired by Haystack ``LanguageClassifier``-style language signals and
    LlamaIndex metadata filters — implemented here as a local retrieval
    postprocessor (not a DOI connector). Language is read from
    ``chunk.metadata`` keys ``language`` or ``lang`` (case-insensitive).

    * ``mode="boost"`` (default): blend prior score with a language signal of
      ``1.0`` / ``0.0`` using ``alpha``.
    * ``mode="filter"``: when at least one hit matches a preferred language,
      drop non-matching rows; if none match, keep every input row.

    Boost mode re-sorts stably by blended score and does not mutate inputs.
    Filter mode preserves original identity, score, and order.
    """

    def __init__(
        self,
        preferred_languages: Iterable[str] | None = None,
        alpha: float = 0.3,
        mode: str = "boost",
    ) -> None:
        """Create a language preferencer.

        Args:
            preferred_languages: Languages to prefer (case-insensitive).
                Defaults to ``("en",)``.
            alpha: Weight assigned to the language signal in boost mode, in
                ``[0.0, 1.0]``. Ignored for filter mode scoring.
            mode: ``"boost"`` to blend scores, or ``"filter"`` to soft-filter
                non-preferred rows when any preferred hit exists.

        Raises:
            ValueError: If ``alpha`` is invalid, ``mode`` is unsupported, or
                ``preferred_languages`` is empty after normalization.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        if mode not in {"boost", "filter"}:
            raise ValueError("mode must be 'boost' or 'filter'")
        raw = ("en",) if preferred_languages is None else preferred_languages
        preferred = {item.strip().casefold() for item in raw if item.strip()}
        if not preferred:
            raise ValueError("preferred_languages must contain at least one language")
        self._alpha = alpha
        self._mode = mode
        self._preferred = preferred

    def prefer(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results preferred toward the configured language(s).

        In boost mode, each result receives
        ``(1 - alpha) * old + alpha * lang_signal`` and is sorted descending
        (stable for ties). In filter mode, non-matching rows are dropped only
        when the batch contains at least one preferred-language hit; otherwise
        the full input is kept. ``top_k`` truncates after preference; ``None``
        keeps every survivor.
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
            lang_signal = 1.0 if self._matches_preferred(result.chunk.metadata) else 0.0
            score = (1.0 - self._alpha) * result.score + self._alpha * lang_signal
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="language_prefer",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]

    def _filter(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        matched = [result for result in results if self._matches_preferred(result.chunk.metadata)]
        survivors = matched if matched else list(results)
        return survivors[:limit]

    def _matches_preferred(self, metadata: dict[str, str]) -> bool:
        for key in _LANG_KEYS:
            raw = metadata.get(key, "").strip().casefold()
            if raw and raw in self._preferred:
                return True
        return False
