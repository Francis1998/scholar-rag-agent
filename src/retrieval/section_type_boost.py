"""Deterministic section-type re-scoring for hybrid retrieval results."""

import math
from collections.abc import Mapping

from retrieval.models import SearchResult

_SECTION_KEYS = ("section", "section_type")
_DEFAULT_SECTION_SCORES: dict[str, float] = {
    "results": 1.0,
    "methods": 1.0,
    "conclusion": 1.0,
    "abstract": 1.0,
}
_UNKNOWN_SCORE = 0.2


class SectionTypeBooster:
    """Re-score results by blending relevance with preferred section types.

    Inspired by LlamaIndex/Haystack metadata boost postprocessors that promote
    chunks from high-value paper sections. Section signals are read from
    ``chunk.metadata`` keys ``section`` or ``section_type``. Preferred labels
    default to ``results``, ``methods``, ``conclusion``, and ``abstract``
    (score ``1.0``); unknown sections score ``0.2``. An optional constructor
    ``section_scores`` map overlays the defaults. The blended score is:

    ```text
    new_score = (1 - alpha) * old + alpha * section_score
    ```

    Results are re-sorted by ``new_score`` descending; equal scores retain
    input order (stable sort). Input objects are not mutated. This is a local
    retrieval postprocessor (not a DOI connector).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        section_scores: Mapping[str, float] | None = None,
    ) -> None:
        """Create a section-type booster.

        Args:
            alpha: Weight assigned to the section score in ``[0.0, 1.0]``. The
                complementary weight ``1 - alpha`` is applied to the previous
                score.
            section_scores: Optional mapping of section labels to finite scores
                in ``[0.0, 1.0]``. Keys are matched case-insensitively and
                overlay the built-in preferred set.

        Raises:
            ValueError: If ``alpha`` is invalid or a section score is invalid.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        merged = dict(_DEFAULT_SECTION_SCORES)
        if section_scores:
            for name, score in section_scores.items():
                label = name.strip().casefold()
                if not label:
                    raise ValueError("section_scores keys must be non-empty")
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(
                        "section_scores values must be finite numbers within [0.0, 1.0]"
                    )
                merged[label] = float(score)
        self._alpha = alpha
        self._section_scores = merged

    def boost(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored and ordered by blended section type.

        Resolution order per chunk:

        1. ``metadata["section"]`` looked up in the merged section-score map.
        2. Else ``metadata["section_type"]`` looked up the same way.
        3. Else unknown score ``0.2``.

        ``top_k`` truncates after sorting; ``None`` keeps every result.
        """
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        rescored: list[SearchResult] = []
        for result in results:
            section_score = self._section_score(result.chunk.metadata)
            score = (1.0 - self._alpha) * result.score + self._alpha * section_score
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="section_type_boost",
                    path=[*result.path, result.retriever],
                )
            )
        # Stable sort: Python's sorted is stable, so equal scores keep input order.
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]

    def _section_score(self, metadata: dict[str, str]) -> float:
        for key in _SECTION_KEYS:
            name = metadata.get(key, "").strip().casefold()
            if not name:
                continue
            if name in self._section_scores:
                return self._section_scores[name]
        return _UNKNOWN_SCORE
