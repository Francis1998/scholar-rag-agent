"""Soft-demote preprint retrieval hits via blended re-scoring."""

import math

from retrieval.models import SearchResult

_SIGNAL_KEYS = ("publication_type", "type", "venue")
_PREPRINT_MARKERS = frozenset(
    {
        "arxiv",
        "biorxiv",
        "medrxiv",
        "preprint",
        "ssrn",
    }
)
_PREPRINT_SCORE = 0.2
_NON_PREPRINT_SCORE = 1.0


class PreprintDemoter:
    """Soft-demote preprint results by blending relevance with a demote score.

    Inspired by LlamaIndex/Haystack metadata boost postprocessors that down-rank
    lower-trust source classes. Preprint signals are read from
    ``chunk.metadata`` keys ``publication_type``, ``type``, or ``venue`` when
    the (casefolded) value contains ``arxiv``, ``biorxiv``, ``medrxiv``,
    ``preprint``, or ``ssrn``. The blended score is:

    ```text
    demote_score = 0.2 if preprint else 1.0
    new_score = (1 - alpha) * old + alpha * demote_score
    ```

    Results are re-sorted by ``new_score`` descending; equal scores retain
    input order (stable sort). Input objects are not mutated. This is a local
    retrieval postprocessor (not a DOI connector).
    """

    def __init__(self, alpha: float = 0.25) -> None:
        """Create a preprint demoter.

        Args:
            alpha: Weight assigned to the demote score in ``[0.0, 1.0]``. The
                complementary weight ``1 - alpha`` is applied to the previous
                score.

        Raises:
            ValueError: If ``alpha`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        self._alpha = alpha

    def demote(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored with preprint soft demotion.

        ``top_k`` truncates after sorting; ``None`` keeps every result.
        """
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        rescored: list[SearchResult] = []
        for result in results:
            demote_score = (
                _PREPRINT_SCORE if self._is_preprint(result.chunk.metadata) else _NON_PREPRINT_SCORE
            )
            score = (1.0 - self._alpha) * result.score + self._alpha * demote_score
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="preprint_demote",
                    path=[*result.path, result.retriever],
                )
            )
        # Stable sort: Python's sorted is stable, so equal scores keep input order.
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]

    @staticmethod
    def _is_preprint(metadata: dict[str, str]) -> bool:
        for key in _SIGNAL_KEYS:
            raw = metadata.get(key, "").strip().casefold()
            if not raw:
                continue
            for marker in _PREPRINT_MARKERS:
                if marker in raw:
                    return True
        return False
