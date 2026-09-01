"""Collapse paraphrase-near-duplicate retrieval hits via character n-grams."""

import math
import re

from retrieval.models import SearchResult

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class ParaphraseCollapser:
    """Hard-drop paraphrase near-duplicates using character n-gram Jaccard.

    Inspired by LlamaIndex ``SimilarityPostprocessor`` paraphrase-aware dedupe
    and Haystack near-duplicate filters. Distinct from
    :class:`~retrieval.near_duplicate_collapse.NearDuplicateCollapser` (word
    ``meaningful_terms`` Jaccard) and
    :class:`~retrieval.novelty_diversify.NoveltyDiversifier` (soft demotion):
    this stage fingerprints lowercased alphanumeric character n-grams so light
    paraphrases still collapse. Survivors keep relative score order. Inputs are
    not mutated. Local postprocessor for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, threshold: float = 0.85, ngram_size: int = 3) -> None:
        """Create a paraphrase collapser.

        Args:
            threshold: Inclusive n-gram Jaccard in ``[0.0, 1.0]`` at which two
                texts are treated as paraphrases. Defaults to ``0.85``.
            ngram_size: Character n-gram width (``>= 2``). Defaults to ``3``.

        Raises:
            ValueError: If knobs are non-finite / out of range.
        """
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite number within [0.0, 1.0]")
        if not isinstance(ngram_size, int) or ngram_size < 2:
            raise ValueError("ngram_size must be an int >= 2")
        self._threshold = threshold
        self._ngram_size = ngram_size

    def collapse(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return paraphrase-collapsed results ordered by descending score."""
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        ordered = sorted(results, key=lambda result: result.score, reverse=True)
        fingerprints = [self._fingerprint(result.chunk.text) for result in ordered]
        kept: list[SearchResult] = []
        kept_fps: list[set[str]] = []
        for result, fingerprint in zip(ordered, fingerprints, strict=True):
            if any(self._jaccard(fingerprint, prior) >= self._threshold for prior in kept_fps):
                continue
            kept.append(
                SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    retriever="paraphrase_collapse",
                    path=[*result.path, result.retriever],
                )
            )
            kept_fps.append(fingerprint)
            if len(kept) >= limit:
                break
        return kept

    def _fingerprint(self, text: str) -> set[str]:
        compact = "".join(_WORD_RE.findall(text.lower()))
        size = self._ngram_size
        if len(compact) < size:
            return {compact} if compact else set()
        return {compact[index : index + size] for index in range(len(compact) - size + 1)}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)
