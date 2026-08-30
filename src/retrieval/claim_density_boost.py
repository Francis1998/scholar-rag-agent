"""Boost retrieval hits by claim-like sentence density in chunk text."""

import math
import re

from retrieval.models import SearchResult

_CLAIM_VERB_RE = re.compile(
    r"\b(show|shows|showed|demonstrate|demonstrates|demonstrated|find|finds|found|"
    r"conclude|concludes|concluded|suggest|suggests|suggested|indicate|indicates|indicated)\b",
    re.IGNORECASE,
)
_CLAIM_PRONOUN_RE = re.compile(r"\bwe\b|\bour results\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class ClaimDensityBooster:
    """Blend relevance with claim-like sentence density in chunk text.

    Inspired by claim-centric reranking in scholarly RAG (PaperQA/LlamaIndex).
    Claim-like sentences contain reporting verbs (show, demonstrate, find,
    conclude, suggest, indicate) or phrases such as ``we`` / ``our results``.
    Density is ``claim_sentences / max(total_sentences, 1)`` in ``[0.0, 1.0]``:

    ```text
    new_score = (1 - alpha) * old + alpha * density
    ```

    Results are re-sorted descending (stable). Inputs are not mutated. Local
    retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create a claim-density booster.

        Args:
            alpha: Weight for the density signal in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``alpha`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        self._alpha = alpha

    def boost(
        self,
        results: list[SearchResult],
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored by claim-density in ``chunk.text``."""
        del query  # API parity with other overlap boosters; density is text-only.
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []
        rescored: list[SearchResult] = []
        for result in results:
            density = self._claim_density(result.chunk.text)
            score = (1.0 - self._alpha) * result.score + self._alpha * density
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="claim_density_boost",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _claim_density(self, text: str) -> float:
        parts = _SENTENCE_SPLIT_RE.split(text.strip())
        sentences = [part.strip() for part in parts if part.strip()]
        if not sentences:
            return 0.0
        claim_count = sum(1 for sentence in sentences if self._is_claim_like(sentence))
        return claim_count / max(len(sentences), 1)

    @staticmethod
    def _is_claim_like(sentence: str) -> bool:
        return bool(_CLAIM_VERB_RE.search(sentence) or _CLAIM_PRONOUN_RE.search(sentence))
