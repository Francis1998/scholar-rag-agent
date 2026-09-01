"""Gate retrieval hits by a local lexical cross-encoder proxy score."""

import math
import re

from retrieval.models import SearchResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class CrossEncoderGate:
    """Drop weak query-document pairs below a local cross-encoder proxy.

    Inspired by LlamaIndex ``SentenceTransformerRerank`` / Haystack
    ``SentenceTransformersRanker`` cross-encoder gates, implemented here as a
    deterministic lexical proxy (no model download):

    ```text
    score = 0.5 * Jaccard(query, doc) + 0.5 * coverage(query in doc)
    ```

    Results at or above ``min_score`` are kept (rewritten provenance); weaker
    hits are dropped. Distinct from
    :class:`~retrieval.answerability_gate.AnswerabilityGate` aggregate gating
    and :class:`~retrieval.lexical_overlap_boost.LexicalOverlapBooster`
    soft-boost. Inputs are not mutated. Local postprocessor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, min_score: float = 0.2) -> None:
        """Create a cross-encoder gate.

        Args:
            min_score: Inclusive keep threshold in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``min_score`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(min_score) or not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be a finite number within [0.0, 1.0]")
        self._min_score = min_score

    def gate(
        self,
        results: list[SearchResult],
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results whose proxy cross-encoder score meets ``min_score``."""
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        query_tokens = self._tokens(query)
        kept: list[SearchResult] = []
        for result in results:
            proxy = self._proxy(query_tokens, self._tokens(result.chunk.text))
            if proxy < self._min_score:
                continue
            kept.append(
                SearchResult(
                    chunk=result.chunk,
                    score=proxy,
                    retriever="cross_encoder_gate",
                    path=[*result.path, result.retriever],
                )
            )
            if len(kept) >= limit:
                break
        return kept

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_RE.findall(text)}

    @staticmethod
    def _proxy(query_tokens: set[str], doc_tokens: set[str]) -> float:
        if not query_tokens and not doc_tokens:
            return 1.0
        if not query_tokens or not doc_tokens:
            return 0.0
        jaccard = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
        coverage = len(query_tokens & doc_tokens) / len(query_tokens)
        return 0.5 * jaccard + 0.5 * coverage
