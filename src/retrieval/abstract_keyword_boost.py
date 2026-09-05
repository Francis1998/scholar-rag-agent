"""Boost retrieval hits when abstract/text contains query keywords."""

import math
import re
from collections.abc import Iterable

from retrieval.models import Chunk, SearchResult

_ABSTRACT_KEYS = ("abstract", "summary", "abstract_text", "keywords")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")


class AbstractKeywordBoost:
    """Boost scores when abstract or chunk text contains query keywords.

    Inspired by LlamaIndex/Haystack keyword boost postprocessors. Keywords are
    taken from the passed ``query`` string, optional ``query_terms``, or
    ``metadata["keywords"]`` on each chunk (comma/semicolon separated) when the
    query is empty. Coverage is ``|matched| / |keywords|`` over casefolded
    tokens found in abstract metadata first, else chunk text. Blended score:

    ```text
    new_score = (1 - alpha) * old + alpha * coverage
    ```

    Distinct from :class:`~retrieval.abstract_overlap_boost.AbstractOverlapBooster`
    (Jaccard overlap) and :class:`~retrieval.keyword_match_gate.KeywordMatchGate`
    (hard coverage filter). Inputs are not mutated. Local postprocessor for
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI
    connector).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create an abstract-keyword booster.

        Args:
            alpha: Weight for the keyword-coverage signal in ``[0.0, 1.0]``.

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
        query_terms: Iterable[str] | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored by abstract/text keyword coverage."""
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        explicit_terms = self._normalize_terms(query_terms) if query_terms is not None else []
        query_keywords = explicit_terms or self._tokens(query)

        rescored: list[SearchResult] = []
        for result in results:
            keywords = query_keywords or self._metadata_keywords(result.chunk.metadata)
            haystack = self._haystack_tokens(result.chunk)
            coverage = self._coverage(keywords, haystack)
            score = (1.0 - self._alpha) * result.score + self._alpha * coverage
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="abstract_keyword_boost",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

    def _haystack_tokens(self, chunk: Chunk) -> set[str]:
        for key in _ABSTRACT_KEYS:
            if key == "keywords":
                continue
            raw = chunk.metadata.get(key, "")
            if raw.strip():
                return set(self._tokens(raw))
        return set(self._tokens(chunk.text))

    def _metadata_keywords(self, metadata: dict[str, str]) -> list[str]:
        raw = metadata.get("keywords", "")
        if not raw.strip():
            return []
        parts = re.split(r"[,;|]", raw)
        return self._normalize_terms(parts)

    @staticmethod
    def _normalize_terms(terms: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for term in terms:
            for token in _TOKEN_RE.findall(str(term)):
                folded = token.casefold()
                if folded not in seen:
                    seen.add(folded)
                    ordered.append(folded)
        return ordered

    @staticmethod
    def _tokens(text: str) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for match in _TOKEN_RE.finditer(text):
            folded = match.group(0).casefold()
            if folded not in seen:
                seen.add(folded)
                ordered.append(folded)
        return ordered

    @staticmethod
    def _coverage(keywords: list[str], haystack: set[str] | list[str]) -> float:
        if not keywords:
            return 0.0
        hay = set(haystack)
        matched = sum(1 for keyword in keywords if keyword in hay)
        return matched / len(keywords)
