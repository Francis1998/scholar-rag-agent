"""Boost retrieval hits that mention public code repositories or availability.

Offline heuristics scan title/abstract for host URLs (github.com, gitlab,
bitbucket) and phrases such as ``code available``, ``source code``, and
``implementation``. Fills a PapersWithCode / Semantic Scholar code-link gap
without an LLM or network call. Optional later narrative can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

Distinct from :class:`~retrieval.open_access_prefer.OpenAccessPreferencer`
(OA metadata preference) and from DOI/Unpaywall connectors. Accepts
``SearchResult`` rows or SearchResult-like mappings and returns boosted scores
with human-readable reasons.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from retrieval.models import SearchResult

_WHITESPACE = re.compile(r"\s+")

_HOST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("github.com", re.compile(r"github\.com", re.IGNORECASE)),
    ("gitlab", re.compile(r"gitlab(?:\.com)?", re.IGNORECASE)),
    ("bitbucket", re.compile(r"bitbucket(?:\.org)?", re.IGNORECASE)),
)

_PHRASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code available", re.compile(r"\bcode\s+available\b", re.IGNORECASE)),
    ("source code", re.compile(r"\bsource\s+code\b", re.IGNORECASE)),
    ("implementation", re.compile(r"\bimplementation\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class CodeAvailabilityBoosted:
    """One hit re-scored by offline code-availability cues."""

    title: str
    score: float
    prior_score: float
    code_signal: float
    reasons: tuple[str, ...]
    result: SearchResult | None = None


class CodeAvailabilityBooster:
    """Boost SearchResult-like hits that advertise public code.

    Offline postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines — PapersWithCode / Semantic Scholar code-link gap.
    Blends prior relevance with a code-availability signal:

    ```text
    new_score = (1 - alpha) * old + alpha * code_signal
    ```

    ``code_signal`` is ``1.0`` when any host or phrase cue matches title or
    abstract text, else ``0.0``. Distinct from
    :class:`~retrieval.open_access_prefer.OpenAccessPreferencer` (OA metadata)
    and from live PapersWithCode connectors. Inputs are not mutated.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create a code-availability booster.

        Args:
            alpha: Weight for the code-availability signal in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``alpha`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        self._alpha = alpha

    def boost(
        self,
        results: Sequence[SearchResult | Mapping[str, object]],
        *,
        top_k: int | None = None,
    ) -> list[CodeAvailabilityBoosted]:
        """Return hits re-scored by code-availability cues with reasons.

        Accepts ``SearchResult`` objects or mappings with ``title``,
        ``abstract`` / ``text`` / ``summary``, and optional ``score``. Empty
        input yields ``[]``. ``top_k`` truncates after sorting; ``None`` keeps
        every row. Results are sorted by descending ``score`` (stable).
        """
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        boosted: list[CodeAvailabilityBoosted] = []
        for item in results:
            title, abstract, prior, search_result = self._coerce(item)
            signal, reasons = self._code_signal(title, abstract)
            score = (1.0 - self._alpha) * prior + self._alpha * signal
            new_result: SearchResult | None = None
            if search_result is not None:
                new_result = SearchResult(
                    chunk=search_result.chunk,
                    score=score,
                    retriever="code_availability",
                    path=[*search_result.path, search_result.retriever],
                )
            boosted.append(
                CodeAvailabilityBoosted(
                    title=title,
                    score=score,
                    prior_score=prior,
                    code_signal=signal,
                    reasons=reasons,
                    result=new_result,
                )
            )
        # Stable sort: equal scores keep input order.
        return sorted(boosted, key=lambda row: row.score, reverse=True)[:limit]

    def _coerce(
        self,
        item: SearchResult | Mapping[str, object],
    ) -> tuple[str, str, float, SearchResult | None]:
        if isinstance(item, SearchResult):
            title = (item.chunk.title or "").strip() or "Untitled"
            abstract = self._abstract_from_search_result(item)
            return title, abstract, float(item.score), item

        title = str(item.get("title", "") or "").strip() or "Untitled"
        abstract = self._abstract_from_mapping(item)
        raw_score = item.get("score", 0.0)
        try:
            prior = float(raw_score)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            prior = 0.0
        if not math.isfinite(prior):
            prior = 0.0
        return title, abstract, prior, None

    @staticmethod
    def _abstract_from_search_result(result: SearchResult) -> str:
        metadata = result.chunk.metadata
        for key in ("abstract", "summary", "abstract_text"):
            raw = metadata.get(key, "").strip()
            if raw:
                return raw
        return (result.chunk.text or "").strip()

    @staticmethod
    def _abstract_from_mapping(item: Mapping[str, object]) -> str:
        for key in ("abstract", "summary", "abstract_text", "text"):
            raw = str(item.get(key, "") or "").strip()
            if raw:
                return raw
        return ""

    def _code_signal(self, title: str, abstract: str) -> tuple[float, tuple[str, ...]]:
        haystack = f"{title}\n{abstract}"
        reasons: list[str] = []
        for label, pattern in _HOST_PATTERNS:
            if pattern.search(haystack):
                reasons.append(f"code host mention ({label})")
        for label, pattern in _PHRASE_PATTERNS:
            if pattern.search(haystack):
                reasons.append(f"code availability phrase ({label})")
        if not reasons:
            return 0.0, ("no code availability signal",)
        return 1.0, tuple(reasons)
