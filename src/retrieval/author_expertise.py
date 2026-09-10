"""Rank papers by author expertise proxies from local metadata.

Deterministic ranking using per-author publication counts and topic/keyword
overlap with a caller-provided topic query. Fills a Semantic Scholar
influential-citation / OpenAlex author-topics gap with offline heuristics
(no LLM or network call). Optional later narrative can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from retrieval.sparse import meaningful_terms

_WHITESPACE = re.compile(r"\s+")
_SPLIT_AUTHORS = re.compile(r"\s*(?:,|;| and )\s*", re.IGNORECASE)


@dataclass(frozen=True)
class RankedPaper:
    """One paper row with an expertise proxy score."""

    title: str
    expertise_score: float
    author_pub_count: int
    topic_overlap: float
    matched_authors: tuple[str, ...]


class AuthorExpertiseRanker:
    """Rank papers by author publication count and topic overlap.

    Offline ranker for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
    pipelines — Semantic Scholar influential-citation / OpenAlex author-topics
    gap. Deterministic: identical inputs and ``topic_query`` always yield the
    same ordering. Distinct from citation-count boosters that ignore author
    identity.
    """

    def __init__(
        self,
        *,
        pub_count_weight: float = 0.5,
        topic_weight: float = 0.5,
    ) -> None:
        """Create an author-expertise ranker.

        Args:
            pub_count_weight: Blend weight for log-normalized max author
                publication count (``[0.0, 1.0]``).
            topic_weight: Blend weight for topic Jaccard overlap
                (``[0.0, 1.0]``). Weights are renormalized to sum to 1.0 when
                both are positive.

        Raises:
            ValueError: If either weight is non-finite or outside ``[0.0, 1.0]``,
                or both weights are ``0.0``.
        """
        for name, value in (
            ("pub_count_weight", pub_count_weight),
            ("topic_weight", topic_weight),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite number within [0.0, 1.0]")
        if pub_count_weight == 0.0 and topic_weight == 0.0:
            raise ValueError("at least one of pub_count_weight or topic_weight must be > 0")
        total = pub_count_weight + topic_weight
        self._pub_count_weight = pub_count_weight / total
        self._topic_weight = topic_weight / total

    def rank(
        self,
        papers: Sequence[dict[str, object]],
        *,
        topic_query: str = "",
        author_pub_counts: Mapping[str, int] | None = None,
    ) -> tuple[RankedPaper, ...]:
        """Return papers ranked by descending expertise proxy score.

        Each paper may carry ``title``, ``abstract``, ``authors`` (list or
        comma/semicolon-separated string), ``topics`` / ``keywords``, and
        optional per-row ``author_publication_count``. Global
        ``author_pub_counts`` maps author display names to publication totals.
        Empty input yields an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        topic_terms = (
            frozenset(meaningful_terms(topic_query)) if topic_query.strip() else frozenset()
        )
        counts = {
            self._norm_author(str(name)): max(0, int(value))
            for name, value in (author_pub_counts or {}).items()
            if str(name).strip()
        }

        ranked: list[RankedPaper] = []
        max_count = 1
        scratch: list[tuple[str, int, float, tuple[str, ...], frozenset[str]]] = []
        for paper in papers:
            title = str(paper.get("title", "") or "").strip() or "Untitled"
            authors = self._authors(paper)
            pub_count = self._pub_count(paper, authors, counts)
            paper_terms = self._paper_terms(paper)
            overlap = self._jaccard(topic_terms, paper_terms) if topic_terms else 0.0
            scratch.append((title, pub_count, overlap, authors, paper_terms))
            if pub_count > max_count:
                max_count = pub_count

        log_max = math.log1p(max_count)
        for title, pub_count, overlap, authors, _terms in scratch:
            pub_signal = math.log1p(pub_count) / log_max if log_max > 0 else 0.0
            score = self._pub_count_weight * pub_signal + self._topic_weight * overlap
            ranked.append(
                RankedPaper(
                    title=title,
                    expertise_score=score,
                    author_pub_count=pub_count,
                    topic_overlap=overlap,
                    matched_authors=authors,
                )
            )
        ranked.sort(key=lambda item: (-item.expertise_score, item.title.lower()))
        return tuple(ranked)

    @staticmethod
    def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @staticmethod
    def _norm_author(name: str) -> str:
        return _WHITESPACE.sub(" ", name.strip().lower())

    def _authors(self, paper: dict[str, object]) -> tuple[str, ...]:
        raw = paper.get("authors", paper.get("author", ""))
        if isinstance(raw, (list, tuple)):
            names = [str(item).strip() for item in raw if str(item).strip()]
        else:
            text = str(raw or "").strip()
            names = [part.strip() for part in _SPLIT_AUTHORS.split(text) if part.strip()]
        # Preserve display order; de-dupe case-insensitively.
        seen: set[str] = set()
        ordered: list[str] = []
        for name in names:
            key = self._norm_author(name)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(name)
        return tuple(ordered)

    def _pub_count(
        self,
        paper: dict[str, object],
        authors: tuple[str, ...],
        counts: Mapping[str, int],
    ) -> int:
        raw = paper.get("author_publication_count", paper.get("author_pub_count"))
        if isinstance(raw, bool):
            pass
        elif isinstance(raw, int):
            return max(0, raw)
        elif isinstance(raw, float):
            return max(0, int(raw))
        elif isinstance(raw, str) and raw.strip():
            try:
                return max(0, int(raw.strip()))
            except ValueError:
                pass
        if not authors or not counts:
            return 0
        return max((counts.get(self._norm_author(name), 0) for name in authors), default=0)

    def _paper_terms(self, paper: dict[str, object]) -> frozenset[str]:
        parts: list[str] = []
        for key in ("topics", "keywords", "abstract", "title"):
            raw = paper.get(key)
            if raw is None or raw == "":
                continue
            if isinstance(raw, (list, tuple)):
                parts.extend(str(item) for item in raw)
            else:
                parts.append(str(raw))
        blob = _WHITESPACE.sub(" ", " ".join(parts).strip())
        return frozenset(meaningful_terms(blob))
