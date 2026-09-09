"""Deterministic survey theme-gap finder from paper metadata.

Given papers plus an expected theme checklist, scores how well each theme is
covered by title/abstract keyword overlap and flags missing or under-covered
themes. Distinct from
:class:`~retrieval.related_works.RelatedWorksComposer` (which *discovers*
themes by clustering) — this module *audits* coverage against expected themes.
Fills an Elicit / ResearchRabbit theme-coverage gap with offline, reproducible
scoring (no LLM call). Optional later narrative drafting can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from retrieval.sparse import meaningful_terms

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class SurveyPaper:
    """Normalized paper row used for theme-coverage scoring."""

    title: str
    abstract: str
    year: int | None
    terms: frozenset[str]


@dataclass(frozen=True)
class SurveyGap:
    """Coverage report for one expected survey theme."""

    theme: str
    coverage_score: float
    missing: bool
    matching_titles: tuple[str, ...]


class SurveyGapFinder:
    """Find missing / under-covered themes against an expected checklist.

    For each ``expected_themes`` entry, computes the best Jaccard overlap
    between the theme's meaningful terms and each paper's title+abstract terms.
    Themes whose best score falls below ``coverage_threshold`` are marked
    ``missing=True``. Offline checklist for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 — Elicit/ResearchRabbit theme-coverage gap.
    Distinct from :class:`~retrieval.related_works.RelatedWorksComposer`.
    """

    def __init__(self, coverage_threshold: float = 0.15) -> None:
        """Create a survey gap finder.

        Args:
            coverage_threshold: Inclusive Jaccard score at/above which a theme
                is considered covered (``[0.0, 1.0]``).

        Raises:
            ValueError: If ``coverage_threshold`` is non-finite or out of range.
        """
        if not math.isfinite(coverage_threshold) or not 0.0 <= coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be a finite number within [0.0, 1.0]")
        self._coverage_threshold = coverage_threshold

    def find(
        self,
        papers: Sequence[dict[str, object] | SurveyPaper],
        *,
        expected_themes: Sequence[str],
    ) -> tuple[SurveyGap, ...]:
        """Return per-theme coverage gaps for ``expected_themes``.

        Accepts :class:`SurveyPaper` instances or dicts with ``title`` and
        optional ``abstract`` / ``year`` keys. Inputs are not mutated. Empty
        papers yield ``coverage_score=0.0`` and ``missing=True`` for every
        non-blank theme. Blank theme strings are skipped.
        """
        themes = [theme.strip() for theme in expected_themes if str(theme).strip()]
        if not themes:
            return ()

        normalized = [self._coerce(paper) for paper in papers]
        normalized = [paper for paper in normalized if paper.title.strip()]

        gaps: list[SurveyGap] = []
        for theme in themes:
            theme_terms = frozenset(meaningful_terms(theme))
            if not normalized or not theme_terms:
                gaps.append(
                    SurveyGap(
                        theme=theme,
                        coverage_score=0.0,
                        missing=True,
                        matching_titles=(),
                    )
                )
                continue

            best = 0.0
            matches: list[tuple[float, str]] = []
            for paper in normalized:
                score = self._jaccard(theme_terms, paper.terms)
                if score > 0.0:
                    matches.append((score, paper.title))
                if score > best:
                    best = score
            matches.sort(key=lambda item: (-item[0], item[1].lower()))
            covered = tuple(title for score, title in matches if score >= self._coverage_threshold)
            matching_titles = (
                covered[:5] if covered else tuple(title for score, title in matches)[:5]
            )
            gaps.append(
                SurveyGap(
                    theme=theme,
                    coverage_score=best,
                    missing=best < self._coverage_threshold,
                    matching_titles=matching_titles,
                )
            )
        return tuple(gaps)

    @staticmethod
    def _jaccard(left: frozenset[str] | set[str], right: frozenset[str] | set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @staticmethod
    def _coerce(paper: dict[str, object] | SurveyPaper) -> SurveyPaper:
        if isinstance(paper, SurveyPaper):
            return paper
        title = str(paper.get("title", "")).strip()
        abstract = str(paper.get("abstract", "") or "").strip()
        year_raw = paper.get("year")
        year: int | None
        if year_raw is None or year_raw == "":
            year = None
        else:
            year_text = str(year_raw).strip()
            year = int(year_text) if _YEAR_RE.match(year_text) else None
        blob = _WHITESPACE.sub(" ", f"{title} {abstract}".strip())
        terms = frozenset(meaningful_terms(blob))
        return SurveyPaper(
            title=title or "Untitled",
            abstract=blob[len(title) :].strip() if title else blob,
            year=year,
            terms=terms,
        )
