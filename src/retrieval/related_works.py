"""Deterministic Related Works section scaffolding from paper metadata.

Composes a structured Related Works outline by clustering paper
title/abstract/year records into themes via keyword overlap. Fills an
Elicit / PaperQA related-work generation gap with offline, reproducible
scaffolding (no LLM call). Optional later prose drafting can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Distinct from
:class:`~retrieval.literature_review_outline.LiteratureReviewOutliner`
(section-type routing) and from retrieval gates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from retrieval.sparse import STOPWORDS, meaningful_terms

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class RelatedWorksTheme:
    """One paper assigned to a related-works theme."""

    title: str
    year: int | None
    abstract: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RelatedWorksSection:
    """One theme bucket in a Related Works outline."""

    theme: str
    summary: str
    papers: tuple[RelatedWorksTheme, ...]


@dataclass(frozen=True)
class RelatedWorksOutline:
    """Complete Related Works section outline."""

    title: str
    sections: tuple[RelatedWorksSection, ...]

    def to_markdown(self) -> str:
        """Render the Related Works outline as markdown scaffolding."""
        lines = [f"# {self.title or 'Related Works'}"]
        lines.append("")
        lines.append(
            "_Deterministic Related Works scaffolding from title/abstract/year "
            "keyword overlap. Optional narrative drafting can use GPT-5.5 / "
            "Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 (Elicit/PaperQA "
            "related-work generation gap)._"
        )
        for section in self.sections:
            lines.append("")
            lines.append(f"## {section.theme}")
            lines.append(section.summary)
            if not section.papers:
                lines.append("- _(no papers)_")
                continue
            for paper in section.papers:
                year = f" ({paper.year})" if paper.year is not None else ""
                lines.append(f"- **{paper.title}**{year}")
                if paper.abstract:
                    if len(paper.abstract) <= 160:
                        snippet = paper.abstract
                    else:
                        snippet = paper.abstract[:157] + "..."
                    lines.append(f"  - {snippet}")
        return "\n".join(lines).rstrip() + "\n"


class RelatedWorksComposer:
    """Compose a Related Works outline via deterministic keyword themes.

    Papers are greedy-clustered by Jaccard overlap of meaningful title+abstract
    terms. Each cluster becomes a theme section named from its top shared
    keywords. Offline scaffolding for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 — Elicit/PaperQA related-work generation gap.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.18,
        max_themes: int = 6,
    ) -> None:
        """Create a related-works composer.

        Args:
            overlap_threshold: Inclusive keyword Jaccard for joining a theme.
            max_themes: Maximum theme sections to emit (``>= 1``).

        Raises:
            ValueError: If knobs are non-finite / out of range.
        """
        if not math.isfinite(overlap_threshold) or not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be a finite number within [0.0, 1.0]")
        if not isinstance(max_themes, int) or max_themes < 1:
            raise ValueError("max_themes must be an int >= 1")
        self._overlap_threshold = overlap_threshold
        self._max_themes = max_themes

    def compose(
        self,
        papers: Sequence[dict[str, object] | RelatedWorksTheme],
        *,
        title: str = "Related Works",
    ) -> RelatedWorksOutline:
        """Return a Related Works outline from paper title/abstract/year rows.

        Accepts :class:`RelatedWorksTheme` instances or dicts with ``title``,
        optional ``abstract`` / ``year`` keys. Inputs are not mutated.
        """
        themes = [self._coerce(paper) for paper in papers]
        themes = [paper for paper in themes if paper.title.strip()]
        if not themes:
            return RelatedWorksOutline(title=title.strip() or "Related Works", sections=())

        term_sets = [set(paper.keywords) for paper in themes]
        assignments: list[int] = [-1] * len(themes)
        cluster_terms: list[set[str]] = []
        cluster_members: list[list[int]] = []

        order = sorted(
            range(len(themes)),
            key=lambda index: (-len(term_sets[index]), themes[index].title.lower()),
        )
        for index in order:
            terms = term_sets[index]
            best_cluster = -1
            best_score = -1.0
            for cluster_index, centroid in enumerate(cluster_terms):
                score = self._jaccard(terms, centroid)
                if score >= self._overlap_threshold and score > best_score:
                    best_score = score
                    best_cluster = cluster_index
            if best_cluster < 0 and len(cluster_terms) < self._max_themes:
                cluster_terms.append(set(terms))
                cluster_members.append([index])
                assignments[index] = len(cluster_terms) - 1
            elif best_cluster >= 0:
                cluster_terms[best_cluster] |= terms
                cluster_members[best_cluster].append(index)
                assignments[index] = best_cluster
            else:
                # Overflow into the most overlapping existing theme.
                overflow = max(
                    range(len(cluster_terms)),
                    key=lambda cluster_index: self._jaccard(terms, cluster_terms[cluster_index]),
                )
                cluster_terms[overflow] |= terms
                cluster_members[overflow].append(index)
                assignments[index] = overflow

        sections: list[RelatedWorksSection] = []
        for cluster_index, members in enumerate(cluster_members):
            member_papers = tuple(
                sorted(
                    (themes[index] for index in members),
                    key=lambda paper: (
                        -(paper.year or 0),
                        paper.title.lower(),
                    ),
                )
            )
            theme_name = self._theme_name(cluster_terms[cluster_index], member_papers)
            summary = (
                f"{len(member_papers)} paper(s) sharing topical keywords "
                f"({', '.join(list(cluster_terms[cluster_index])[:5]) or 'general'})."
            )
            sections.append(
                RelatedWorksSection(
                    theme=theme_name,
                    summary=summary,
                    papers=member_papers,
                )
            )

        sections.sort(key=lambda section: (-len(section.papers), section.theme.lower()))
        return RelatedWorksOutline(
            title=title.strip() or "Related Works",
            sections=tuple(sections),
        )

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @staticmethod
    def _theme_name(terms: set[str], papers: tuple[RelatedWorksTheme, ...]) -> str:
        ranked = sorted(terms, key=lambda term: (-len(term), term))
        if ranked:
            head = " / ".join(word.title() for word in ranked[:3])
            return f"{head} Theme"
        if papers:
            first = papers[0].title.strip()
            return (first[:48] + "…") if len(first) > 48 else first
        return "General Theme"

    @staticmethod
    def _coerce(paper: dict[str, object] | RelatedWorksTheme) -> RelatedWorksTheme:
        if isinstance(paper, RelatedWorksTheme):
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
        blob = f"{title} {abstract}".strip()
        keywords = tuple(sorted(meaningful_terms(blob)))
        # Drop ultra-short tokens already excluded by STOPWORDS; keep stable order.
        keywords = tuple(term for term in keywords if term not in STOPWORDS and len(term) > 1)
        return RelatedWorksTheme(
            title=title or "Untitled",
            year=year,
            abstract=_WHITESPACE.sub(" ", abstract),
            keywords=keywords,
        )
