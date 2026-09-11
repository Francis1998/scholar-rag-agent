"""Triage unread papers by novelty x authority for offline reading lists.

Deterministic heuristics combine citation-count authority, publication-year
freshness, and batch-relative keyword novelty. Fills a Zotero / ResearchRabbit
unread-triage gap without an LLM or network call. Optional later narrative can
use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

Distinct from :class:`~retrieval.freshness.FreshnessBooster`,
:class:`~retrieval.novelty_diversify.NoveltyDiversifier`, and
:class:`~retrieval.authority_boost.AuthorityBooster` (retrieval boosters);
this module ranks a reading list, not ``SearchResult`` hits.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from retrieval.sparse import meaningful_terms

_WHITESPACE = re.compile(r"\s+")
_YEAR = re.compile(r"(?:^|\D)((?:19|20)\d{2})(?:\D|$)")


@dataclass(frozen=True)
class PrioritizedPaper:
    """One unread paper with a novelty x authority priority score."""

    title: str
    priority_score: float
    novelty: float
    authority: float
    reasons: tuple[str, ...]


class ReadingListPrioritizer:
    """Rank unread papers by novelty x authority heuristics.

    Offline reading-list triage for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines — Zotero / ResearchRabbit unread-triage gap. Composes
    citation-count authority with year freshness and keyword novelty into a
    multiplicative ``priority_score``. Distinct from
    :class:`~retrieval.freshness.FreshnessBooster`,
    :class:`~retrieval.novelty_diversify.NoveltyDiversifier`, and
    :class:`~retrieval.authority_boost.AuthorityBooster` (those re-score
    retrieval hits; this triages a reading list).
    """

    def __init__(
        self,
        *,
        freshness_weight: float = 0.5,
        reference_year: int = 2026,
        half_life_years: float = 3.0,
    ) -> None:
        """Create a reading-list prioritizer.

        Args:
            freshness_weight: Blend weight for year freshness versus keyword
                novelty inside the novelty signal (``[0.0, 1.0]``). Novelty is
                ``freshness_weight * year_freshness + (1 - freshness_weight) *
                keyword_novelty``.
            reference_year: Anchor year for freshness (newer years score higher).
            half_life_years: Years after which the year-freshness signal halves
                (must be positive and finite).

        Raises:
            ValueError: If ``freshness_weight`` is outside ``[0.0, 1.0]``,
                ``reference_year`` is non-positive, or ``half_life_years`` is
                non-positive / non-finite.
        """
        if not math.isfinite(freshness_weight) or not 0.0 <= freshness_weight <= 1.0:
            raise ValueError("freshness_weight must be a finite number within [0.0, 1.0]")
        if not isinstance(reference_year, int) or reference_year <= 0:
            raise ValueError("reference_year must be a positive integer")
        if not math.isfinite(half_life_years) or half_life_years <= 0:
            raise ValueError("half_life_years must be a positive finite number")
        self._freshness_weight = freshness_weight
        self._reference_year = reference_year
        self._half_life_years = half_life_years

    def prioritize(
        self,
        papers: Sequence[Mapping[str, object]],
    ) -> tuple[PrioritizedPaper, ...]:
        """Return unread papers ranked by descending novelty x authority.

        Each paper may carry ``title``, ``year`` / ``published_at`` / ``date``,
        ``citation_count`` / ``cited_by_count`` / ``citations``, and
        ``keywords`` / ``topics`` / ``abstract``. Empty input yields an empty
        tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        titles = [self._title(paper) for paper in papers]
        citations = [self._citation_count(paper) for paper in papers]
        years = [self._year(paper) for paper in papers]
        term_sets = [self._paper_terms(paper) for paper in papers]

        max_cites = max(citations) if citations else 0
        log_max = math.log1p(max_cites) if max_cites > 0 else 1.0
        doc_freq = Counter(term for terms in term_sets for term in terms)
        n_docs = len(papers)

        ranked: list[PrioritizedPaper] = []
        for title, cites, year, terms in zip(titles, citations, years, term_sets, strict=True):
            authority = math.log1p(cites) / log_max if max_cites > 0 else 0.0
            year_freshness = self._year_freshness(year)
            keyword_novelty = self._keyword_novelty(terms, doc_freq, n_docs)
            novelty = (
                self._freshness_weight * year_freshness
                + (1.0 - self._freshness_weight) * keyword_novelty
            )
            priority = novelty * authority
            reasons = self._reasons(
                cites=cites,
                year=year,
                year_freshness=year_freshness,
                keyword_novelty=keyword_novelty,
                terms=terms,
                doc_freq=doc_freq,
                n_docs=n_docs,
            )
            ranked.append(
                PrioritizedPaper(
                    title=title,
                    priority_score=priority,
                    novelty=novelty,
                    authority=authority,
                    reasons=reasons,
                )
            )
        ranked.sort(key=lambda item: (-item.priority_score, -item.authority, item.title.lower()))
        return tuple(ranked)

    def _year_freshness(self, year: int | None) -> float:
        if year is None:
            return 0.0
        age = max(self._reference_year - year, 0)
        return math.exp(-math.log(2.0) * age / self._half_life_years)

    @staticmethod
    def _keyword_novelty(
        terms: frozenset[str],
        doc_freq: Counter[str],
        n_docs: int,
    ) -> float:
        if not terms or n_docs <= 0:
            return 0.0
        # Rare-in-batch terms contribute more novelty (1 - df/n).
        scores = [(1.0 - doc_freq[term] / n_docs) for term in terms]
        return sum(scores) / len(scores)

    def _reasons(
        self,
        *,
        cites: int,
        year: int | None,
        year_freshness: float,
        keyword_novelty: float,
        terms: frozenset[str],
        doc_freq: Counter[str],
        n_docs: int,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if cites > 0:
            reasons.append(f"citation authority ({cites} citations)")
        else:
            reasons.append("no citation signal")
        if year is not None:
            reasons.append(f"year freshness ({year}, score={year_freshness:.2f})")
        else:
            reasons.append("missing publication year")
        if terms:
            rare = sorted(
                terms,
                key=lambda term: (doc_freq[term], term),
            )[:3]
            rare_fmt = ", ".join(rare)
            reasons.append(
                f"keyword novelty ({keyword_novelty:.2f}; rare terms: {rare_fmt})"
            )
        else:
            reasons.append("no keyword novelty signal")
        if n_docs == 1:
            reasons.append("single-paper reading list")
        return tuple(reasons)

    @staticmethod
    def _title(paper: Mapping[str, object]) -> str:
        return str(paper.get("title", "") or "").strip() or "Untitled"

    @staticmethod
    def _citation_count(paper: Mapping[str, object]) -> int:
        for key in ("citation_count", "cited_by_count", "citations", "cite_count"):
            raw = paper.get(key)
            if isinstance(raw, bool):
                continue
            if isinstance(raw, int):
                return max(0, raw)
            if isinstance(raw, float) and math.isfinite(raw):
                return max(0, int(raw))
            if isinstance(raw, str) and raw.strip():
                try:
                    return max(0, int(float(raw.strip())))
                except ValueError:
                    continue
        return 0

    def _year(self, paper: Mapping[str, object]) -> int | None:
        for key in ("year", "published_at", "date", "publication_year"):
            raw = paper.get(key)
            if raw is None or raw == "":
                continue
            if isinstance(raw, bool):
                continue
            if isinstance(raw, int):
                if 1800 <= raw <= self._reference_year + 1:
                    return raw
                continue
            if isinstance(raw, float) and math.isfinite(raw):
                year = int(raw)
                if 1800 <= year <= self._reference_year + 1:
                    return year
                continue
            text = str(raw).strip()
            if text.isdigit():
                year = int(text)
                if 1800 <= year <= self._reference_year + 1:
                    return year
            match = _YEAR.search(text)
            if match:
                year = int(match.group(1))
                if 1800 <= year <= self._reference_year + 1:
                    return year
        return None

    @staticmethod
    def _paper_terms(paper: Mapping[str, object]) -> frozenset[str]:
        parts: list[str] = []
        for key in ("keywords", "topics", "abstract", "title"):
            raw = paper.get(key)
            if raw is None or raw == "":
                continue
            if isinstance(raw, (list, tuple)):
                parts.extend(str(item) for item in raw if str(item).strip())
            else:
                parts.append(str(raw))
        blob = _WHITESPACE.sub(" ", " ".join(parts).strip())
        return frozenset(meaningful_terms(blob))
