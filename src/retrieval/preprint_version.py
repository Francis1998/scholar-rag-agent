"""Compare preprint vs published paper metadata offline.

Deterministic heuristics inspect version labels (v1/v2), publication-year
delta, and venue presence to classify the relationship between a preprint row
and a published row. Fills a Semantic Scholar / ResearchRabbit version-diff gap
without an LLM or network call. Optional later narrative can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

Distinct from :class:`~retrieval.preprint_demote.PreprintDemoter` (retrieval
soft demotion of preprint hits). This module is a version-comparison helper
over caller-supplied metadata pairs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

_WHITESPACE = re.compile(r"\s+")
_YEAR = re.compile(r"(?:^|\D)((?:19|20)\d{2})(?:\D|$)")
_VERSION = re.compile(r"\bv(?:ersion)?\s*([0-9]{1,2})\b", re.IGNORECASE)

PreprintVersionStatus = Literal[
    "same_work_preprint_older",
    "published_preferred",
    "preprint_only",
    "unknown",
]

PreferSide = Literal["preprint", "published", "either", "none"]


@dataclass(frozen=True)
class PreprintVersionDiff:
    """Offline comparison of preprint vs published metadata."""

    status: PreprintVersionStatus
    reasons: tuple[str, ...]
    prefer: PreferSide


class PreprintVersionDiffer:
    """Compare preprint and published metadata for the same scholarly work.

    Offline helper for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
    pipelines — Semantic Scholar / ResearchRabbit version-diff gap. Uses
    version labels, year delta, and venue presence. Distinct from
    :class:`~retrieval.preprint_demote.PreprintDemoter` (which demotes preprint
    ``SearchResult`` hits during retrieval). Inputs are not mutated.
    """

    def diff(
        self,
        preprint: Mapping[str, object] | None,
        published: Mapping[str, object] | None = None,
    ) -> PreprintVersionDiff:
        """Return a version-diff report for ``preprint`` vs ``published``.

        Each mapping may carry ``title``, ``year`` / ``published_at`` / ``date``,
        ``version`` / ``arxiv_version``, and ``venue`` / ``journal``. Missing
        published metadata yields ``preprint_only``. Missing both sides yields
        ``unknown``. When both sides exist, venue presence and newer year or
        higher version favor the published record
        (``published_preferred`` / ``same_work_preprint_older``).
        """
        if not preprint and not published:
            return PreprintVersionDiff(
                status="unknown",
                reasons=("no preprint or published metadata provided",),
                prefer="none",
            )
        if preprint and not published:
            return PreprintVersionDiff(
                status="preprint_only",
                reasons=("published metadata missing; preprint record only",),
                prefer="preprint",
            )
        if published and not preprint:
            return PreprintVersionDiff(
                status="published_preferred",
                reasons=("preprint metadata missing; published record only",),
                prefer="published",
            )

        assert preprint is not None and published is not None
        reasons: list[str] = []

        pre_year = self._year(preprint)
        pub_year = self._year(published)
        pre_ver = self._version(preprint)
        pub_ver = self._version(published)
        pre_venue = self._venue(preprint)
        pub_venue = self._venue(published)

        year_delta: int | None = None
        if pre_year is not None and pub_year is not None:
            year_delta = pub_year - pre_year
            reasons.append(f"year delta (published - preprint) = {year_delta}")
        elif pre_year is None and pub_year is None:
            reasons.append("missing publication years on both sides")
        else:
            reasons.append("incomplete year pair for delta")

        if pre_ver is not None:
            reasons.append(f"preprint version label v{pre_ver}")
        if pub_ver is not None:
            reasons.append(f"published version label v{pub_ver}")

        venue_published = bool(pub_venue)
        venue_preprint = bool(pre_venue)
        if venue_published and not venue_preprint:
            reasons.append(f"published venue present ({pub_venue})")
        elif venue_published and venue_preprint:
            reasons.append(f"both sides have venue labels (published={pub_venue})")
        elif not venue_published and not venue_preprint:
            reasons.append("no venue on either side")
        else:
            reasons.append(f"preprint venue present without published venue ({pre_venue})")

        newer_published = year_delta is not None and year_delta > 0
        higher_pub_version = pre_ver is not None and pub_ver is not None and pub_ver > pre_ver

        if venue_published and newer_published:
            reasons.append("preprint appears older than published venue version")
            return PreprintVersionDiff(
                status="same_work_preprint_older",
                reasons=tuple(reasons),
                prefer="published",
            )

        if venue_published:
            reasons.append("published venue preferred over preprint metadata")
            return PreprintVersionDiff(
                status="published_preferred",
                reasons=tuple(reasons),
                prefer="published",
            )

        if newer_published or higher_pub_version:
            reasons.append("published metadata looks newer than preprint")
            return PreprintVersionDiff(
                status="same_work_preprint_older",
                reasons=tuple(reasons),
                prefer="published",
            )

        if year_delta is None and pre_ver is None and pub_ver is None:
            reasons.append("insufficient signals to compare versions")
            return PreprintVersionDiff(
                status="unknown",
                reasons=tuple(reasons),
                prefer="either",
            )

        if year_delta is not None and year_delta < 0:
            reasons.append("preprint year newer without published venue")
            return PreprintVersionDiff(
                status="unknown",
                reasons=tuple(reasons),
                prefer="either",
            )

        reasons.append("default preference for published record when paired")
        return PreprintVersionDiff(
            status="published_preferred",
            reasons=tuple(reasons),
            prefer="published",
        )

    @staticmethod
    def _year(row: Mapping[str, object]) -> int | None:
        for key in ("year", "published_at", "date", "publication_year"):
            raw = row.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if not text:
                continue
            if text.isdigit() and len(text) == 4:
                year = int(text)
                if 1900 <= year <= 2100:
                    return year
            match = _YEAR.search(text)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _version(row: Mapping[str, object]) -> int | None:
        for key in ("version", "arxiv_version", "preprint_version"):
            raw = row.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if not text:
                continue
            if text.isdigit():
                return int(text)
            match = _VERSION.search(text)
            if match:
                return int(match.group(1))
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
                return int(text.split(".", 1)[0])
        title = str(row.get("title", "") or "")
        match = _VERSION.search(title)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _venue(row: Mapping[str, object]) -> str:
        for key in ("venue", "journal", "conference", "booktitle"):
            raw = str(row.get(key, "") or "").strip()
            if not raw:
                continue
            folded = raw.casefold()
            if any(
                marker in folded for marker in ("arxiv", "biorxiv", "medrxiv", "preprint", "ssrn")
            ):
                continue
            return _WHITESPACE.sub(" ", raw)
        return ""
