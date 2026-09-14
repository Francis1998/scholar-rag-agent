"""HITL PRISMA-style screening checklist builder for systematic reviews.

Builds advisory title/abstract screening checklist rows from caller-supplied
inclusion/exclusion criteria and paper metadata. Decisions always remain
``pending`` — this module never auto-includes or auto-excludes papers.
Fills a Covidence / Elicit / Rayyan PRISMA screening-checklist gap with
offline, reproducible scaffolding (no LLM call). Optional later narrative
can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")
_TEXT_FIELDS = ("title", "abstract", "summary")


@dataclass(frozen=True)
class PrismaScreeningRow:
    """One HITL PRISMA screening checklist row (always pending)."""

    paper_id: str
    stage: str
    decision: str
    inclusion_hits: tuple[str, ...]
    exclusion_hits: tuple[str, ...]
    checklist: tuple[str, ...]


class PrismaScreeningChecklist:
    """Build HITL PRISMA-style screening checklist rows offline.

    Offline screening checklist builder for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Covidence/Elicit/Rayyan PRISMA HITL gap.
    Lexical inclusion/exclusion hits are advisory cues only; ``decision`` is
    always ``pending``. Never auto-includes or auto-excludes papers. Never
    mutates inputs and never calls the network.
    """

    def build(
        self,
        papers: Sequence[dict[str, object]],
        *,
        inclusion: Sequence[str],
        exclusion: Sequence[str],
    ) -> tuple[PrismaScreeningRow, ...]:
        """Return pending screening rows for ``papers``.

        Each row includes matched inclusion/exclusion cue strings and a short
        checklist of human review prompts. ``decision`` is always ``pending``.
        Empty ``papers`` yields an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        inclusion_terms = [term.strip() for term in inclusion if str(term).strip()]
        exclusion_terms = [term.strip() for term in exclusion if str(term).strip()]

        rows: list[PrismaScreeningRow] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            folded = text.casefold()
            inc_hits = tuple(term for term in inclusion_terms if term.casefold() in folded)
            exc_hits = tuple(term for term in exclusion_terms if term.casefold() in folded)
            checklist = self._checklist(inc_hits, exc_hits, inclusion_terms, exclusion_terms)
            rows.append(
                PrismaScreeningRow(
                    paper_id=paper_id,
                    stage="title_abstract",
                    decision="pending",
                    inclusion_hits=inc_hits,
                    exclusion_hits=exc_hits,
                    checklist=checklist,
                )
            )
        return tuple(rows)

    @staticmethod
    def _checklist(
        inc_hits: tuple[str, ...],
        exc_hits: tuple[str, ...],
        inclusion_terms: list[str],
        exclusion_terms: list[str],
    ) -> tuple[str, ...]:
        items: list[str] = [
            "HITL: confirm title/abstract relevance before full-text screening.",
            "Decision remains pending — do not auto-include or auto-exclude.",
        ]
        if inclusion_terms:
            if inc_hits:
                items.append("Review inclusion cues matched: " + ", ".join(inc_hits) + ".")
            else:
                items.append(
                    "No inclusion cues matched; verify against: " + ", ".join(inclusion_terms) + "."
                )
        if exclusion_terms:
            if exc_hits:
                items.append("Review exclusion cues matched: " + ", ".join(exc_hits) + ".")
            else:
                items.append("No exclusion cues matched in title/abstract.")
        return tuple(items)

    @staticmethod
    def _resolve_id(paper: dict[str, object], index: int) -> str:
        for key in ("paper_id", "id", "doi", "document_id"):
            value = paper.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"paper-{index}"

    @staticmethod
    def _collect_text(paper: dict[str, object]) -> str:
        parts: list[str] = []
        for key in _TEXT_FIELDS:
            value = paper.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(_WHITESPACE.sub(" ", value.strip()))
        return "\n".join(parts)
