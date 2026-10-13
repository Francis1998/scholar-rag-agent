"""Offline DepletionOfSusceptiblesCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from related bias cue extractors.
Fills an Elicit/Consensus/AJE depletion-of-susceptibles cue extractors gap.
Optional later narrative can use frontier LLMs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = (
    "abstract",
    "title",
    "summary",
    "methods",
    "method",
    "results",
    "discussion",
)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "depletion_of_susceptibles",
        re.compile(
            r"\bdepletion of susceptibles\b",
            re.IGNORECASE,
        ),
    ),
    (
        "survivor_cohort_depletion",
        re.compile(
            r"\bsurvivor cohort depletion\b",
            re.IGNORECASE,
        ),
    ),
    (
        "waning_hazard_selection",
        re.compile(
            r"\bwaning hazard selection\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class DepletionOfSusceptiblesCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class DepletionOfSusceptiblesCueExtractor:
    """Extract offline advisory cues."""

    def extract(
        self, papers: Sequence[dict[str, object]]
    ) -> tuple[DepletionOfSusceptiblesCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[DepletionOfSusceptiblesCue] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            cues: list[str] = []
            matched: list[str] = []
            kind: str | None = None
            if text:
                for label, pattern in _PATTERNS:
                    match = pattern.search(text)
                    if not match:
                        continue
                    cues.append(label)
                    matched.append(match.group(0))
                    if kind is None:
                        kind = label
            results.append(
                DepletionOfSusceptiblesCue(
                    paper_id=paper_id,
                    cue_kind=kind,
                    cues=tuple(cues),
                    matched=tuple(matched),
                    flagged=bool(cues),
                )
            )
        return tuple(results)

    @staticmethod
    def _resolve_id(paper: dict[str, object], index: int) -> str:
        """Resolve stable paper id."""

        for key in ("paper_id", "id", "doi"):
            value = paper.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"paper-{index}"

    @staticmethod
    def _collect_text(paper: dict[str, object]) -> str:
        """Join searchable text fields."""

        chunks: list[str] = []
        for key in _TEXT_FIELDS:
            value = paper.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value)
        return "\n".join(chunks)
