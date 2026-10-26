"""Offline CompetingRiskInformativeCensoringCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from `CompetingEventBiasCueExtractor` / `AttritionInformativeCensoringCueExtractor`.
Fills an Elicit/Consensus/Connected Papers competing-risk informative-censoring cue extractors gap.
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
        "competing_risk_informative_censoring",
        re.compile(
            r"\bcompeting[- ]risk informative censoring\b",
            re.IGNORECASE,
        ),
    ),
    (
        "informative_censoring_competing",
        re.compile(
            r"\binformative censoring.{0,40}competing risk",
            re.IGNORECASE,
        ),
    ),
    (
        "dependent_censoring_competing",
        re.compile(
            r"\bdependent censoring.{0,40}competing",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class CompetingRiskInformativeCensoringCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class CompetingRiskInformativeCensoringCueExtractor:
    """Extract offline advisory cues."""

    def extract(
        self, papers: Sequence[dict[str, object]]
    ) -> tuple[CompetingRiskInformativeCensoringCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[CompetingRiskInformativeCensoringCue] = []
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
                    matched.append(match.group(0).strip())
                    if kind is None:
                        kind = label
            results.append(
                CompetingRiskInformativeCensoringCue(
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
                parts.append(value)
        return "\n".join(parts)
