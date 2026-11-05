"""Offline HeapingBiasResidualCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from
`MeasurementErrorBiasCueExtractor` / `SurveyModeEffectCueExtractor`.
Fills an Elicit/Consensus/Connected Papers heaping-bias residual cue extractors gap.
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
        "heaping_bias_residual",
        re.compile(
            r"\bheaping[- ]bias[- ]residual\b",
            re.IGNORECASE,
        ),
    ),
    (
        "digit_preference_residual",
        re.compile(
            r"\bdigit[- ]preference[- ]residual\b",
            re.IGNORECASE,
        ),
    ),
    (
        "age_heaping_residual",
        re.compile(
            r"\bage[- ]heaping[- ]residual\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class HeapingBiasResidualCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class HeapingBiasResidualCueExtractor:
    """Extract offline advisory cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[HeapingBiasResidualCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[HeapingBiasResidualCue] = []
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
                HeapingBiasResidualCue(
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
        raw = paper.get("paper_id") or paper.get("id") or f"paper-{index}"
        return str(raw)

    @staticmethod
    def _collect_text(paper: dict[str, object]) -> str:
        parts: list[str] = []
        for field in _TEXT_FIELDS:
            value = paper.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        return "\n".join(parts)
