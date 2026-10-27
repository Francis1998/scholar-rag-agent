"""Offline SurveyModeEffectCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from `MeasurementScaleShiftCueExtractor` / `MeasurementErrorBiasCueExtractor`.
Fills an Elicit/Consensus/Connected Papers survey-mode effect cue extractors gap.
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
        "survey_mode_effect",
        re.compile(
            r"\bsurvey[- ]mode effect\b",
            re.IGNORECASE,
        ),
    ),
    (
        "mode_effect_survey",
        re.compile(
            r"\bmode effect.{0,30}survey",
            re.IGNORECASE,
        ),
    ),
    (
        "mixed_mode_survey_bias",
        re.compile(
            r"\bmixed[- ]mode survey bias\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class SurveyModeEffectCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class SurveyModeEffectCueExtractor:
    """Extract offline advisory cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[SurveyModeEffectCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[SurveyModeEffectCue] = []
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
                SurveyModeEffectCue(
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
