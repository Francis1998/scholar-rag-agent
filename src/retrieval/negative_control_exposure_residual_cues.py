"""Offline NegativeControlExposureResidualCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from
`NegativeControlOutcomeCueExtractor` / `ResidualConfoundingCueExtractor`.
Fills an Elicit/Consensus/Connected Papers negative-control exposure residual cue extractors gap.
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
        "negative_control_exposure_residual",
        re.compile(
            r"\bnegative[- ]control[- ]exposure[- ]residual\b",
            re.IGNORECASE,
        ),
    ),
    (
        "negative_control_exposure_residual_bias",
        re.compile(
            r"\bnegative[- ]control[- ]exposure[- ]residual[- ]bias\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sensitivity_negative_control_exposure_residual",
        re.compile(
            r"\bsensitivity[- ]negative[- ]control[- ]exposure[- ]residual\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class NegativeControlExposureResidualCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class NegativeControlExposureResidualCueExtractor:
    """Extract offline advisory cues."""

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[NegativeControlExposureResidualCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[NegativeControlExposureResidualCue] = []
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
                NegativeControlExposureResidualCue(
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
