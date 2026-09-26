"""Offline WashoutPeriodCueExtractor extraction.

Detects advisory washout / run-in / wash-in period cues from
title/abstract/methods via deterministic patterns. Never calls the network.
Distinct from IntentionToTreatCueExtractor and ProtocolDeviationCueExtractor.
Fills an Elicit / Consensus / CONSORT washout-period gap.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "methods", "method", "results", "discussion")

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "washout_period",
        re.compile(
            r"\bwash[- ]?out period\b|\bwashout of\b|\bafter a washout\b",
            re.IGNORECASE,
        ),
    ),
    (
        "run_in_period",
        re.compile(
            r"\brun[- ]in period\b|\bplacebo run[- ]in\b|\brun[- ]in phase\b",
            re.IGNORECASE,
        ),
    ),
    (
        "wash_in_period",
        re.compile(
            r"\bwash[- ]?in period\b|\bwashin phase\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class WashoutPeriodCue:
    """Advisory washout-period cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class WashoutPeriodCueExtractor:
    """Extract offline washout / run-in period cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[WashoutPeriodCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[WashoutPeriodCue] = []
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
                WashoutPeriodCue(
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
                parts.append(value.strip())
        return "\n".join(parts)
