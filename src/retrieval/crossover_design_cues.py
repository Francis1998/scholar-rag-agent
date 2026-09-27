"""Offline CrossoverDesignCueExtractor extraction.

Detects advisory crossover / Latin-square / ABAB design cues from
title/abstract/methods via deterministic patterns. Never calls the network.
Distinct from ClusterRandomizationCueExtractor and WashoutPeriodCueExtractor.
Fills an Elicit / Consensus / CONSORT crossover-design gap.
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
        "crossover_trial",
        re.compile(
            r"\bcrossover trial\b|\bcross[- ]over study\b|\bcrossover design\b",
            re.IGNORECASE,
        ),
    ),
    (
        "latin_square",
        re.compile(
            r"\bLatin square\b|\blatin-square design\b",
            re.IGNORECASE,
        ),
    ),
    (
        "abab_design",
        re.compile(
            r"\bABAB design\b|\bA-B-A-B\b|\breversal design\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class CrossoverDesignCue:
    """Advisory crossover-design cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class CrossoverDesignCueExtractor:
    """Extract offline crossover / Latin-square design cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[CrossoverDesignCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[CrossoverDesignCue] = []
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
                CrossoverDesignCue(
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
