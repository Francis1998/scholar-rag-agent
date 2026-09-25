"""Offline ClusterRandomizationCueExtractor extraction.

Detects advisory cluster-randomized / stepped-wedge / ICC cues from
title/abstract/methods via deterministic patterns. Never calls the network.
Distinct from AllocationConcealmentCueExtractor and SampleSizeHintExtractor.
Fills an Elicit / Consensus / CONSORT-cluster gap.
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
        "cluster_randomized",
        re.compile(
            r"\bcluster[- ]randomi[sz]ed\b|\bcluster RCT\b|\bCRCT\b",
            re.IGNORECASE,
        ),
    ),
    (
        "stepped_wedge",
        re.compile(
            r"\bstepped[- ]wedge\b|\bstep[- ]wedge design\b",
            re.IGNORECASE,
        ),
    ),
    (
        "intracluster_correlation",
        re.compile(
            r"\bintracluster correlation\b|\bICC\s*[=:]\s*\d|\bintra[- ]cluster correlation\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class ClusterRandomizationCue:
    """Advisory cluster-randomization cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class ClusterRandomizationCueExtractor:
    """Extract offline cluster-randomization cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[ClusterRandomizationCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[ClusterRandomizationCue] = []
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
                ClusterRandomizationCue(
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
