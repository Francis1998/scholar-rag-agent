"""Offline HealthyWorkerSurvivorCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from `HealthyAdhererBiasCueExtractor` / `HealthyUserBiasCueExtractor`.
Fills an Elicit/Consensus/Connected Papers healthy-worker survivor cue extractors gap.
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
        "healthy_worker_survivor",
        re.compile(
            r"\bhealthy[- ]worker survivor(?: bias)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "healthy_worker_effect",
        re.compile(
            r"\bhealthy[- ]worker effect\b",
            re.IGNORECASE,
        ),
    ),
    (
        "survivor_bias_occupational",
        re.compile(
            r"\bsurvivor bias.{0,24}occupational",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class HealthyWorkerSurvivorCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class HealthyWorkerSurvivorCueExtractor:
    """Extract offline advisory cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[HealthyWorkerSurvivorCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[HealthyWorkerSurvivorCue] = []
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
                HealthyWorkerSurvivorCue(
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
