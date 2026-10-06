"""Offline SelectionBiasCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from related causal-inference cue extractors.
Fills an Elicit/Consensus/AJE selection-bias cue extractors gap.
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
        "selection_bias",
        re.compile(
            r"\bselection bias\b",
            re.IGNORECASE,
        ),
    ),
    (
        "selection_effect",
        re.compile(
            r"\bselection effect\b",
            re.IGNORECASE,
        ),
    ),
    (
        "collider_selection",
        re.compile(
            r"\bcollider selection\b",
            re.IGNORECASE,
        ),
    ),
    (
        "healthy_user_bias",
        re.compile(
            r"\bhealthy[- ]user bias\b",
            re.IGNORECASE,
        ),
    ),
    (
        "volunteer_bias",
        re.compile(
            r"\bvolunteer bias\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class SelectionBiasCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class SelectionBiasCueExtractor:
    """Extract offline advisory cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[SelectionBiasCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[SelectionBiasCue] = []
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
                SelectionBiasCue(
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
