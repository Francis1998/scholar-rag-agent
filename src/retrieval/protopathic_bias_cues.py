"""Offline ProtopathicBiasCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from related bias cue extractors.
Fills an Elicit/Consensus/AJE protopathic-bias cue extractors gap.
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
        "protopathic_bias",
        re.compile(
            r"\bprotopathic bias\b",
            re.IGNORECASE,
        ),
    ),
    (
        "protopathic_confounding",
        re.compile(
            r"\bprotopathic confounding\b",
            re.IGNORECASE,
        ),
    ),
    (
        "lag_time_bias",
        re.compile(
            r"\blag[- ]time bias\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class ProtopathicBiasCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class ProtopathicBiasCueExtractor:
    """Extract offline advisory cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[ProtopathicBiasCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[ProtopathicBiasCue] = []
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
                ProtopathicBiasCue(
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
