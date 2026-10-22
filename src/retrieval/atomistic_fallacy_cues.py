"""Offline AtomisticFallacyCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from `EcologicalFallacyCueExtractor` / `SimpsonParadoxCueExtractor`.
Fills an Elicit/Consensus/Connected Papers atomistic-fallacy cue extractors gap.
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
        "atomistic_fallacy",
        re.compile(
            r"\batomistic fallacy\b",
            re.IGNORECASE,
        ),
    ),
    (
        "individual_to_group",
        re.compile(
            r"\bindividual[- ]level (?:results?|effects?) (?:to|onto) (?:group|population)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cross_level_inference",
        re.compile(
            r"\bcross[- ]level inference\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class AtomisticFallacyCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class AtomisticFallacyCueExtractor:
    """Extract offline advisory cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[AtomisticFallacyCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[AtomisticFallacyCue] = []
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
                AtomisticFallacyCue(
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
