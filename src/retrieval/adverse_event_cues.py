"""Offline adverse-event reporting cue extraction.

Detects advisory adverse event / SAE cues from title/abstract/methods via
deterministic patterns. Never calls the network. Distinct from
:class:`~retrieval.blinding_status_cues.BlindingStatusCueExtractor` and
:class:`~retrieval.publication_bias_cues.PublicationBiasCueExtractor`.
Fills an Elicit / Consensus / Cochrane adverse-event reporting gap.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "methods", "method", "results", "safety")

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "serious_adverse_event",
        re.compile(r"\bserious adverse events?\b|\bSAEs?\b", re.IGNORECASE),
    ),
    (
        "adverse_event",
        re.compile(r"\badverse events?\b|\btreatment[- ]emergent\b", re.IGNORECASE),
    ),
    (
        "safety_endpoint",
        re.compile(r"\bsafety (?:endpoint|outcome|profile)\b|\btolerability\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class AdverseEventCue:
    """Advisory adverse-event cues for one paper row."""

    paper_id: str
    adverse_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class AdverseEventCueExtractor:
    """Extract adverse-event reporting cues offline."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[AdverseEventCue, ...]:
        """Return adverse-event cues for ``papers``."""

        if not papers:
            return ()
        results: list[AdverseEventCue] = []
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
                AdverseEventCue(
                    paper_id=paper_id,
                    adverse_kind=kind,
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
