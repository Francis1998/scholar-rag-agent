"""Offline SurrogateEndpointCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network. Distinct from PrimaryEndpointCueExtractor and EffectSizeHintExtractor.
Fills an Elicit / Consensus / CONSORT evidence-methods gap.
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
        "surrogate_endpoint",
        re.compile(
            r"\bsurrogate (?:endpoint|outcome|marker)\b|\bintermediate endpoint\b",
            re.IGNORECASE,
        ),
    ),
    (
        "clinical_endpoint",
        re.compile(
            r"\bclinical (?:endpoint|outcome)\b|\bhard (?:clinical )?endpoint\b",
            re.IGNORECASE,
        ),
    ),
    (
        "biomarker_endpoint",
        re.compile(
            r"\bbiomarker (?:endpoint|outcome)\b|\bpseudosurrogate\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class SurrogateEndpointCue:
    """Advisory cues for one paper row."""

    paper_id: str
    endpoint_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class SurrogateEndpointCueExtractor:
    """Extract offline evidence cues."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[SurrogateEndpointCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[SurrogateEndpointCue] = []
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
                SurrogateEndpointCue(
                    paper_id=paper_id,
                    endpoint_kind=kind,
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
