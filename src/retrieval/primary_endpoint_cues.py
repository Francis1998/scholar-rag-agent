"""Offline primary-endpoint cue extraction.

Detects advisory primary endpoint / primary outcome cues from
title/abstract/methods via deterministic patterns. Never calls the network.
Distinct from :class:`~retrieval.effect_size_hint.EffectSizeHintExtractor`
and :class:`~retrieval.sample_size_hint.SampleSizeHintExtractor`.
Fills an Elicit / Consensus / ClinicalTrials.gov / PaperQA primary-endpoint gap.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "methods", "method", "results")

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "primary_endpoint",
        re.compile(
            r"\bprimary\s+end[- ]?point\b|\bprimary\s+outcome\b|"
            r"\bprimary\s+efficacy\s+end[- ]?point\b",
            re.IGNORECASE,
        ),
    ),
    (
        "co_primary",
        re.compile(r"\bco[- ]primary\s+(?:end[- ]?point|outcome)s?\b", re.IGNORECASE),
    ),
    (
        "secondary_endpoint",
        re.compile(
            r"\bsecondary\s+end[- ]?point\b|\bsecondary\s+outcome\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class PrimaryEndpointCue:
    """Advisory primary-endpoint cues for one paper row."""

    paper_id: str
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class PrimaryEndpointCueExtractor:
    """Extract primary/co-primary/secondary endpoint cues offline."""

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[PrimaryEndpointCue, ...]:
        """Return endpoint cues for ``papers``."""
        if not papers:
            return ()
        results: list[PrimaryEndpointCue] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            cues: list[str] = []
            matched: list[str] = []
            if text:
                for label, pattern in _PATTERNS:
                    match = pattern.search(text)
                    if not match:
                        continue
                    cues.append(label)
                    matched.append(match.group(0).strip())
            results.append(
                PrimaryEndpointCue(
                    paper_id=paper_id,
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
