"""Offline intention-to-treat (ITT) / per-protocol cue extraction.

Detects advisory ITT, modified ITT, and per-protocol analysis cues from
title/abstract/methods via deterministic patterns. Never calls the network.
Distinct from :class:`~retrieval.blinding_status_cues.BlindingStatusCueExtractor`
and :class:`~retrieval.risk_of_bias_cues.RiskOfBiasCueExtractor`.
Fills an Elicit / Consensus / Cochrane / PaperQA ITT-analysis gap.
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
        "intention_to_treat",
        re.compile(
            r"\bintention[- ]to[- ]treat\b|\bintent[- ]to[- ]treat\b|\bITT\b",
            re.IGNORECASE,
        ),
    ),
    (
        "modified_itt",
        re.compile(r"\bmodified\s+intention[- ]to[- ]treat\b|\bmITT\b", re.IGNORECASE),
    ),
    (
        "per_protocol",
        re.compile(r"\bper[- ]protocol\b|\bPP\s+analysis\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class IntentionToTreatCue:
    """Advisory ITT / per-protocol cues for one paper row."""

    paper_id: str
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class IntentionToTreatCueExtractor:
    """Extract ITT / mITT / per-protocol analysis cues offline."""

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[IntentionToTreatCue, ...]:
        """Return ITT cues for ``papers``."""
        if not papers:
            return ()
        results: list[IntentionToTreatCue] = []
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
                IntentionToTreatCue(
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
