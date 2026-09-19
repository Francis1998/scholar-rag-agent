"""Offline publication-bias cue extraction from paper text.

Detects advisory publication-bias cues such as funnel plot asymmetry,
Egger's test, small-study effects, and trim-and-fill from title/abstract/
methods/results/discussion via deterministic phrase patterns. Never calls
the network. Distinct from
:class:`~retrieval.heterogeneity_i2_hint.HeterogeneityI2HintExtractor`,
:class:`~retrieval.risk_of_bias_cues.RiskOfBiasCueExtractor`, and
:class:`~retrieval.effect_size_hint.EffectSizeHintExtractor`.
Fills an Elicit / Consensus / SciSpace / PaperQA publication-bias gap.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.
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
        "funnel_plot",
        re.compile(
            r"\bfunnel\s+plot\b|\basymmetry\s+of\s+the\s+funnel\b",
            re.IGNORECASE,
        ),
    ),
    (
        "egger_test",
        re.compile(
            r"\begger(?:'s)?\s+test\b|\begger\s+regression\b",
            re.IGNORECASE,
        ),
    ),
    (
        "small_study",
        re.compile(
            r"\bsmall[- ]study\s+effects?\b|\bpublication\s+bias\b",
            re.IGNORECASE,
        ),
    ),
    (
        "trim_fill",
        re.compile(
            r"\btrim[- ]and[- ]fill\b|\btrim\s+and\s+fill\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class PublicationBiasCue:
    """Advisory publication-bias cues for one paper row."""

    paper_id: str
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class PublicationBiasCueExtractor:
    """Extract publication-bias cues from paper text offline."""

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[PublicationBiasCue, ...]:
        """Return publication-bias cues for ``papers``."""
        if not papers:
            return ()
        results: list[PublicationBiasCue] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            cues: list[str] = []
            matched: list[str] = []
            if text:
                for label, pattern in _PATTERNS:
                    match = pattern.search(text)
                    if match:
                        cues.append(label)
                        matched.append(match.group(0).strip())
            results.append(
                PublicationBiasCue(
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
