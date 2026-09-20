"""Offline blinding-status cue extraction (single/double/triple blind).

Detects advisory blinding cues from title/abstract/methods via deterministic
patterns. Never calls the network. Distinct from
:class:`~retrieval.risk_of_bias_cues.RiskOfBiasCueExtractor` and
:class:`~retrieval.preregistration_flag.PreregistrationFlagDetector`.
Fills an Elicit / Consensus / Cochrane / PaperQA blinding-status gap.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "methods", "method")

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "triple_blind",
        re.compile(r"\btriple[- ]blind(?:ed)?\b|\btriple[- ]masked\b", re.IGNORECASE),
    ),
    (
        "double_blind",
        re.compile(r"\bdouble[- ]blind(?:ed)?\b|\bdouble[- ]masked\b", re.IGNORECASE),
    ),
    (
        "single_blind",
        re.compile(r"\bsingle[- ]blind(?:ed)?\b|\bsingle[- ]masked\b", re.IGNORECASE),
    ),
    (
        "open_label",
        re.compile(r"\bopen[- ]label\b|\bunblinded\b|\bnon[- ]blind(?:ed)?\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class BlindingStatusCue:
    """Advisory blinding-status cues for one paper row."""

    paper_id: str
    blinding_level: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class BlindingStatusCueExtractor:
    """Extract single/double/triple-blind and open-label cues offline."""

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[BlindingStatusCue, ...]:
        """Return blinding cues for ``papers``."""
        if not papers:
            return ()
        results: list[BlindingStatusCue] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            cues: list[str] = []
            matched: list[str] = []
            level: str | None = None
            if text:
                for label, pattern in _PATTERNS:
                    match = pattern.search(text)
                    if not match:
                        continue
                    cues.append(label)
                    matched.append(match.group(0).strip())
                    if level is None and label != "open_label":
                        level = label
                    elif level is None and label == "open_label":
                        level = "open_label"
            results.append(
                BlindingStatusCue(
                    paper_id=paper_id,
                    blinding_level=level,
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
