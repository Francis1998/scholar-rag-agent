"""Offline attrition/dropout cue extraction (CONSORT-style).

Detects advisory attrition and dropout cues from title/abstract/methods via
deterministic patterns. Never calls the network. Distinct from
:class:`~retrieval.sample_size_hint.SampleSizeHintExtractor` and
:class:`~retrieval.intention_to_treat_cues.IntentionToTreatCueExtractor`.
Fills an Elicit / Consensus / Cochrane / CONSORT attrition-reporting gap.
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
        "dropout",
        re.compile(
            r"\bdrop[- ]?outs?\b|\battrit(?:ion|ed)\b|\blost to follow[- ]?up\b",
            re.IGNORECASE,
        ),
    ),
    (
        "withdrawal",
        re.compile(
            r"\bwithdraw(?:al|n|als|ew)?\b|\bwithdrew\b|\bdiscontinu(?:ed|ation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "completion_rate",
        re.compile(r"\bcompletion rate\b|\bcompleted the (?:study|trial)\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class AttritionRateCue:
    """Advisory attrition cues for one paper row."""

    paper_id: str
    attrition_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class AttritionRateCueExtractor:
    """Extract attrition/dropout/withdrawal cues offline."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[AttritionRateCue, ...]:
        """Return attrition cues for ``papers``."""

        if not papers:
            return ()
        results: list[AttritionRateCue] = []
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
                AttritionRateCue(
                    paper_id=paper_id,
                    attrition_kind=kind,
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
