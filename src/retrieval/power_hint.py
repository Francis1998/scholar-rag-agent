"""Offline statistical power / sample-size calculation cue extraction.

Detects advisory power analysis cues (80% power, power calculation,
sample size calculation) from title/abstract/methods via deterministic
patterns. Never calls the network. Distinct from
:class:`~retrieval.sample_size_hint.SampleSizeHintExtractor`,
:class:`~retrieval.p_value_hint.PValueHintExtractor`, and
:class:`~retrieval.confidence_interval_hint.ConfidenceIntervalHintExtractor`.
Fills an Elicit / Consensus / SciSpace / PaperQA power-analysis gap.
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
        "power_percent",
        re.compile(
            r"\b(\d{1,3})\s*%\s*power\b|\bpowered\s+(?:at|to)\s+(\d{1,3})\s*%",
            re.IGNORECASE,
        ),
    ),
    (
        "power_calculation",
        re.compile(
            r"\bpower\s+(?:calculation|analysis|estimate)\b|"
            r"\bsample[- ]size\s+calculation\b|"
            r"\ba\s+priori\s+power\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class StatisticalPowerHint:
    """Advisory statistical-power cues for one paper row."""

    paper_id: str
    power_percent: int | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class StatisticalPowerHintExtractor:
    """Extract statistical power / sample-size calculation cues offline."""

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[StatisticalPowerHint, ...]:
        """Return power hints for ``papers``."""
        if not papers:
            return ()
        results: list[StatisticalPowerHint] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            cues: list[str] = []
            matched: list[str] = []
            power_percent: int | None = None
            if text:
                for label, pattern in _PATTERNS:
                    match = pattern.search(text)
                    if not match:
                        continue
                    cues.append(label)
                    matched.append(match.group(0).strip())
                    if label == "power_percent":
                        raw = match.group(1) or match.group(2)
                        if raw is not None:
                            power_percent = int(raw)
            results.append(
                StatisticalPowerHint(
                    paper_id=paper_id,
                    power_percent=power_percent,
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
