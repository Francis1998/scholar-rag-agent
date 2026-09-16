"""Offline confidence-interval numeric hint extraction from abstracts.

Extracts advisory 95% CI / confidence interval cues such as
``95% CI 1.2-3.4``, ``95% CI: 0.8 to 1.1``, and ``confidence interval
(0.45, 0.62)`` from title/abstract/results text via deterministic phrase
patterns. Never calls the network. Distinct from
:class:`~retrieval.effect_size_hint.EffectSizeHintExtractor` (Cohen's d /
OR / HR / RR / AUC) and
:class:`~retrieval.p_value_hint.PValueHintExtractor` (p-value cues).
Fills an Elicit / Consensus / SciSpace / PaperQA confidence-interval
surfacing gap without an LLM call. Optional later narrative can use
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "results")

_NUM = r"-?\d+(?:\.\d+)?"

_CI_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "95% CI",
        re.compile(
            rf"\b(?P<level>95)\s*%\s*(?:CI|C\.I\.|confidence\s+interval)\s*"
            rf"[=:]?\s*"
            rf"(?:\[|\()?\s*(?P<low>{_NUM})\s*"
            rf"(?:-|to|,)\s*"
            rf"(?P<high>{_NUM})\s*(?:\]|\))?",
            re.IGNORECASE,
        ),
    ),
    (
        "confidence interval",
        re.compile(
            rf"\bconfidence\s+interval(?:s)?\s*"
            rf"(?:of\s+)?(?:\[|\()?\s*(?P<low>{_NUM})\s*"
            rf"(?:-|to|,)\s*"
            rf"(?P<high>{_NUM})\s*(?:\]|\))?",
            re.IGNORECASE,
        ),
    ),
    (
        "CI",
        re.compile(
            rf"\bCI\s*[=:]?\s*"
            rf"(?:\[|\()?\s*(?P<low>{_NUM})\s*"
            rf"(?:-|to|,)\s*"
            rf"(?P<high>{_NUM})\s*(?:\]|\))?",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class ConfidenceIntervalHint:
    """Advisory confidence-interval hint for one paper row."""

    paper_id: str
    level: int | None
    low: float | None
    high: float | None
    matched_cue: str


class ConfidenceIntervalHintExtractor:
    """Extract 95% CI / confidence interval numeric hints offline.

    Offline heuristic CI extractor for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace/PaperQA
    confidence-interval gap. Distinct from
    :class:`~retrieval.effect_size_hint.EffectSizeHintExtractor` and
    :class:`~retrieval.p_value_hint.PValueHintExtractor`. Never mutates
    inputs and never calls the network.
    """

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[ConfidenceIntervalHint, ...]:
        """Return confidence-interval hints for ``papers``.

        Reads ``abstract``, ``title``, ``summary``, and ``results`` string
        fields. When multiple cues match, prefers labeled ``95% CI``, then
        ``confidence interval``, then bare ``CI``. Empty input yields an
        empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[ConfidenceIntervalHint] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            level, low, high, cue = self._parse(text)
            results.append(
                ConfidenceIntervalHint(
                    paper_id=paper_id,
                    level=level,
                    low=low,
                    high=high,
                    matched_cue=cue,
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

    @staticmethod
    def _parse(text: str) -> tuple[int | None, float | None, float | None, str]:
        if not text:
            return None, None, None, ""
        for label, pattern in _CI_PATTERNS:
            match = pattern.search(text)
            if match:
                level: int | None = None
                if "level" in match.groupdict() and match.group("level"):
                    level = int(match.group("level"))
                elif label == "95% CI" or label.startswith("95"):
                    level = 95
                return (
                    level,
                    float(match.group("low")),
                    float(match.group("high")),
                    match.group(0).strip(),
                )
        return None, None, None, ""
