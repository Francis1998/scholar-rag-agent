"""Offline I2 / heterogeneity cue extraction from abstracts.

Extracts advisory meta-analysis heterogeneity cues such as ``I2 = 45%``,
``I^2=60%``, and phrases like ``substantial heterogeneity`` from
title/abstract/results text via deterministic patterns. Never calls the
network. Distinct from
:class:`~retrieval.effect_size_hint.EffectSizeHintExtractor` (Cohen's d /
OR / HR / RR / AUC),
:class:`~retrieval.confidence_interval_hint.ConfidenceIntervalHintExtractor`
(CI bounds), and
:class:`~retrieval.p_value_hint.PValueHintExtractor` (p-value cues).
Fills an Elicit / Consensus / SciSpace / PaperQA heterogeneity / I2
surfacing gap without an LLM call. Optional later narrative can use
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "results")

_NUM = r"\d+(?:\.\d+)?"

# Prefer numeric I2 forms (ASCII I2 / I^2 and unicode superscript-2), then
# qualitative heterogeneity phrases.
_I2_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "i2",
        re.compile(
            rf"\bI\s*(?:\^?2|\u00b2)\s*"
            rf"(?:=|:|\bwas\b|\bof\b)?\s*"
            rf"(?P<value>{_NUM})\s*%?",
            re.IGNORECASE,
        ),
    ),
    (
        "heterogeneity",
        re.compile(
            r"\b(?:substantial|considerable|significant|high|moderate|low|"
            r"marked|important)\s+heterogeneity\b"
            r"|\bheterogeneity\s+(?:was|is)\s+"
            r"(?:substantial|considerable|significant|high|moderate|low|"
            r"marked|important)\b"
            r"|\bbetween[- ]study\s+heterogeneity\b"
            r"|\bheterogeneity\s+(?:across|among)\s+studies\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class HeterogeneityI2Hint:
    """Advisory I2 / heterogeneity hint for one paper row."""

    paper_id: str
    i2_percent: float | None
    cue_kind: str | None
    matched_cue: str


class HeterogeneityI2HintExtractor:
    """Extract I2 / heterogeneity cues from abstracts offline.

    Offline heuristic I2 / heterogeneity extractor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines —
    Elicit/Consensus/SciSpace/PaperQA meta-analysis heterogeneity gap.
    Distinct from
    :class:`~retrieval.effect_size_hint.EffectSizeHintExtractor`,
    :class:`~retrieval.confidence_interval_hint.ConfidenceIntervalHintExtractor`,
    and :class:`~retrieval.p_value_hint.PValueHintExtractor`. Never mutates
    inputs and never calls the network.
    """

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[HeterogeneityI2Hint, ...]:
        """Return I2 / heterogeneity hints for ``papers``.

        Reads ``abstract``, ``title``, ``summary``, and ``results`` string
        fields. Prefers numeric I2 matches over qualitative heterogeneity
        phrases. Empty input yields an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[HeterogeneityI2Hint] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            i2_percent, cue_kind, cue = self._parse(text)
            results.append(
                HeterogeneityI2Hint(
                    paper_id=paper_id,
                    i2_percent=i2_percent,
                    cue_kind=cue_kind,
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
    def _parse(text: str) -> tuple[float | None, str | None, str]:
        if not text:
            return None, None, ""
        for kind, pattern in _I2_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            value: float | None = None
            if "value" in match.groupdict() and match.group("value"):
                value = float(match.group("value"))
            return value, kind, match.group(0).strip()
        return None, None, ""
