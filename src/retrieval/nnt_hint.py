"""Offline NNT / NNH / ARR cue extraction from abstracts.

Extracts advisory number-needed-to-treat (NNT), number-needed-to-harm
(NNH), and absolute risk reduction (ARR) numeric hints from
title/abstract/results text via deterministic phrase patterns. Never calls
the network. Distinct from
:class:`~retrieval.effect_size_hint.EffectSizeHintExtractor` (Cohen's d /
OR / HR / RR / AUC),
:class:`~retrieval.sample_size_hint.SampleSizeHintExtractor` (N= sample-size
integers), and
:class:`~retrieval.p_value_hint.PValueHintExtractor` (p-value cues).
Fills an Elicit / Consensus / SciSpace / PaperQA NNT / NNH / ARR
surfacing gap without an LLM call. Optional later narrative can use
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "results")

_NUM = r"\d+(?:\.\d+)?"

_NNT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "NNT",
        re.compile(
            rf"\b(?:number\s+needed\s+to\s+treat|NNT)\s*"
            rf"(?:=|:|\bof\b|\bwas\b)?\s*"
            rf"(?P<value>{_NUM})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "NNH",
        re.compile(
            rf"\b(?:number\s+needed\s+to\s+harm|NNH)\s*"
            rf"(?:=|:|\bof\b|\bwas\b)?\s*"
            rf"(?P<value>{_NUM})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ARR",
        re.compile(
            rf"\b(?:absolute\s+risk\s+reduction|ARR)\s*"
            rf"(?:=|:|\bof\b|\bwas\b)?\s*"
            rf"(?P<value>{_NUM})\s*%?",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class NumberNeededToTreatHint:
    """Advisory NNT / NNH / ARR hint for one paper row."""

    paper_id: str
    metric: str | None
    value: float | None
    matched_cue: str


class NumberNeededToTreatHintExtractor:
    """Extract NNT / NNH / ARR numeric hints from abstracts offline.

    Offline heuristic NNT extractor for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace/PaperQA
    NNT / NNH / ARR gap. Distinct from
    :class:`~retrieval.effect_size_hint.EffectSizeHintExtractor`,
    :class:`~retrieval.sample_size_hint.SampleSizeHintExtractor`, and
    :class:`~retrieval.p_value_hint.PValueHintExtractor`. Never mutates
    inputs and never calls the network.
    """

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[NumberNeededToTreatHint, ...]:
        """Return NNT / NNH / ARR hints for ``papers``.

        Reads ``abstract``, ``title``, ``summary``, and ``results`` string
        fields. Prefers NNT, then NNH, then ARR when multiple cues match.
        Empty input yields an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[NumberNeededToTreatHint] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            metric, value, cue = self._parse(text)
            results.append(
                NumberNeededToTreatHint(
                    paper_id=paper_id,
                    metric=metric,
                    value=value,
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
    def _parse(text: str) -> tuple[str | None, float | None, str]:
        if not text:
            return None, None, ""
        for metric, pattern in _NNT_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            return metric, float(match.group("value")), match.group(0).strip()
        return None, None, ""
