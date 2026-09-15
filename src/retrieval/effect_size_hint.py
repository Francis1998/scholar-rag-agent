"""Offline effect-size numeric hint extraction from abstracts.

Extracts advisory Cohen's d / odds ratio (OR) / hazard ratio (HR) /
relative risk (RR) / AUC numeric hints from title/abstract text via
deterministic phrase cues. Never calls the network. Distinct from
:class:`~retrieval.sample_size_hint.SampleSizeHintExtractor` (N= sample-size
integers). Fills an Elicit / Consensus / SciSpace / PaperQA effect-size
surfacing gap without an LLM call. Optional later narrative can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "results")

# Prefer labeled metric forms; capture metric kind + numeric value.
_EFFECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "cohens_d",
        re.compile(
            r"\b(?:Cohen(?:'s)?\s*d|cohens?\s*d)\s*[=:]?\s*"
            r"(?P<value>-?\d+(?:\.\d+)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "OR",
        re.compile(
            r"\b(?:odds\s+ratio|OR)\s*[=:]?\s*(?P<value>\d+(?:\.\d+)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "HR",
        re.compile(
            r"\b(?:hazard\s+ratio|HR)\s*[=:]?\s*(?P<value>\d+(?:\.\d+)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "RR",
        re.compile(
            r"\b(?:relative\s+risk|risk\s+ratio|RR)\s*[=:]?\s*"
            r"(?P<value>\d+(?:\.\d+)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "AUC",
        re.compile(
            r"\b(?:AUC|AUROC|area\s+under\s+the\s+(?:ROC\s+)?curve)\s*[=:]?\s*"
            r"(?P<value>\d+(?:\.\d+)?)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class EffectSizeHint:
    """Advisory effect-size hint for one paper row."""

    paper_id: str
    metric: str | None
    value: float | None
    matched_cue: str


class EffectSizeHintExtractor:
    """Extract Cohen's d / OR / HR / RR / AUC hints from abstracts offline.

    Offline heuristic effect-size extractor for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace/PaperQA
    effect-size gap. Distinct from
    :class:`~retrieval.sample_size_hint.SampleSizeHintExtractor` (sample-size
    N). Never mutates inputs and never calls the network.
    """

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[EffectSizeHint, ...]:
        """Return effect-size hints for ``papers``.

        Reads ``abstract``, ``title``, ``summary``, and ``results`` string
        fields. When multiple metrics match, prefers Cohen's d, then OR, HR,
        RR, then AUC (first match in that order). Empty input yields an empty
        tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[EffectSizeHint] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            metric, value, cue = self._parse(text)
            results.append(
                EffectSizeHint(
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
        for metric, pattern in _EFFECT_PATTERNS:
            match = pattern.search(text)
            if match:
                return metric, float(match.group("value")), match.group(0).strip()
        return None, None, ""
