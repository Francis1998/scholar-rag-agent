"""Offline p-value numeric hint extraction from abstracts.

Extracts advisory p-value / P-value cues such as ``p < 0.05``, ``p=0.01``,
and ``P-value = 0.003`` from title/abstract/results text via deterministic
phrase patterns. Never calls the network. Distinct from
:class:`~retrieval.effect_size_hint.EffectSizeHintExtractor` (Cohen's d / OR /
HR / RR / AUC) and
:class:`~retrieval.sample_size_hint.SampleSizeHintExtractor` (N= sample-size
integers). Fills an Elicit / Consensus / SciSpace / PaperQA p-value
surfacing gap without an LLM call. Optional later narrative can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "results")

# Prefer explicit inequality / equality forms; capture operator + numeric value.
_P_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b[Pp](?:\s*[- ]?\s*values?)?\s*"
        r"(?P<op><=|>=|<|>|=|:)\s*"
        r"(?P<value>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\b",
    ),
    re.compile(
        r"\b[Pp]\s*(?P<op><=|>=|<|>|=)\s*"
        r"(?P<value>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\b",
    ),
)


@dataclass(frozen=True)
class PValueHint:
    """Advisory p-value hint for one paper row."""

    paper_id: str
    operator: str | None
    value: float | None
    matched_cue: str


class PValueHintExtractor:
    """Extract p-value / P-value numeric hints from abstracts offline.

    Offline heuristic p-value extractor for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace/PaperQA
    p-value gap. Distinct from
    :class:`~retrieval.effect_size_hint.EffectSizeHintExtractor` and
    :class:`~retrieval.sample_size_hint.SampleSizeHintExtractor`. Never
    mutates inputs and never calls the network.
    """

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[PValueHint, ...]:
        """Return p-value hints for ``papers``.

        Reads ``abstract``, ``title``, ``summary``, and ``results`` string
        fields. When multiple cues match, prefers the first match in text
        order. Empty input yields an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[PValueHint] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            operator, value, cue = self._parse(text)
            results.append(
                PValueHint(
                    paper_id=paper_id,
                    operator=operator,
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
        best: re.Match[str] | None = None
        for pattern in _P_VALUE_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            if best is None or match.start() < best.start():
                best = match
        if best is None:
            return None, None, ""
        op = best.group("op")
        if op == ":":
            op = "="
        return op, float(best.group("value")), best.group(0).strip()
