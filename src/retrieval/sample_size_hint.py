"""Offline sample-size (N=) hint extraction from abstracts.

Extracts advisory sample-size integers from title/abstract text via
deterministic ``N=`` / ``n =`` / ``sample size of`` cues. Never calls the
network. Distinct from :class:`~retrieval.method_extract_card.MethodExtractCard`
(PICO / study-design cards without sample-size N). Fills an Elicit /
Consensus / SciSpace sample-size surfacing gap without an LLM call.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary")

# Prefer explicit N=/n= forms first; then "sample size of/was/=" phrases.
_N_EQUALS = re.compile(
    r"\b[Nn]\s*=\s*(?P<n>\d{1,7})\b",
)
_SAMPLE_SIZE = re.compile(
    r"\bsample\s+size\s+(?:of|was|=|:)?\s*(?P<n>\d{1,7})\b",
    re.IGNORECASE,
)
_ENROLLED = re.compile(
    r"\b(?:enrolled|included|analyzed|randomised|randomized)\s+"
    r"(?P<n>\d{1,7})\s+(?:patients?|participants?|subjects?|adults?|"
    r"children|records?|mice|rats)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SampleSizeHint:
    """Advisory sample-size hint for one paper row."""

    paper_id: str
    sample_size: int | None
    matched_cue: str


class SampleSizeHintExtractor:
    """Extract sample-size (N=) hints from abstract text offline.

    Offline heuristic sample-size extractor for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace sample-size
    gap. Distinct from :class:`~retrieval.method_extract_card.MethodExtractCard`
    (study-design cues without N). Never mutates inputs and never calls the
    network.
    """

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[SampleSizeHint, ...]:
        """Return sample-size hints for ``papers``.

        Reads ``abstract``, ``title``, and ``summary`` string fields. When
        multiple cues match, prefers an explicit ``N=`` / ``n=`` match over
        ``sample size`` phrases, then enrollment phrasing. Empty input yields
        an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[SampleSizeHint] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            sample_size, cue = self._parse(text)
            results.append(
                SampleSizeHint(
                    paper_id=paper_id,
                    sample_size=sample_size,
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
    def _parse(text: str) -> tuple[int | None, str]:
        if not text:
            return None, ""
        match = _N_EQUALS.search(text)
        if match:
            return int(match.group("n")), match.group(0).strip()
        match = _SAMPLE_SIZE.search(text)
        if match:
            return int(match.group("n")), match.group(0).strip()
        match = _ENROLLED.search(text)
        if match:
            return int(match.group("n")), match.group(0).strip()
        return None, ""
