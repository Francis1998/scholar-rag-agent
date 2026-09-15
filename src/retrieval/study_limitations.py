"""Offline study-limitation cue extraction from abstract/discussion text.

Extracts advisory limitation categories (sample size, generalizability,
confounding, bias) from local paper text via deterministic phrase cues.
Never calls the network. Distinct from
:class:`~retrieval.sample_size_hint.SampleSizeHintExtractor` (N= integers),
:class:`~retrieval.conflict_of_interest.ConflictOfInterestFlagger` (COI
disclosures), and :class:`~retrieval.method_extract_card.MethodExtractCard`
(PICO / study-design cards). Fills an Elicit / Consensus / SciSpace /
PaperQA limitation-cue surfacing gap without an LLM call. Optional later
narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = (
    "abstract",
    "discussion",
    "limitations",
    "limitation",
    "summary",
    "title",
)

# Category label -> patterns that surface that limitation cue.
_LIMITATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "sample_size",
        re.compile(
            r"\bsmall\s+sample\s+size\b"
            r"|\blimited\s+sample\s+size\b"
            r"|\bunderpowered\b"
            r"|\binsufficient(?:ly)?\s+powered\b"
            r"|\bsample\s+size\s+(?:was\s+)?(?:too\s+)?(?:small|limited)\b"
            r"|\bfew\s+(?:patients?|participants?|subjects?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "generalizability",
        re.compile(
            r"\bgeneralizabilit(?:y|ies)\b"
            r"|\bgeneralisabilit(?:y|ies)\b"
            r"|\blimited\s+external\s+validity\b"
            r"|\bmay\s+not\s+generalize\b"
            r"|\bmay\s+not\s+generalise\b"
            r"|\bnot\s+(?:be\s+)?generalizable\b"
            r"|\bnot\s+(?:be\s+)?generalisable\b"
            r"|\bsingle[- ]center\b|\bsingle[- ]centre\b"
            r"|\bsingle[- ]site\b",
            re.IGNORECASE,
        ),
    ),
    (
        "confounding",
        re.compile(
            r"\bconfound(?:ing|ers?|ed)\b"
            r"|\bunmeasured\s+confound",
            re.IGNORECASE,
        ),
    ),
    (
        "bias",
        re.compile(
            r"\bselection\s+bias\b"
            r"|\brecall\s+bias\b"
            r"|\bpublication\s+bias\b"
            r"|\bmeasurement\s+bias\b"
            r"|\binformation\s+bias\b"
            r"|\bresponder\s+bias\b"
            r"|\bsurvivor(?:ship)?\s+bias\b"
            r"|\brisk\s+of\s+bias\b"
            r"|\bpotential\s+bias\b"
            r"|\bbias(?:es)?\s+(?:cannot|may|might|could)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class StudyLimitationCue:
    """Advisory study-limitation cues for one paper row."""

    paper_id: str
    categories: tuple[str, ...]
    matched_cues: tuple[str, ...]
    flagged: bool


class StudyLimitationCueExtractor:
    """Extract study-limitation cues from abstract/discussion text offline.

    Offline heuristic limitation-cue scan for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace/PaperQA
    limitation-cue gap. Distinct from
    :class:`~retrieval.sample_size_hint.SampleSizeHintExtractor`,
    :class:`~retrieval.conflict_of_interest.ConflictOfInterestFlagger`, and
    :class:`~retrieval.method_extract_card.MethodExtractCard`. Never mutates
    inputs and never calls the network.
    """

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[StudyLimitationCue, ...]:
        """Return study-limitation cues for ``papers``.

        Reads ``abstract``, ``discussion``, ``limitations`` / ``limitation``,
        ``summary``, and ``title`` string fields. Categories are returned in
        discovery order without duplicates. Empty input yields an empty tuple.
        Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[StudyLimitationCue] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            categories: list[str] = []
            cues: list[str] = []
            if text:
                for label, pattern in _LIMITATION_PATTERNS:
                    match = pattern.search(text)
                    if match:
                        categories.append(label)
                        cues.append(match.group(0).strip())
            results.append(
                StudyLimitationCue(
                    paper_id=paper_id,
                    categories=tuple(categories),
                    matched_cues=tuple(cues),
                    flagged=bool(categories),
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
