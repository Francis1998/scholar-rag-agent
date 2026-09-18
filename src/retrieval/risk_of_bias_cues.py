"""Offline Cochrane-style risk-of-bias cue extraction from paper text.

Detects advisory RoB domain cues such as randomization, allocation
concealment, blinding / double-blind, and incomplete outcome / attrition
from title/abstract/methods text via deterministic phrase patterns. Never
calls the network. Distinct from
:class:`~retrieval.study_limitations.StudyLimitationCueExtractor`
(generic limitation categories),
:class:`~retrieval.conflict_of_interest.ConflictOfInterestFlagger`
(COI disclosures), and
:class:`~retrieval.method_extract_card.MethodExtractCard`
(PICO / study-design cards). Fills an Elicit / Consensus / SciSpace /
PaperQA Cochrane risk-of-bias surfacing gap without an LLM call. Optional
later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = (
    "abstract",
    "title",
    "summary",
    "methods",
    "method",
    "results",
)

_ROB_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "randomization",
        re.compile(
            r"\brandom(?:ly|i[sz]ed|isation|ization)\b"
            r"|\brandom\s+allocation\b"
            r"|\brandom\s+assignment\b"
            r"|\brandom\s+sequence\b"
            r"|\bsequence\s+generat(?:ion|ed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "allocation_concealment",
        re.compile(
            r"\ballocation\s+concealment\b"
            r"|\bconcealed\s+allocation\b"
            r"|\bopaque\s+envelope(?:s)?\b"
            r"|\bsealed\s+(?:opaque\s+)?envelope(?:s)?\b"
            r"|\bcentral\s+(?:telephone\s+)?randomi[sz]ation\b",
            re.IGNORECASE,
        ),
    ),
    (
        "blinding",
        re.compile(
            r"\b(?:double|single|triple)[- ]blind(?:ed)?\b"
            r"|\bblinding\b"
            r"|\bblinded\b"
            r"|\bmasking\b"
            r"|\bmasked\s+(?:assessors?|outcome)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "incomplete_outcome",
        re.compile(
            r"\bincomplete\s+outcome\s+data\b"
            r"|\battrition\s+bias\b"
            r"|\battrition\b"
            r"|\bloss\s+to\s+follow[- ]?up\b"
            r"|\bintention[- ]to[- ]treat\b"
            r"|\bITT\s+analysis\b"
            r"|\bmissing\s+outcome\s+data\b"
            r"|\bdrop[- ]?outs?\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class RiskOfBiasCue:
    """Advisory Cochrane-style risk-of-bias cues for one paper row."""

    paper_id: str
    domains: tuple[str, ...]
    matched_cues: tuple[str, ...]
    flagged: bool


class RiskOfBiasCueExtractor:
    """Extract Cochrane-style risk-of-bias cues from paper text offline.

    Offline heuristic RoB domain scan for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace/PaperQA
    Cochrane risk-of-bias gap. Distinct from
    :class:`~retrieval.study_limitations.StudyLimitationCueExtractor`,
    :class:`~retrieval.conflict_of_interest.ConflictOfInterestFlagger`, and
    :class:`~retrieval.method_extract_card.MethodExtractCard`. Never mutates
    inputs and never calls the network.
    """

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[RiskOfBiasCue, ...]:
        """Return risk-of-bias cues for ``papers``.

        Reads ``abstract``, ``title``, ``summary``, ``methods`` / ``method``,
        and ``results`` string fields. Domains are returned in discovery
        order without duplicates. Empty input yields an empty tuple. Inputs
        are not mutated.
        """
        if not papers:
            return ()

        results: list[RiskOfBiasCue] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            domains: list[str] = []
            cues: list[str] = []
            if text:
                for label, pattern in _ROB_PATTERNS:
                    match = pattern.search(text)
                    if match:
                        domains.append(label)
                        cues.append(match.group(0).strip())
            results.append(
                RiskOfBiasCue(
                    paper_id=paper_id,
                    domains=tuple(domains),
                    matched_cues=tuple(cues),
                    flagged=bool(domains),
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
