"""Offline preregistration / registry cue detection from paper text.

Detects advisory clinicaltrials.gov / OSF / ISRCTN / preregistration /
pre-registration cues in local text fields. Never calls the network.
Distinct from
:class:`~retrieval.funding_disclosure.FundingDisclosureFlagger` (funder/grant
cues) and
:class:`~retrieval.prisma_screening.PrismaScreeningChecklist` (HITL PRISMA
screening rows). Fills an Elicit / Consensus / SciSpace / PaperQA
preregistration-cue surfacing gap without an LLM call. Optional later
narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
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
    "registration",
    "trial_registration",
    "acknowledgements",
    "acknowledgment",
)

_PREREG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "clinicaltrials.gov",
        re.compile(
            r"\bclinicaltrials\.gov\b"
            r"|\bNCT\d{8}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "OSF",
        re.compile(
            r"\bOSF\b"
            r"|\bosf\.io\b"
            r"|\bOpen\s+Science\s+Framework\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ISRCTN",
        re.compile(r"\bISRCTN\d+\b", re.IGNORECASE),
    ),
    (
        "preregistration",
        re.compile(
            r"\bpreregistration\b"
            r"|\bpre-registration\b"
            r"|\bpreregistered\b"
            r"|\bpre-registered\b"
            r"|\bprospectively\s+registered\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class PreregistrationFlag:
    """Advisory preregistration / registry flag for one paper row."""

    paper_id: str
    flagged: bool
    reasons: tuple[str, ...]
    matched_cues: tuple[str, ...]


class PreregistrationFlagDetector:
    """Detect clinicaltrials.gov / OSF / ISRCTN / preregistration cues offline.

    Offline heuristic preregistration scan for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace/PaperQA
    preregistration-cue gap. Distinct from
    :class:`~retrieval.funding_disclosure.FundingDisclosureFlagger` and
    :class:`~retrieval.prisma_screening.PrismaScreeningChecklist`. Never
    mutates inputs and never calls the network.
    """

    def detect(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[PreregistrationFlag, ...]:
        """Return preregistration flags for ``papers``.

        Reads ``abstract``, ``title``, ``summary``, ``methods``,
        ``registration`` / ``trial_registration``, and acknowledgement string
        fields. Empty input yields an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[PreregistrationFlag] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            reasons: list[str] = []
            cues: list[str] = []
            if text:
                for label, pattern in _PREREG_PATTERNS:
                    match = pattern.search(text)
                    if match:
                        reasons.append(f"Matched preregistration cue: {label}")
                        cues.append(match.group(0).strip())
            results.append(
                PreregistrationFlag(
                    paper_id=paper_id,
                    flagged=bool(reasons),
                    reasons=tuple(reasons),
                    matched_cues=tuple(cues),
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
