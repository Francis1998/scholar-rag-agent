"""Offline open-data / data-availability cue flagging from paper text.

Detects advisory open-data cues such as ``data available``, ``open data``,
Zenodo, OSF/osf.io, Dryad, Figshare, ``github.com/...``, and
``supplementary data`` in local text fields. Never calls the network.
Distinct from :class:`~retrieval.code_availability.CodeAvailabilityBooster`
(code-host / source-code score boosting),
:class:`~retrieval.funding_disclosure.FundingDisclosureFlagger` (funder cues),
and
:class:`~retrieval.preregistration_flag.PreregistrationFlagDetector`
(registry / preregistration cues). Fills an Elicit / Consensus / SciSpace /
PaperQA open-data-availability surfacing gap without an LLM call. Optional
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
    "data_availability",
    "data_availability_statement",
    "availability",
    "supplementary",
    "supplemental",
    "acknowledgements",
    "acknowledgment",
)

_OPEN_DATA_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "data available",
        re.compile(
            r"\bdata\s+(?:are|is|were)?\s*available\b"
            r"|\bdata\s+availability\b"
            r"|\bopen\s+data\b"
            r"|\bdataset\s+available\b"
            r"|\bdata\s+deposited\b",
            re.IGNORECASE,
        ),
    ),
    (
        "zenodo",
        re.compile(r"\bzenodo(?:\.org)?\b", re.IGNORECASE),
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
        "dryad",
        re.compile(r"\bdryad(?:\.org)?\b", re.IGNORECASE),
    ),
    (
        "figshare",
        re.compile(r"\bfigshare(?:\.com)?\b", re.IGNORECASE),
    ),
    (
        "github.com",
        re.compile(r"\bgithub\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", re.IGNORECASE),
    ),
    (
        "supplementary data",
        re.compile(
            r"\bsupplementary\s+data\b"
            r"|\bsupplemental\s+data\b"
            r"|\bsupporting\s+data\b"
            r"|\bdata\s+in\s+the\s+supplement(?:ary|al)?\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class OpenDataAvailabilityFlag:
    """Advisory open-data availability flag for one paper row."""

    paper_id: str
    flagged: bool
    reasons: tuple[str, ...]
    matched_cues: tuple[str, ...]


class OpenDataAvailabilityFlagger:
    """Flag open data / data-availability cues in paper text offline.

    Offline heuristic open-data scan for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace/PaperQA
    open-data-availability gap. Distinct from
    :class:`~retrieval.code_availability.CodeAvailabilityBooster` (code
    boosting), :class:`~retrieval.funding_disclosure.FundingDisclosureFlagger`,
    and
    :class:`~retrieval.preregistration_flag.PreregistrationFlagDetector`.
    Never mutates inputs and never calls the network.
    """

    def flag(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[OpenDataAvailabilityFlag, ...]:
        """Return open-data availability flags for ``papers``.

        Reads ``abstract``, ``title``, ``summary``, data-availability /
        supplementary fields, and acknowledgement string fields. Empty input
        yields an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[OpenDataAvailabilityFlag] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            reasons: list[str] = []
            cues: list[str] = []
            if text:
                for label, pattern in _OPEN_DATA_PATTERNS:
                    match = pattern.search(text)
                    if match:
                        reasons.append(f"Matched open-data cue: {label}")
                        cues.append(match.group(0).strip())
            results.append(
                OpenDataAvailabilityFlag(
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
