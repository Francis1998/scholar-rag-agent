"""Offline heuristic funding-disclosure flags from abstract/acknowledgements.

Scans local text fields for funder cues (NIH, NSF, ERC, ``funded by``, grant
numbers) and returns advisory flags with reasons. Never calls the network.
Distinct from :mod:`ingestion.crossref_funder` (Crossref Funder Registry
connector). Fills a Scite / Elicit / Consensus funding-disclosure surfacing
gap without an LLM call. Optional later narrative can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = (
    "abstract",
    "acknowledgements",
    "acknowledgment",
    "acknowledgement",
    "funding",
    "funding_text",
    "sponsor",
)

_FUNDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("NIH", re.compile(r"\bNIH\b|\bNational Institutes of Health\b", re.IGNORECASE)),
    ("NSF", re.compile(r"\bNSF\b|\bNational Science Foundation\b", re.IGNORECASE)),
    ("ERC", re.compile(r"\bERC\b|\bEuropean Research Council\b", re.IGNORECASE)),
    (
        "funded by",
        re.compile(r"\bfunded by\b|\bsupported by\b|\bsponsored by\b", re.IGNORECASE),
    ),
)

# Common US/EU-style grant tokens (R01..., U01..., NSF-..., ERC-..., grant no.).
_GRANT_NUMBER = re.compile(
    r"\b(?:grant(?:\s+number)?|award)\s*[:=]?\s*[A-Z0-9][-A-Z0-9]{3,}\b"
    r"|\b(?:R\d{2}|U\d{2}|P\d{2}|K\d{2}|F\d{2}|T\d{2})-?[A-Z]{0,4}\d{5,}\b"
    r"|\bNSF[- ][A-Z]{2,5}[- ]?\d{4,}\b"
    r"|\bERC[- ]?(?:StG|CoG|AdG|SyG)?[- ]?\d{4,}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FundingDisclosureFlag:
    """Advisory funding-disclosure flag for one paper row."""

    paper_id: str
    flagged: bool
    reasons: tuple[str, ...]
    matched_cues: tuple[str, ...]


class FundingDisclosureFlagger:
    """Flag funding cues in abstract/acknowledgements text offline.

    Offline heuristic funding-disclosure scan for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Scite/Elicit/Consensus funding-cue gap.
    Distinct from Crossref Funder Registry ingestion
    (:mod:`ingestion.crossref_funder`). Never mutates inputs and never calls
    the network.
    """

    def flag(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[FundingDisclosureFlag, ...]:
        """Return funding-disclosure flags for ``papers``.

        Reads ``abstract``, ``acknowledgements`` / ``acknowledgment``,
        ``funding``, ``funding_text``, and ``sponsor`` string fields. Empty
        input yields an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[FundingDisclosureFlag] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            reasons: list[str] = []
            cues: list[str] = []
            if text:
                for label, pattern in _FUNDER_PATTERNS:
                    if pattern.search(text):
                        reasons.append(f"Matched funder cue: {label}")
                        cues.append(label)
                if _GRANT_NUMBER.search(text):
                    reasons.append("Matched grant-number pattern")
                    cues.append("grant_number")
            results.append(
                FundingDisclosureFlag(
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
