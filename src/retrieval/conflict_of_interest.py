"""Offline heuristic conflict-of-interest / disclosure flags from paper text.

Scans local text fields for COI cues (``conflict of interest``, competing
interests, financial disclosures, advisory/consultant language) and returns
advisory flags with reasons. Never calls the network. Distinct from
:mod:`retrieval.funding_disclosure` (funder/grant cues). Fills a Scite /
Elicit / Consensus conflict-of-interest surfacing gap without an LLM call.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.
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
    "disclosure",
    "disclosures",
    "conflicts",
    "conflict_of_interest",
    "competing_interests",
    "coi",
)

_COI_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "conflict of interest",
        re.compile(r"\bconflicts?\s+of\s+interest\b", re.IGNORECASE),
    ),
    (
        "competing interest",
        re.compile(r"\bcompeting\s+interests?\b", re.IGNORECASE),
    ),
    (
        "financial disclosure",
        re.compile(
            r"\bfinancial\s+disclosures?\b"
            r"|\bdisclosures?\s*:"
            r"|\bauthors?\s+disclose\b",
            re.IGNORECASE,
        ),
    ),
    (
        "no conflicts declaration",
        re.compile(
            r"\bno\s+conflicts?\s+of\s+interest\b|\bdeclare\s+no\s+conflicts?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "advisory / consultant",
        re.compile(
            r"\badvisory\s+board\b|\bconsultant\s+for\b|\bconsulting\s+fees?\b"
            r"|\bhonoraria\b|\bstock\s+options?\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class ConflictOfInterestFlag:
    """Advisory conflict-of-interest flag for one paper row."""

    paper_id: str
    flagged: bool
    reasons: tuple[str, ...]
    matched_cues: tuple[str, ...]


class ConflictOfInterestFlagger:
    """Flag conflict-of-interest / disclosure cues in paper text offline.

    Offline heuristic COI-disclosure scan for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Scite/Elicit/Consensus COI-cue gap.
    Distinct from :class:`~retrieval.funding_disclosure.FundingDisclosureFlagger`
    (funder/grant cues). Never mutates inputs and never calls the network.
    """

    def flag(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[ConflictOfInterestFlag, ...]:
        """Return conflict-of-interest flags for ``papers``.

        Reads ``abstract``, acknowledgements variants, ``disclosure`` /
        ``disclosures``, ``conflicts``, ``conflict_of_interest``,
        ``competing_interests``, and ``coi`` string fields. Empty input yields
        an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        results: list[ConflictOfInterestFlag] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            reasons: list[str] = []
            cues: list[str] = []
            if text:
                for label, pattern in _COI_PATTERNS:
                    if pattern.search(text):
                        reasons.append(f"Matched COI cue: {label}")
                        cues.append(label)
            results.append(
                ConflictOfInterestFlag(
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
