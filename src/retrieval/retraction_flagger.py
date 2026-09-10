"""Offline advisory retraction flags from a caller-supplied flag set.

Maps paper identifiers (DOI / paper_id / id) against an in-memory flag set of
retracted / withdrawn / expression-of-concern statuses and returns advisory
flags without dropping rows or calling the network. Distinct from
:class:`~retrieval.retracted_filter.RetractedFilter` (which filters or demotes
``SearchResult`` hits via per-row metadata). Fills a Semantic Scholar /
OpenAlex retraction-signal gap with an offline stub (no LLM call). Optional
later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_STATUS_ALIASES = {
    "retracted": "retracted",
    "retraction": "retracted",
    "withdrawn": "withdrawn",
    "withdrawal": "withdrawn",
    "expression_of_concern": "expression_of_concern",
    "expression-of-concern": "expression_of_concern",
    "eoc": "expression_of_concern",
    "concern": "expression_of_concern",
}


@dataclass(frozen=True)
class RetractionFlag:
    """Advisory retraction/withdrawal flag for one paper row."""

    paper_id: str
    status: str
    advisory: str
    matched: bool


class RetractionWatchFlagger:
    """Flag papers against a caller-supplied offline retraction set.

    Offline advisory flagger for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines — Semantic Scholar/OpenAlex retraction-signal gap.
    Distinct from :class:`~retrieval.retracted_filter.RetractedFilter` and from
    live OpenAlex retraction connectors. Never mutates inputs and never drops
    papers; unmatched rows return ``matched=False`` with ``status=\"clear\"``.
    """

    def __init__(self, flag_set: Mapping[str, str] | None = None) -> None:
        """Create a retraction flagger.

        Args:
            flag_set: Mapping of normalized paper identifiers (DOI, paper_id,
                or id) to status strings such as ``retracted``, ``withdrawn``,
                or ``expression_of_concern``. Keys are matched case-insensitively
                after stripping whitespace and a leading ``https://doi.org/``
                prefix. ``None`` or empty means every paper is clear.
        """
        normalized: dict[str, str] = {}
        for raw_key, raw_status in (flag_set or {}).items():
            key = self._normalize_id(str(raw_key))
            if not key:
                continue
            status = self._normalize_status(str(raw_status))
            normalized[key] = status
        self._flag_set = normalized

    def flag(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[RetractionFlag, ...]:
        """Return advisory flags for ``papers`` using the configured flag set.

        Looks up ``doi``, then ``paper_id``, then ``id`` on each dict. Empty
        input yields an empty tuple. Inputs are not mutated. Matched statuses
        produce advisory text suitable for UI warnings; unmatched papers are
        labelled ``status=\"clear\"`` with ``matched=False``.
        """
        if not papers:
            return ()

        flags: list[RetractionFlag] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            lookup = self._normalize_id(paper_id)
            status = self._flag_set.get(lookup)
            if status is None:
                flags.append(
                    RetractionFlag(
                        paper_id=paper_id,
                        status="clear",
                        advisory="No retraction/withdrawal flag in local set.",
                        matched=False,
                    )
                )
                continue
            advisory = {
                "retracted": "Advisory: paper marked retracted in local flag set.",
                "withdrawn": "Advisory: paper marked withdrawn in local flag set.",
                "expression_of_concern": (
                    "Advisory: expression of concern recorded in local flag set."
                ),
            }.get(status, f"Advisory: flagged as {status} in local flag set.")
            flags.append(
                RetractionFlag(
                    paper_id=paper_id,
                    status=status,
                    advisory=advisory,
                    matched=True,
                )
            )
        return tuple(flags)

    @staticmethod
    def _normalize_id(value: str) -> str:
        text = value.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        return text.strip()

    @staticmethod
    def _normalize_status(value: str) -> str:
        key = value.strip().lower().replace(" ", "_")
        return _STATUS_ALIASES.get(key, key or "unknown")

    @staticmethod
    def _resolve_id(paper: dict[str, object], index: int) -> str:
        for key in ("doi", "paper_id", "id"):
            raw = paper.get(key)
            if raw is None or raw == "":
                continue
            text = str(raw).strip()
            if text:
                return text
        title = str(paper.get("title", "") or "").strip()
        return title or f"paper-{index}"
