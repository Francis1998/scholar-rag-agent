"""Offline PrevalenceIncidenceBiasCueExtractor extraction.

Detects advisory cues from title/abstract/methods via deterministic patterns.
Never calls the network.
Distinct from `SelectionBiasCueExtractor` / `ImmortalTimeBiasCueExtractor`.
Fills an Elicit/Consensus/Semantic Scholar prevalence-incidence bias cue extractors gap.
Optional later narrative can use frontier LLMs.
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
    "discussion",
)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "prevalence_incidence_bias",
        re.compile(
            r"\bprevalence.?incidence bias\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prevalent_user_bias",
        re.compile(
            r"\bprevalent user bias\b",
            re.IGNORECASE,
        ),
    ),
    (
        "incidence_density_bias",
        re.compile(
            r"\bincidence density bias\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class PrevalenceIncidenceBiasCue:
    """Advisory cues for one paper row."""

    paper_id: str
    cue_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class PrevalenceIncidenceBiasCueExtractor:
    """Extract offline advisory cues."""

    def extract(
        self, papers: Sequence[dict[str, object]]
    ) -> tuple[PrevalenceIncidenceBiasCue, ...]:
        """Return cues for ``papers``."""

        if not papers:
            return ()
        results: list[PrevalenceIncidenceBiasCue] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            cues: list[str] = []
            matched: list[str] = []
            kind: str | None = None
            if text:
                for label, pattern in _PATTERNS:
                    match = pattern.search(text)
                    if not match:
                        continue
                    cues.append(label)
                    matched.append(match.group(0))
                    if kind is None:
                        kind = label
            results.append(
                PrevalenceIncidenceBiasCue(
                    paper_id=paper_id,
                    cue_kind=kind,
                    cues=tuple(cues),
                    matched=tuple(matched),
                    flagged=bool(cues),
                )
            )
        return tuple(results)

    @staticmethod
    def _resolve_id(paper: dict[str, object], index: int) -> str:
        """Resolve stable paper id."""

        for key in ("paper_id", "id", "doi"):
            value = paper.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"paper-{index}"

    @staticmethod
    def _collect_text(paper: dict[str, object]) -> str:
        """Join searchable text fields."""

        chunks: list[str] = []
        for key in _TEXT_FIELDS:
            value = paper.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value)
        return "\n".join(chunks)
