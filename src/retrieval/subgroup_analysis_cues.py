"""Offline subgroup analysis cue extraction.

Detects advisory subgroup analysis cues from title/abstract/methods via
deterministic patterns. Never calls the network. Distinct from
:class:`~retrieval.primary_endpoint_cues.PrimaryEndpointCueExtractor` and
:class:`~retrieval.heterogeneity_i2_hint.HeterogeneityI2HintExtractor`.
Fills an Elicit / Consensus / Cochrane subgroup-reporting gap.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "methods", "method", "results")

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "prespecified_subgroup",
        re.compile(r"\bpre[- ]?specified subgroup\b|\bprespecified subgroup\b", re.IGNORECASE),
    ),
    (
        "post_hoc_subgroup",
        re.compile(r"\bpost[- ]?hoc subgroup\b|\bexploratory subgroup\b", re.IGNORECASE),
    ),
    (
        "subgroup_analysis",
        re.compile(
            r"\bsubgroup analys(?:is|es)\b|\binteraction (?:test|p[- ]?value)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class SubgroupAnalysisCue:
    """Advisory subgroup analysis cues for one paper row."""

    paper_id: str
    subgroup_kind: str | None
    cues: tuple[str, ...]
    matched: tuple[str, ...]
    flagged: bool


class SubgroupAnalysisCueExtractor:
    """Extract subgroup analysis cues offline."""

    def extract(self, papers: Sequence[dict[str, object]]) -> tuple[SubgroupAnalysisCue, ...]:
        """Return subgroup cues for ``papers``."""

        if not papers:
            return ()
        results: list[SubgroupAnalysisCue] = []
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
                    matched.append(match.group(0).strip())
                    if kind is None:
                        kind = label
            results.append(
                SubgroupAnalysisCue(
                    paper_id=paper_id,
                    subgroup_kind=kind,
                    cues=tuple(cues),
                    matched=tuple(matched),
                    flagged=bool(cues),
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
                parts.append(value.strip())
        return "\n".join(parts)
