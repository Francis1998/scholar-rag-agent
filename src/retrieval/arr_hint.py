"""Offline absolute risk reduction (ARR) hint extraction from paper text.

Detects advisory ARR / absolute risk reduction / risk difference numeric
hints from title/abstract/results via deterministic patterns. Never calls
the network. Distinct from
:class:`~retrieval.nnt_hint.NumberNeededToTreatHintExtractor`,
:class:`~retrieval.effect_size_hint.EffectSizeHintExtractor`, and
:class:`~retrieval.p_value_hint.PValueHintExtractor`.
Fills an Elicit / Consensus / SciSpace / PaperQA ARR surfacing gap.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TEXT_FIELDS = ("abstract", "title", "summary", "methods", "method", "results")

_PATTERN = re.compile(
    r"(?:"
    r"\bARR\b|\babsolute\s+risk\s+reduction\b|\brisk\s+difference\b|\bARD\b"
    r")"
    r"\s*(?:=|:)?\s*"
    r"(-?\d+(?:\.\d+)?)\s*(%|percent|percentage\s+points?|pp)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AbsoluteRiskReductionHint:
    """Advisory ARR numeric hint for one paper row."""

    paper_id: str
    value: float | None
    unit: str | None
    matched: str | None
    flagged: bool


class AbsoluteRiskReductionHintExtractor:
    """Extract ARR / absolute risk reduction hints offline."""

    def extract(
        self,
        papers: Sequence[dict[str, object]],
    ) -> tuple[AbsoluteRiskReductionHint, ...]:
        """Return ARR hints for ``papers``."""
        if not papers:
            return ()
        results: list[AbsoluteRiskReductionHint] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            text = self._collect_text(paper)
            match = _PATTERN.search(text) if text else None
            if match:
                value = float(match.group(1))
                unit = (match.group(2) or "").strip() or None
                results.append(
                    AbsoluteRiskReductionHint(
                        paper_id=paper_id,
                        value=value,
                        unit=unit,
                        matched=match.group(0).strip(),
                        flagged=True,
                    )
                )
            else:
                results.append(
                    AbsoluteRiskReductionHint(
                        paper_id=paper_id,
                        value=None,
                        unit=None,
                        matched=None,
                        flagged=False,
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
