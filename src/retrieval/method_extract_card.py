"""Extract PICO/methods cards from paper abstracts.

Deterministic heuristic extraction of Population, Intervention, Comparison,
Outcome, and study-design cues from title/abstract text. Fills an Elicit /
Consensus / SciSpace methods-extraction gap with offline, reproducible
parsing (no LLM call). Optional later enrichment can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_POPULATION_PATTERNS = (
    re.compile(
        r"(?i)\b(?:in|among|of)\s+(?P<span>(?:patients?|participants?|subjects?|"
        r"adults?|children|infants?|cohorts?|individuals?)(?:\s+with\s+[^.|;]{3,80})?)"
    ),
    re.compile(r"(?i)\b(?P<span>(?:patients?|participants?|subjects?)\s+with\s+[^.|;]{3,80})"),
)
_INTERVENTION_PATTERNS = (
    re.compile(
        r"(?i)\b(?:treated with|receiving|received|administered|using|intervention(?:\s+of)?|"
        r"therapy with)\s+(?P<span>[^.|;]{3,80})"
    ),
    re.compile(r"(?i)\b(?P<span>(?:drug|vaccine|device|procedure)\s+[^.|;]{3,60})"),
)
_COMPARISON_PATTERNS = (
    re.compile(
        r"(?i)\b(?:compared\s+(?:with|to)|versus|vs\.?|relative to|control(?:\s+group)?|"
        r"placebo)\s*(?:of\s+)?(?P<span>[^.|;]{0,80})"
    ),
)
_OUTCOME_PATTERNS = (
    re.compile(
        r"(?i)\b(?:primary outcome|outcome(?:s)?|resulted in|associated with|"
        r"improved|reduced|increased|decreased|mortality|survival|efficacy)\s*"
        r"(?P<span>[^.|;]{0,80})"
    ),
)
_DESIGN_CUES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(?:randomized\s+controlled\s+trial|randomised\s+controlled\s+trial|\brct\b)"
        ),
        "rct",
    ),
    (re.compile(r"(?i)\bmeta[- ]analysis\b"), "meta-analysis"),
    (re.compile(r"(?i)\bsystematic\s+review\b"), "systematic-review"),
    (re.compile(r"(?i)\bcase[- ]control\b"), "case-control"),
    (re.compile(r"(?i)\bcross[- ]sectional\b"), "cross-sectional"),
    (re.compile(r"(?i)\bprospective\s+cohort\b"), "prospective-cohort"),
    (re.compile(r"(?i)\bretrospective\s+cohort\b"), "retrospective-cohort"),
    (re.compile(r"(?i)\bcohort\s+study\b"), "cohort"),
    (re.compile(r"(?i)\brandomized\b|\brandomised\b"), "randomized"),
    (re.compile(r"(?i)\bin\s+vitro\b"), "in-vitro"),
    (re.compile(r"(?i)\bin\s+vivo\b"), "in-vivo"),
)


@dataclass(frozen=True)
class MethodCard:
    """Normalized PICO/methods card extracted from an abstract."""

    population: str
    intervention: str
    comparison: str
    outcome: str
    study_design: str
    confidence: float


class MethodExtractCard:
    """Extract a PICO/methods card from abstract (and optional title) text.

    Offline heuristic extractor for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit/Consensus/SciSpace methods
    extraction gap. Deterministic: identical inputs always yield the same card.
    """

    def __init__(self, min_span_chars: int = 3) -> None:
        """Create a methods-card extractor.

        Args:
            min_span_chars: Minimum non-blank span length retained for a PICO
                field (``>= 1``). Shorter matches are discarded.

        Raises:
            ValueError: If ``min_span_chars`` is not an int ``>= 1``.
        """
        if not isinstance(min_span_chars, int) or isinstance(min_span_chars, bool):
            raise ValueError("min_span_chars must be an int >= 1")
        if min_span_chars < 1:
            raise ValueError("min_span_chars must be an int >= 1")
        self._min_span_chars = min_span_chars

    def extract(self, abstract: str, *, title: str = "") -> MethodCard:
        """Return a :class:`MethodCard` for ``abstract`` (and optional ``title``).

        Empty or blank input yields empty string fields and ``confidence=0.0``.
        Inputs are not mutated. Confidence is the fraction of the five card
        fields that are non-empty (``[0.0, 1.0]``).
        """
        blob = _WHITESPACE.sub(" ", f"{title} {abstract}".strip())
        if not blob:
            return MethodCard(
                population="",
                intervention="",
                comparison="",
                outcome="",
                study_design="",
                confidence=0.0,
            )

        population = self._first_span(blob, _POPULATION_PATTERNS)
        intervention = self._first_span(blob, _INTERVENTION_PATTERNS)
        comparison = self._first_span(blob, _COMPARISON_PATTERNS)
        outcome = self._first_span(blob, _OUTCOME_PATTERNS)
        study_design = self._study_design(blob)

        filled = sum(
            1 for value in (population, intervention, comparison, outcome, study_design) if value
        )
        confidence = filled / 5.0
        if not math.isfinite(confidence):
            confidence = 0.0
        return MethodCard(
            population=population,
            intervention=intervention,
            comparison=comparison,
            outcome=outcome,
            study_design=study_design,
            confidence=confidence,
        )

    def _first_span(self, text: str, patterns: tuple[re.Pattern[str], ...]) -> str:
        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            span = _WHITESPACE.sub(" ", match.group("span").strip(" .,;:"))
            if len(span) >= self._min_span_chars:
                return span[:120]
        return ""

    @staticmethod
    def _study_design(text: str) -> str:
        for pattern, label in _DESIGN_CUES:
            if pattern.search(text):
                return label
        return ""
