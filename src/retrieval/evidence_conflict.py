"""Detect conflicting claims across evidence snippets.

Scans pairs of plain-text snippets for negation cues and opposite polarity
keywords (increase vs decrease, effective vs ineffective, support vs refute).
Distinct from
:class:`~retrieval.self_rag_reflection_gate.SelfRagReflectionGate` (which
emits SUPPORT/PARTIAL/REFUSE decisions over ``SearchResult`` hits) — this
module returns an inspectable conflict list with left/right snippet indices.
Fills a Consensus / Elicit conflicting-evidence view gap with offline,
deterministic heuristics (no LLM call). Optional later adjudication can use
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from retrieval.sparse import meaningful_terms, tokenize

_NEGATIONS = frozenset({"no", "not", "never", "neither", "without", "nor"})
_CONFLICT_AXES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "direction": (
        frozenset(
            {
                "better",
                "higher",
                "improve",
                "improved",
                "improves",
                "increase",
                "increased",
                "increases",
                "positive",
            }
        ),
        frozenset(
            {
                "decrease",
                "decreased",
                "decreases",
                "lower",
                "negative",
                "worse",
                "worsen",
                "worsened",
                "worsens",
            }
        ),
    ),
    "efficacy": (
        frozenset({"beneficial", "benefit", "benefits", "effective", "works"}),
        frozenset({"harmful", "harms", "ineffective", "fails"}),
    ),
    "support": (
        frozenset(
            {
                "confirm",
                "confirmed",
                "confirms",
                "consistent",
                "support",
                "supported",
                "supports",
            }
        ),
        frozenset(
            {
                "contradict",
                "contradicted",
                "contradicts",
                "inconsistent",
                "refute",
                "refuted",
                "refutes",
                "unsupported",
            }
        ),
    ),
}


@dataclass(frozen=True)
class EvidenceConflict:
    """One conflicting snippet pair with an inspectable reason."""

    left_index: int
    right_index: int
    reason: str


class EvidenceConflictDetector:
    """Detect opposing claims across evidence snippets.

    Pairwise lexical polarity scan for Consensus/Elicit-style conflicting
    evidence views. Offline detector for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Consensus/Elicit conflicting-evidence gap.
    Distinct from :class:`~retrieval.self_rag_reflection_gate.SelfRagReflectionGate`.
    """

    def __init__(self, min_shared_terms: int = 1) -> None:
        """Create an evidence conflict detector.

        Args:
            min_shared_terms: Minimum shared meaningful terms required before
                a polarity clash is reported (``>= 0``). ``0`` reports clashes
                even without topical overlap.

        Raises:
            ValueError: If ``min_shared_terms`` is not an int ``>= 0``.
        """
        if not isinstance(min_shared_terms, int) or isinstance(min_shared_terms, bool):
            raise ValueError("min_shared_terms must be an int >= 0")
        if min_shared_terms < 0:
            raise ValueError("min_shared_terms must be an int >= 0")
        self._min_shared_terms = min_shared_terms

    def detect(self, snippets: Sequence[str]) -> tuple[EvidenceConflict, ...]:
        """Return conflicts among ``snippets`` with left/right indices.

        Empty input yields an empty tuple. Agreeing snippets (same polarity
        axis side, or no opposing cues) produce no conflicts. Inputs are not
        mutated.
        """
        texts = [str(snippet) for snippet in snippets]
        if len(texts) < 2:
            return ()

        term_sets = [meaningful_terms(text) for text in texts]
        polarities = [self._polarities(text) for text in texts]
        conflicts: list[EvidenceConflict] = []

        for left, right in combinations(range(len(texts)), 2):
            shared = term_sets[left] & term_sets[right]
            if len(shared) < self._min_shared_terms:
                continue
            left_pol = polarities[left]
            right_pol = polarities[right]
            for axis in _CONFLICT_AXES:
                left_sides = left_pol.get(axis, frozenset())
                right_sides = right_pol.get(axis, frozenset())
                if not left_sides or not right_sides:
                    continue
                if any(
                    left_side != right_side
                    for left_side in left_sides
                    for right_side in right_sides
                ):
                    conflicts.append(
                        EvidenceConflict(
                            left_index=left,
                            right_index=right,
                            reason=(
                                f"opposing {axis} polarity between snippets "
                                f"{left} and {right}"
                            ),
                        )
                    )
                    break
            else:
                # Negation of a shared content claim: "X increases" vs "X does not increase".
                if self._negation_clash(texts[left], texts[right], shared):
                    conflicts.append(
                        EvidenceConflict(
                            left_index=left,
                            right_index=right,
                            reason=(
                                f"negation clash on shared terms between snippets "
                                f"{left} and {right}"
                            ),
                        )
                    )
        return tuple(conflicts)

    @staticmethod
    def _polarities(text: str) -> dict[str, frozenset[bool]]:
        tokens = tokenize(text)
        polarities: dict[str, set[bool]] = {}
        for index, token in enumerate(tokens):
            negated = any(previous in _NEGATIONS for previous in tokens[max(0, index - 2) : index])
            for axis, (positive_cues, negative_cues) in _CONFLICT_AXES.items():
                if token in positive_cues:
                    polarities.setdefault(axis, set()).add(not negated)
                elif token in negative_cues and not negated:
                    polarities.setdefault(axis, set()).add(False)
        return {axis: frozenset(sides) for axis, sides in polarities.items()}

    @staticmethod
    def _negation_clash(left: str, right: str, shared: set[str]) -> bool:
        """True when one snippet negates a polarity cue the other affirms."""
        if not shared:
            return False
        left_tokens = tokenize(left)
        right_tokens = tokenize(right)
        cue_terms = set().union(*(pos | neg for pos, neg in _CONFLICT_AXES.values()))
        for tokens_a, tokens_b in ((left_tokens, right_tokens), (right_tokens, left_tokens)):
            for index, token in enumerate(tokens_a):
                if token not in cue_terms:
                    continue
                a_negated = any(
                    previous in _NEGATIONS for previous in tokens_a[max(0, index - 2) : index]
                )
                if token not in tokens_b:
                    continue
                b_indices = [i for i, t in enumerate(tokens_b) if t == token]
                for b_index in b_indices:
                    b_negated = any(
                        previous in _NEGATIONS
                        for previous in tokens_b[max(0, b_index - 2) : b_index]
                    )
                    if a_negated != b_negated:
                        return True
        return False
