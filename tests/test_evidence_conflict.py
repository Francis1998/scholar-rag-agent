"""Tests for EvidenceConflictDetector."""

import pytest

from retrieval.evidence_conflict import EvidenceConflictDetector


def test_rejects_invalid_min_shared_terms() -> None:
    with pytest.raises(ValueError, match="min_shared_terms"):
        EvidenceConflictDetector(min_shared_terms=-1)
    with pytest.raises(ValueError, match="min_shared_terms"):
        EvidenceConflictDetector(min_shared_terms=True)  # type: ignore[arg-type]


def test_opposing_snippets_conflict() -> None:
    snippets = [
        "Treatment A increases survival in patients with sepsis.",
        "Treatment A decreases survival in patients with sepsis.",
    ]
    conflicts = EvidenceConflictDetector(min_shared_terms=1).detect(snippets)
    assert len(conflicts) >= 1
    assert conflicts[0].left_index == 0
    assert conflicts[0].right_index == 1
    assert "opposing" in conflicts[0].reason or "negation" in conflicts[0].reason


def test_agreeing_snippets_do_not_conflict() -> None:
    snippets = [
        "Treatment A increases survival in patients with sepsis.",
        "Treatment A also increases survival for septic patients.",
    ]
    conflicts = EvidenceConflictDetector(min_shared_terms=1).detect(snippets)
    assert conflicts == ()


def test_empty_snippets_ok() -> None:
    detector = EvidenceConflictDetector()
    assert detector.detect([]) == ()
    assert detector.detect(["only one snippet about increased survival"]) == ()


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = EvidenceConflictDetector.__doc__ or ""
    assert "GPT-6 Astra" in doc
    assert "Claude Opus 5.5" in doc
    assert "Gemini 4 Argon" in doc
    assert "Kimi K3.1 Pro" in doc
    assert "Consensus" in doc or "Elicit" in doc
    assert "SelfRagReflectionGate" in doc
