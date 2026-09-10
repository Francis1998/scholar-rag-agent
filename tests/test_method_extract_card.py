"""Tests for MethodExtractCard."""

import pytest

from retrieval.method_extract_card import MethodExtractCard


def test_rejects_invalid_min_span_chars() -> None:
    with pytest.raises(ValueError, match="min_span_chars"):
        MethodExtractCard(min_span_chars=0)
    with pytest.raises(ValueError, match="min_span_chars"):
        MethodExtractCard(min_span_chars=True)  # type: ignore[arg-type]


def test_empty_abstract_yields_empty_card() -> None:
    card = MethodExtractCard().extract("")
    assert card.population == ""
    assert card.intervention == ""
    assert card.comparison == ""
    assert card.outcome == ""
    assert card.study_design == ""
    assert card.confidence == 0.0


def test_extracts_pico_and_design_from_clinical_abstract() -> None:
    abstract = (
        "In a randomized controlled trial, patients with sepsis received "
        "hydrocortisone compared to placebo. The primary outcome was 28-day "
        "mortality."
    )
    card = MethodExtractCard().extract(abstract, title="Steroids in sepsis")
    assert "patients with sepsis" in card.population.lower()
    assert "hydrocortisone" in card.intervention.lower()
    assert card.comparison  # placebo / compared to
    assert card.outcome  # mortality / primary outcome
    assert card.study_design == "rct"
    assert card.confidence >= 0.8


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = MethodExtractCard.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "Elicit" in doc or "Consensus" in doc or "SciSpace" in doc
