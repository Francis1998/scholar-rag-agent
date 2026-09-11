"""Tests for ClaimSupportScorer."""

import pytest

from retrieval.claim_support import ClaimSupportScorer


def test_rejects_invalid_weights_and_thresholds() -> None:
    with pytest.raises(ValueError, match="coverage_weight"):
        ClaimSupportScorer(coverage_weight=1.5)
    with pytest.raises(ValueError, match="supported_threshold"):
        ClaimSupportScorer(supported_threshold=-0.1)
    with pytest.raises(ValueError, match="partial_threshold"):
        ClaimSupportScorer(partial_threshold=0.9, supported_threshold=0.5)


def test_empty_claim_raises() -> None:
    with pytest.raises(ValueError, match="claim"):
        ClaimSupportScorer().score("   ", ["some evidence"])
    with pytest.raises(ValueError, match="claim"):
        ClaimSupportScorer().score_one("", "evidence text")


def test_empty_evidence_is_unsupported() -> None:
    results = ClaimSupportScorer().score("Graph retrieval improves multi-hop reasoning.", [])
    assert len(results) == 1
    assert results[0].support_score == 0.0
    assert results[0].label == "unsupported"
    assert results[0].matched_terms == ()
    assert results[0].evidence_id is None


def test_ranks_by_support_strength_and_labels() -> None:
    claim = "Graph retrieval improves multi-hop reasoning over scientific papers."
    evidence = [
        {
            "evidence_id": "weak",
            "text": "Laser cavity modes are stable under optical cooling.",
        },
        {
            "id": "strong",
            "passage": (
                "Graph retrieval improves multi-hop reasoning over scientific "
                "papers in literature corpora."
            ),
        },
        "Graph retrieval helps multi-hop literature search.",
    ]
    ranked = ClaimSupportScorer().score(claim, evidence)
    assert len(ranked) == 3
    assert ranked[0].support_score >= ranked[1].support_score >= ranked[2].support_score
    assert ranked[0].evidence_id == "strong"
    assert ranked[0].label == "supported"
    assert "graph" in ranked[0].matched_terms
    assert "retrieval" in ranked[0].matched_terms
    assert any(row.label in {"partial", "supported"} for row in ranked[1:])
    assert ranked[-1].label == "unsupported"
    assert ranked[-1].evidence_id == "weak"


def test_score_one_accepts_string_and_mapping() -> None:
    claim = "Transformer attention scales quadratically with sequence length."
    strong = ClaimSupportScorer().score_one(
        claim,
        {
            "chunk_id": "c1",
            "abstract": "Transformer attention scales quadratically with sequence length.",
        },
    )
    weak = ClaimSupportScorer().score_one(claim, "Unrelated astronomy abstract.")
    assert strong.evidence_id == "c1"
    assert strong.support_score > weak.support_score
    assert strong.label == "supported"
    assert weak.label == "unsupported"
    assert weak.evidence_id is None


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = ClaimSupportScorer.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "Elicit" in doc or "Semantic Scholar" in doc
    assert "ClaimVerificationGate" in doc
    assert "CitationGroundednessScorer" in doc
