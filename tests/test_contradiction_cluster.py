"""Tests for ContradictionClusterFinder."""

import pytest

from retrieval.contradiction_cluster import ContradictionClusterFinder


def test_rejects_invalid_overlap_knobs() -> None:
    with pytest.raises(ValueError, match="support_overlap"):
        ContradictionClusterFinder(support_overlap=1.5)
    with pytest.raises(ValueError, match="contradict_overlap"):
        ContradictionClusterFinder(contradict_overlap=-0.1)
    with pytest.raises(ValueError, match="contradict_overlap"):
        ContradictionClusterFinder(support_overlap=0.2, contradict_overlap=0.5)


def test_empty_claim_raises() -> None:
    with pytest.raises(ValueError, match="claim"):
        ContradictionClusterFinder().find("   ", ["some evidence"])


def test_empty_passages_are_zero_tension() -> None:
    result = ContradictionClusterFinder().find(
        "Graph retrieval improves multi-hop reasoning.",
        [],
    )
    assert result.supporting_ids == ()
    assert result.contradicting_ids == ()
    assert result.tension_score == 0.0
    assert result.labels == ()


def test_clusters_support_vs_negation_and_scores_tension() -> None:
    claim = "Graph retrieval improves multi-hop reasoning over scientific papers."
    passages = [
        {
            "evidence_id": "support",
            "text": (
                "Graph retrieval improves multi-hop reasoning over scientific "
                "papers in literature corpora."
            ),
        },
        {
            "id": "negate",
            "passage": (
                "Graph retrieval does not improve multi-hop reasoning over scientific papers."
            ),
        },
        {
            "chunk_id": "fail",
            "abstract": (
                "Graph retrieval fails to improve multi-hop reasoning across "
                "scientific paper corpora."
            ),
        },
        "Unrelated laser cavity modes under optical cooling.",
    ]
    result = ContradictionClusterFinder().find(claim, passages)
    assert "support" in result.supporting_ids
    assert "negate" in result.contradicting_ids
    assert "fail" in result.contradicting_ids
    assert result.tension_score > 0.0
    assert 0.0 <= result.tension_score <= 1.0
    label_map = dict(result.labels)
    assert label_map["support"] == "supporting"
    assert label_map["negate"] == "contradicting"
    assert label_map["fail"] == "contradicting"
    assert any(label == "neutral" for _pid, label in result.labels)


def test_antonym_cues_mark_contradicting() -> None:
    claim = "Attention improves retrieval quality for long documents."
    passages = [
        {
            "evidence_id": "pro",
            "text": "Attention improves retrieval quality for long documents.",
        },
        {
            "evidence_id": "con",
            "text": "Attention decreases retrieval quality for long documents.",
        },
    ]
    result = ContradictionClusterFinder().find(claim, passages)
    assert result.supporting_ids == ("pro",)
    assert result.contradicting_ids == ("con",)
    assert result.tension_score > 0.0


def test_string_passages_get_synthetic_ids() -> None:
    claim = "Sparse attention improves transformer scaling on long sequences."
    result = ContradictionClusterFinder().find(
        claim,
        [
            "Sparse attention improves transformer scaling on long sequences.",
            "Sparse attention never improves transformer scaling on long sequences.",
        ],
    )
    assert result.supporting_ids[0].startswith("passage-")
    assert result.contradicting_ids[0].startswith("passage-")
    assert len(result.labels) == 2


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = ContradictionClusterFinder.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "Elicit" in doc
    assert "ClaimSupportScorer" in doc
    assert "ClaimVerificationGate" in doc
