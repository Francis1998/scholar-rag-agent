"""Unit tests for ClusterRandomizationCueExtractor."""

from __future__ import annotations

from retrieval.cluster_randomization_cues import ClusterRandomizationCueExtractor


def test_flags_cluster_randomized() -> None:
    """Cluster-randomized phrase is flagged."""

    rows = ClusterRandomizationCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "A cluster-randomized trial of schools."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cue_kind == "cluster_randomized"


def test_flags_icc() -> None:
    """ICC cue is flagged."""

    rows = ClusterRandomizationCueExtractor().extract(
        [{"id": "p2", "methods": "We assumed an ICC = 0.02 for sample size."}]
    )
    assert rows[0].flagged is True
    assert "intracluster_correlation" in rows[0].cues


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert ClusterRandomizationCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = ClusterRandomizationCueExtractor().extract(
        [{"paper_id": "p3", "abstract": "An individually randomized trial."}]
    )
    assert rows[0].flagged is False
