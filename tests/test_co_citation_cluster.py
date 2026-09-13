"""Tests for CoCitationClusterFinder."""

import pytest

from retrieval.co_citation_cluster import CoCitationClusterFinder


def test_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="min_shared"):
        CoCitationClusterFinder(min_shared=0)
    with pytest.raises(ValueError, match="min_jaccard"):
        CoCitationClusterFinder(min_jaccard=1.5)
    with pytest.raises(ValueError, match="min_jaccard"):
        CoCitationClusterFinder(min_jaccard=-0.1)


def test_empty_papers_ok() -> None:
    assert CoCitationClusterFinder().find([]) == ()


def test_clusters_papers_sharing_neighbors_above_threshold() -> None:
    papers = [
        {
            "paper_id": "A",
            "neighbors": {"n1", "n2", "n3", "n9"},
        },
        {
            "id": "B",
            "citation_neighbors": ["n1", "n2", "n3", "n8"],
        },
        {
            "paper_id": "C",
            "neighbors": {"n1", "n2", "n7"},
        },
        {
            "paper_id": "D",
            "neighbors": {"x1", "x2", "x3"},
        },
    ]
    clusters = CoCitationClusterFinder(min_shared=2, min_jaccard=0.0).find(papers)
    assert clusters
    member_sets = [set(cluster.paper_ids) for cluster in clusters]
    assert any({"A", "B", "C"} <= members or {"A", "B"} <= members for members in member_sets)
    # Isolated paper remains a singleton or is omitted from multi-paper clusters.
    multi = [c for c in clusters if len(c.paper_ids) >= 2]
    assert all("D" not in c.paper_ids for c in multi)
    assert all(c.shared_neighbor_count >= 2 for c in multi)


def test_jaccard_threshold_filters_weak_overlap() -> None:
    papers = [
        {"paper_id": "A", "neighbors": {"n1", "n2", "n3", "n4", "n5", "n6"}},
        {"paper_id": "B", "neighbors": {"n1", "n2", "z1", "z2", "z3", "z4"}},
    ]
    # Shared = 2, union = 10, jaccard = 0.2 — below 0.5.
    weak = CoCitationClusterFinder(min_shared=1, min_jaccard=0.5).find(papers)
    multi = [c for c in weak if len(c.paper_ids) >= 2]
    assert multi == []
    strong = CoCitationClusterFinder(min_shared=2, min_jaccard=0.15).find(papers)
    multi_strong = [c for c in strong if len(c.paper_ids) >= 2]
    assert any(set(c.paper_ids) == {"A", "B"} for c in multi_strong)


def test_never_mutates_input() -> None:
    neighbors = {"n1", "n2"}
    paper = {"paper_id": "A", "neighbors": neighbors}
    CoCitationClusterFinder(min_shared=1).find([paper, {"paper_id": "B", "neighbors": {"n1"}}])
    assert paper["neighbors"] is neighbors
    assert neighbors == {"n1", "n2"}


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = CoCitationClusterFinder.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "CitationGraph" in doc or "citation_graph" in doc
    assert "ContradictionCluster" in doc or "contradiction_cluster" in doc
