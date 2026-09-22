"""Tests for SubgroupAnalysisCueExtractor."""

from retrieval.subgroup_analysis_cues import SubgroupAnalysisCueExtractor


def test_empty_ok() -> None:
    assert SubgroupAnalysisCueExtractor().extract([]) == ()


def test_prespecified() -> None:
    rows = SubgroupAnalysisCueExtractor().extract(
        [{"paper_id": "p1", "methods": "A pre-specified subgroup was age >= 65."}]
    )
    assert rows[0].flagged is True
    assert rows[0].subgroup_kind == "prespecified_subgroup"


def test_post_hoc() -> None:
    rows = SubgroupAnalysisCueExtractor().extract(
        [{"id": "p2", "results": "Post-hoc subgroup findings were exploratory."}]
    )
    assert rows[0].flagged is True
    assert "post_hoc_subgroup" in rows[0].cues


def test_clear() -> None:
    rows = SubgroupAnalysisCueExtractor().extract(
        [{"paper_id": "clear", "abstract": "Primary endpoint was mortality."}]
    )
    assert rows[0].flagged is False
