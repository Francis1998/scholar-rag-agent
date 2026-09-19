"""Tests for PublicationBiasCueExtractor."""

from retrieval.publication_bias_cues import PublicationBiasCueExtractor


def test_empty_papers_ok() -> None:
    assert PublicationBiasCueExtractor().extract([]) == ()


def test_extracts_publication_bias_cues() -> None:
    papers = [
        {"paper_id": "p1", "abstract": "Funnel plot asymmetry suggested bias."},
        {"id": "p2", "methods": "Egger's test was performed."},
        {"doi": "10.1/x", "discussion": "We assessed small-study effects."},
        {"paper_id": "p3", "results": "Trim-and-fill adjusted the pooled estimate."},
        {"paper_id": "clear", "abstract": "No meta-bias language here."},
    ]
    rows = PublicationBiasCueExtractor().extract(papers)
    by_id = {r.paper_id: r for r in rows}
    assert by_id["p1"].flagged and "funnel_plot" in by_id["p1"].cues
    assert by_id["p2"].flagged and "egger_test" in by_id["p2"].cues
    assert by_id["10.1/x"].flagged and "small_study" in by_id["10.1/x"].cues
    assert by_id["p3"].flagged and "trim_fill" in by_id["p3"].cues
    assert by_id["clear"].flagged is False


def test_does_not_mutate_inputs() -> None:
    papers = [{"paper_id": "p", "abstract": "publication bias noted"}]
    original = dict(papers[0])
    PublicationBiasCueExtractor().extract(papers)
    assert papers[0] == original
