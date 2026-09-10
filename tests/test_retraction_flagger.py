"""Tests for RetractionWatchFlagger."""

from retrieval.retraction_flagger import RetractionWatchFlagger


def test_empty_papers_ok() -> None:
    flagger = RetractionWatchFlagger({"10.1000/xyz": "retracted"})
    assert flagger.flag([]) == ()


def test_matches_doi_and_advises_without_dropping() -> None:
    papers = [
        {"doi": "https://doi.org/10.1000/xyz", "title": "Retracted Study"},
        {"doi": "10.1000/abc", "title": "Clean Study"},
    ]
    flags = RetractionWatchFlagger(
        {"10.1000/xyz": "retracted", "10.1000/withdrawn": "withdrawn"}
    ).flag(papers)
    assert len(flags) == 2
    assert flags[0].matched is True
    assert flags[0].status == "retracted"
    assert "Advisory" in flags[0].advisory
    assert flags[1].matched is False
    assert flags[1].status == "clear"
    # Advisory only — both input papers still represented.
    assert {flag.paper_id for flag in flags} == {
        "https://doi.org/10.1000/xyz",
        "10.1000/abc",
    }


def test_withdrawn_and_expression_of_concern_aliases() -> None:
    papers = [
        {"paper_id": "p1"},
        {"id": "p2"},
    ]
    flags = RetractionWatchFlagger({"p1": "withdrawal", "p2": "expression of concern"}).flag(papers)
    assert flags[0].status == "withdrawn"
    assert flags[1].status == "expression_of_concern"
    assert flags[0].matched and flags[1].matched


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = RetractionWatchFlagger.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "Semantic Scholar" in doc or "OpenAlex" in doc
    assert "RetractedFilter" in doc
