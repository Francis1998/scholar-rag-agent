"""Tests for PreprintVersionDiffer."""

from retrieval.preprint_version import PreprintVersionDiffer


def test_unknown_when_both_missing() -> None:
    report = PreprintVersionDiffer().diff(None, None)
    assert report.status == "unknown"
    assert report.prefer == "none"
    assert report.reasons


def test_preprint_only() -> None:
    report = PreprintVersionDiffer().diff(
        {"title": "Draft", "year": 2023, "version": "v1"},
        None,
    )
    assert report.status == "preprint_only"
    assert report.prefer == "preprint"


def test_published_only_preferred() -> None:
    report = PreprintVersionDiffer().diff(
        None,
        {"title": "Journal version", "year": 2024, "venue": "Nature"},
    )
    assert report.status == "published_preferred"
    assert report.prefer == "published"


def test_same_work_preprint_older_by_year_and_venue() -> None:
    report = PreprintVersionDiffer().diff(
        {"title": "Work", "year": 2022, "version": "v1", "venue": "arXiv"},
        {"title": "Work", "year": 2024, "version": "v2", "venue": "NeurIPS"},
    )
    assert report.status == "same_work_preprint_older"
    assert report.prefer == "published"
    assert any("year delta" in reason for reason in report.reasons)
    assert any("venue" in reason for reason in report.reasons)


def test_published_preferred_when_venue_without_year_delta() -> None:
    report = PreprintVersionDiffer().diff(
        {"title": "Work", "version": "v1"},
        {"title": "Work", "version": "v1", "journal": "JMLR"},
    )
    assert report.status == "published_preferred"
    assert report.prefer == "published"


def test_unknown_when_insufficient_signals() -> None:
    report = PreprintVersionDiffer().diff(
        {"title": "A"},
        {"title": "B"},
    )
    assert report.status == "unknown"
    assert report.prefer == "either"


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = PreprintVersionDiffer.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "PreprintDemoter" in doc
    assert "Semantic Scholar" in doc or "ResearchRabbit" in doc
