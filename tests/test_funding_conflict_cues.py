"""Unit tests for FundingConflictCueExtractor."""

from __future__ import annotations

from retrieval.funding_conflict_cues import FundingConflictCueExtractor

_ABS = "This industry-funded trial disclosed conflicts of interest."


def test_flagged_on_match() -> None:
    """Known cue text is flagged."""

    rows = FundingConflictCueExtractor().extract([{"paper_id": "p1", "abstract": _ABS}])
    assert rows[0].flagged is True
    assert rows[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert FundingConflictCueExtractor().extract([]) == ()


def test_unflagged_without_cues() -> None:
    """Benign abstract is not flagged."""

    rows = FundingConflictCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "We report general laboratory methods."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi_id() -> None:
    """Falls back to doi when paper_id missing."""

    rows = FundingConflictCueExtractor().extract([{"doi": "10.1/x", "abstract": _ABS}])
    assert rows[0].paper_id == "10.1/x"
