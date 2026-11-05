"""Tests for AttritionResidualCueExtractor."""

from __future__ import annotations

from retrieval.attrition_residual_cues import (
    AttritionResidualCueExtractor,
)


def test_empty() -> None:
    """Empty input returns empty."""

    assert AttritionResidualCueExtractor().extract([]) == ()


def test_match() -> None:
    """Matching abstract is flagged."""

    rows = AttritionResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss attrition residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = AttritionResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methodology discussed."}]
    )
    assert rows[0].flagged is False


def test_missing_id() -> None:
    """Missing paper_id falls back."""

    rows = AttritionResidualCueExtractor().extract([{"abstract": "attrition residual"}])
    assert rows[0].paper_id.startswith("paper-")
