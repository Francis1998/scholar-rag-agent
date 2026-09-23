"""Tests for SurrogateEndpointCueExtractor."""

from __future__ import annotations

from retrieval.surrogate_endpoint_cues import SurrogateEndpointCueExtractor


def test_extracts_cue() -> None:
    rows = SurrogateEndpointCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "The primary surrogate endpoint was ORR."}]
    )
    assert rows[0].flagged is True
    assert rows[0].endpoint_kind is not None
    assert "surrogate_endpoint" in rows[0].cues or rows[0].cues


def test_empty_papers() -> None:
    assert SurrogateEndpointCueExtractor().extract([]) == ()


def test_unflagged_without_match() -> None:
    rows = SurrogateEndpointCueExtractor().extract(
        [{"id": "x", "abstract": "No relevant methods text."}]
    )
    assert rows[0].flagged is False
