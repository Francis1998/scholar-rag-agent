"""Tests for PrimaryEndpointCueExtractor."""

from retrieval.primary_endpoint_cues import PrimaryEndpointCueExtractor


def test_empty_ok() -> None:
    assert PrimaryEndpointCueExtractor().extract([]) == ()


def test_primary_endpoint() -> None:
    rows = PrimaryEndpointCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": "The primary endpoint was overall survival at 12 months.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert "primary_endpoint" in rows[0].cues


def test_co_primary() -> None:
    rows = PrimaryEndpointCueExtractor().extract(
        [{"id": "p2", "abstract": "Co-primary outcomes included PFS and ORR."}]
    )
    assert rows[0].flagged is True
    assert "co_primary" in rows[0].cues


def test_clear() -> None:
    rows = PrimaryEndpointCueExtractor().extract(
        [{"paper_id": "clear", "abstract": "We describe a retrospective chart review."}]
    )
    assert rows[0].flagged is False
