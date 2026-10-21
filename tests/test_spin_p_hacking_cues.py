"""Tests for SpinPHackingCueExtractor."""

from __future__ import annotations

from retrieval.spin_p_hacking_cues import SpinPHackingCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert SpinPHackingCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known phrase flags paper."""

    rows = SpinPHackingCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Authors show spin reporting and possible p-hacking concerns.",
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "spin_reporting" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract stays clean."""

    rows = SpinPHackingCueExtractor().extract(
        [{"id": "p2", "title": "Routine cohort description without bias terms."}]
    )
    assert rows[0].flagged is False
