"""Unit tests for HealthyWorkerEffectCueExtractor."""

from __future__ import annotations

from retrieval.healthy_worker_effect_cues import HealthyWorkerEffectCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = HealthyWorkerEffectCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "We discuss the healthy worker effect as a selection bias in"
                    "occupational cohorts."
                ),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert HealthyWorkerEffectCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = HealthyWorkerEffectCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
