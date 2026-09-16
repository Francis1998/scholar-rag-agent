"""Tests for ConfidenceIntervalHintExtractor."""

from retrieval.confidence_interval_hint import ConfidenceIntervalHintExtractor


def test_empty_papers_ok() -> None:
    assert ConfidenceIntervalHintExtractor().extract([]) == ()


def test_extracts_95_ci_and_confidence_interval_forms() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "OR was 2.1 (95% CI 1.2-3.4) for the primary endpoint.",
        },
        {
            "id": "p2",
            "abstract": "Hazard ratio 0.9 with 95% CI: 0.8 to 1.1.",
        },
        {
            "doi": "10.1000/ci",
            "abstract": "Accuracy had a confidence interval (0.45, 0.62).",
        },
        {
            "paper_id": "p3",
            "results": "Mean difference CI = [1.5, 2.8] across cohorts.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without reported interval estimates.",
        },
    ]
    hints = ConfidenceIntervalHintExtractor().extract(papers)
    assert len(hints) == 5
    by_id = {hint.paper_id: hint for hint in hints}
    assert by_id["p1"].level == 95
    assert by_id["p1"].low == 1.2
    assert by_id["p1"].high == 3.4
    assert by_id["p1"].matched_cue
    assert by_id["p2"].level == 95
    assert by_id["p2"].low == 0.8
    assert by_id["p2"].high == 1.1
    assert by_id["10.1000/ci"].low == 0.45
    assert by_id["10.1000/ci"].high == 0.62
    assert by_id["p3"].low == 1.5
    assert by_id["p3"].high == 2.8
    assert by_id["clear"].level is None
    assert by_id["clear"].low is None
    assert by_id["clear"].high is None
    assert by_id["clear"].matched_cue == ""


def test_prefers_labeled_95_ci_over_bare_ci() -> None:
    hints = ConfidenceIntervalHintExtractor().extract(
        [
            {
                "paper_id": "both",
                "abstract": "Secondary CI = 9-10; primary endpoint 95% CI 1.0-2.0.",
            }
        ]
    )
    # Prefer labeled 95% CI pattern over bare CI regardless of text order.
    assert hints[0].level == 95
    assert hints[0].low == 1.0
    assert hints[0].high == 2.0


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "Effect 1.4 (95% CI 1.1-1.8)."}
    snapshot = dict(paper)
    ConfidenceIntervalHintExtractor().extract([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = ConfidenceIntervalHintExtractor.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "EffectSizeHintExtractor" in doc
        or "PValueHintExtractor" in doc
        or "Elicit" in doc
        or "Consensus" in doc
        or "PaperQA" in doc
    )
