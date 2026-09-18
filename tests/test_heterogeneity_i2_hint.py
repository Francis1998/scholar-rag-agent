"""Tests for HeterogeneityI2HintExtractor."""

from retrieval.heterogeneity_i2_hint import HeterogeneityI2HintExtractor


def test_empty_papers_ok() -> None:
    assert HeterogeneityI2HintExtractor().extract([]) == ()


def test_extracts_i2_and_heterogeneity_forms() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "Pooled analysis showed I2 = 45% across trials.",
        },
        {
            "id": "p2",
            "abstract": "We observed I^2=60% with a random-effects model.",
        },
        {
            "doi": "10.1000/i2",
            "abstract": "Heterogeneity was substantial (I\u00b2 = 72%).",
        },
        {
            "paper_id": "p3",
            "results": "There was considerable heterogeneity among studies.",
        },
        {
            "paper_id": "p4",
            "abstract": "Between-study heterogeneity limited the pooled estimate.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without meta-analytic pooling.",
        },
    ]
    hints = HeterogeneityI2HintExtractor().extract(papers)
    assert len(hints) == 6
    by_id = {hint.paper_id: hint for hint in hints}
    assert by_id["p1"].cue_kind == "i2"
    assert by_id["p1"].i2_percent == 45.0
    assert by_id["p1"].matched_cue
    assert by_id["p2"].cue_kind == "i2"
    assert by_id["p2"].i2_percent == 60.0
    assert by_id["10.1000/i2"].cue_kind == "i2"
    assert by_id["10.1000/i2"].i2_percent == 72.0
    assert by_id["p3"].cue_kind == "heterogeneity"
    assert by_id["p3"].i2_percent is None
    assert by_id["p4"].cue_kind == "heterogeneity"
    assert by_id["clear"].cue_kind is None
    assert by_id["clear"].i2_percent is None
    assert by_id["clear"].matched_cue == ""


def test_prefers_numeric_i2_over_heterogeneity_phrase() -> None:
    hints = HeterogeneityI2HintExtractor().extract(
        [
            {
                "paper_id": "both",
                "abstract": ("Substantial heterogeneity was noted; I2 = 55% overall."),
            }
        ]
    )
    assert hints[0].cue_kind == "i2"
    assert hints[0].i2_percent == 55.0


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "I2 = 30% in the sensitivity analysis."}
    snapshot = dict(paper)
    HeterogeneityI2HintExtractor().extract([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = HeterogeneityI2HintExtractor.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "EffectSizeHintExtractor" in doc
        or "ConfidenceIntervalHintExtractor" in doc
        or "PValueHintExtractor" in doc
        or "Elicit" in doc
        or "Consensus" in doc
        or "PaperQA" in doc
    )
