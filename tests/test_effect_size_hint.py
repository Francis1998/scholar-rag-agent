"""Tests for EffectSizeHintExtractor."""

from retrieval.effect_size_hint import EffectSizeHintExtractor


def test_empty_papers_ok() -> None:
    assert EffectSizeHintExtractor().extract([]) == ()


def test_extracts_cohens_d_or_hr_rr_auc() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "The intervention yielded Cohen's d = 0.82.",
        },
        {
            "id": "p2",
            "abstract": "Adjusted odds ratio OR=1.45 for the primary endpoint.",
        },
        {
            "doi": "10.1000/hr",
            "abstract": "Hazard ratio HR = 0.67 favored treatment.",
        },
        {
            "paper_id": "p3",
            "abstract": "Relative risk RR: 1.20 across cohorts.",
        },
        {
            "paper_id": "p4",
            "results": "Classifier AUC = 0.91 on the hold-out set.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without reported effect sizes.",
        },
    ]
    hints = EffectSizeHintExtractor().extract(papers)
    assert len(hints) == 6
    by_id = {hint.paper_id: hint for hint in hints}
    assert by_id["p1"].metric == "cohens_d"
    assert by_id["p1"].value == 0.82
    assert by_id["p1"].matched_cue
    assert by_id["p2"].metric == "OR"
    assert by_id["p2"].value == 1.45
    assert by_id["10.1000/hr"].metric == "HR"
    assert by_id["10.1000/hr"].value == 0.67
    assert by_id["p3"].metric == "RR"
    assert by_id["p3"].value == 1.2
    assert by_id["p4"].metric == "AUC"
    assert by_id["p4"].value == 0.91
    assert by_id["clear"].metric is None
    assert by_id["clear"].value is None
    assert by_id["clear"].matched_cue == ""


def test_prefers_cohens_d_over_later_metrics() -> None:
    hints = EffectSizeHintExtractor().extract(
        [
            {
                "paper_id": "both",
                "abstract": "Cohen's d = 0.5 with OR=2.1 and AUC=0.8.",
            }
        ]
    )
    assert hints[0].metric == "cohens_d"
    assert hints[0].value == 0.5


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "OR = 1.1 in subgroup analysis."}
    snapshot = dict(paper)
    EffectSizeHintExtractor().extract([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = EffectSizeHintExtractor.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "SampleSizeHintExtractor" in doc
        or "Elicit" in doc
        or "Consensus" in doc
        or "PaperQA" in doc
    )
