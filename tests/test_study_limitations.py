"""Tests for StudyLimitationCueExtractor."""

from retrieval.study_limitations import StudyLimitationCueExtractor


def test_empty_papers_ok() -> None:
    assert StudyLimitationCueExtractor().extract([]) == ()


def test_extracts_sample_size_generalizability_confounding_bias() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "A small sample size limited power in this pilot.",
        },
        {
            "id": "p2",
            "discussion": "Findings may not generalize beyond this single-center cohort.",
        },
        {
            "doi": "10.1000/lim",
            "limitations": "Residual confounding and unmeasured confounders remain.",
        },
        {
            "paper_id": "p3",
            "abstract": "Selection bias and recall bias cannot be excluded.",
        },
        {
            "paper_id": "clear",
            "abstract": "We report transformer latency on long documents.",
        },
    ]
    cues = StudyLimitationCueExtractor().extract(papers)
    assert len(cues) == 5
    by_id = {row.paper_id: row for row in cues}
    assert by_id["p1"].flagged
    assert "sample_size" in by_id["p1"].categories
    assert by_id["p1"].matched_cues
    assert by_id["p2"].flagged
    assert "generalizability" in by_id["p2"].categories
    assert by_id["10.1000/lim"].flagged
    assert "confounding" in by_id["10.1000/lim"].categories
    assert by_id["p3"].flagged
    assert "bias" in by_id["p3"].categories
    assert by_id["clear"].flagged is False
    assert by_id["clear"].categories == ()
    assert by_id["clear"].matched_cues == ()


def test_collects_multiple_categories_without_duplicates() -> None:
    rows = StudyLimitationCueExtractor().extract(
        [
            {
                "paper_id": "multi",
                "abstract": (
                    "Limitations include a small sample size, potential "
                    "confounding, and limited generalizability."
                ),
            }
        ]
    )
    assert rows[0].categories == ("sample_size", "generalizability", "confounding")
    assert len(rows[0].matched_cues) == 3


def test_never_mutates_input() -> None:
    paper = {
        "paper_id": "x",
        "abstract": "Underpowered due to small sample size.",
    }
    snapshot = dict(paper)
    StudyLimitationCueExtractor().extract([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = StudyLimitationCueExtractor.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "SampleSizeHintExtractor" in doc
        or "ConflictOfInterestFlagger" in doc
        or "MethodExtractCard" in doc
        or "Elicit" in doc
        or "Consensus" in doc
        or "PaperQA" in doc
    )
