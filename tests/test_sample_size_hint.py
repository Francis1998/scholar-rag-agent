"""Tests for SampleSizeHintExtractor."""

from retrieval.sample_size_hint import SampleSizeHintExtractor


def test_empty_papers_ok() -> None:
    assert SampleSizeHintExtractor().extract([]) == ()


def test_extracts_n_equals_and_sample_size_phrases() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "We enrolled patients (N=120) in a multicenter trial.",
        },
        {
            "id": "p2",
            "abstract": "A sample size of 85 adults completed the protocol.",
        },
        {
            "doi": "10.1000/n",
            "abstract": "Secondary analysis included n = 42 records.",
        },
        {
            "paper_id": "p3",
            "title": "Pilot study",
            "abstract": "Fifty participants were randomized; total sample size was 50.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers on long documents without cohort counts.",
        },
    ]
    hints = SampleSizeHintExtractor().extract(papers)
    assert len(hints) == 5
    by_id = {hint.paper_id: hint for hint in hints}
    assert by_id["p1"].sample_size == 120
    assert by_id["p1"].matched_cue
    assert by_id["p2"].sample_size == 85
    assert "sample size" in by_id["p2"].matched_cue.lower()
    assert by_id["10.1000/n"].sample_size == 42
    assert by_id["p3"].sample_size == 50
    assert by_id["clear"].sample_size is None
    assert by_id["clear"].matched_cue == ""


def test_prefers_explicit_n_over_ambiguous_counts() -> None:
    hints = SampleSizeHintExtractor().extract(
        [
            {
                "paper_id": "both",
                "abstract": "Among 1000 screened candidates, N = 64 were analyzed.",
            }
        ]
    )
    assert hints[0].sample_size == 64


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "N=10 mice were tested."}
    snapshot = dict(paper)
    SampleSizeHintExtractor().extract([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = SampleSizeHintExtractor.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "MethodExtractCard" in doc
        or "method_extract" in doc
        or "Elicit" in doc
        or "Consensus" in doc
    )
