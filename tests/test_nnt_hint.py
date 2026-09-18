"""Tests for NumberNeededToTreatHintExtractor."""

from retrieval.nnt_hint import NumberNeededToTreatHintExtractor


def test_empty_papers_ok() -> None:
    assert NumberNeededToTreatHintExtractor().extract([]) == ()


def test_extracts_nnt_nnh_and_arr_forms() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "The number needed to treat was 12 for the primary endpoint.",
        },
        {
            "id": "p2",
            "abstract": "Secondary safety analysis found NNH = 45.",
        },
        {
            "doi": "10.1000/arr",
            "abstract": "Absolute risk reduction ARR: 8.5% versus placebo.",
        },
        {
            "paper_id": "p3",
            "results": "NNT of 7 over five years of follow-up.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without absolute-risk reporting.",
        },
    ]
    hints = NumberNeededToTreatHintExtractor().extract(papers)
    assert len(hints) == 5
    by_id = {hint.paper_id: hint for hint in hints}
    assert by_id["p1"].metric == "NNT"
    assert by_id["p1"].value == 12.0
    assert by_id["p1"].matched_cue
    assert by_id["p2"].metric == "NNH"
    assert by_id["p2"].value == 45.0
    assert by_id["10.1000/arr"].metric == "ARR"
    assert by_id["10.1000/arr"].value == 8.5
    assert by_id["p3"].metric == "NNT"
    assert by_id["p3"].value == 7.0
    assert by_id["clear"].metric is None
    assert by_id["clear"].value is None
    assert by_id["clear"].matched_cue == ""


def test_prefers_nnt_over_arr_when_both_present() -> None:
    hints = NumberNeededToTreatHintExtractor().extract(
        [
            {
                "paper_id": "both",
                "abstract": "ARR was 5% corresponding to NNT = 20.",
            }
        ]
    )
    assert hints[0].metric == "NNT"
    assert hints[0].value == 20.0


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "NNT = 9 in the intention-to-treat set."}
    snapshot = dict(paper)
    NumberNeededToTreatHintExtractor().extract([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = NumberNeededToTreatHintExtractor.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "EffectSizeHintExtractor" in doc
        or "SampleSizeHintExtractor" in doc
        or "PValueHintExtractor" in doc
        or "Elicit" in doc
        or "Consensus" in doc
        or "PaperQA" in doc
    )
