"""Tests for PValueHintExtractor."""

from retrieval.p_value_hint import PValueHintExtractor


def test_empty_papers_ok() -> None:
    assert PValueHintExtractor().extract([]) == ()


def test_extracts_p_lt_eq_and_pvalue_forms() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "The primary endpoint reached significance (p < 0.05).",
        },
        {
            "id": "p2",
            "abstract": "Secondary analysis found p=0.01 for the subgroup.",
        },
        {
            "doi": "10.1000/pval",
            "abstract": "Reported P-value = 0.003 after Bonferroni correction.",
        },
        {
            "paper_id": "p3",
            "results": "Interaction term P value: 0.12 was not significant.",
        },
        {
            "paper_id": "p4",
            "abstract": "Survival differed with p<=0.001 across arms.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without reported significance tests.",
        },
    ]
    hints = PValueHintExtractor().extract(papers)
    assert len(hints) == 6
    by_id = {hint.paper_id: hint for hint in hints}
    assert by_id["p1"].operator == "<"
    assert by_id["p1"].value == 0.05
    assert by_id["p1"].matched_cue
    assert by_id["p2"].operator == "="
    assert by_id["p2"].value == 0.01
    assert by_id["10.1000/pval"].operator == "="
    assert by_id["10.1000/pval"].value == 0.003
    assert by_id["p3"].operator == "="
    assert by_id["p3"].value == 0.12
    assert by_id["p4"].operator == "<="
    assert by_id["p4"].value == 0.001
    assert by_id["clear"].operator is None
    assert by_id["clear"].value is None
    assert by_id["clear"].matched_cue == ""


def test_prefers_earliest_p_value_in_text() -> None:
    hints = PValueHintExtractor().extract(
        [
            {
                "paper_id": "both",
                "abstract": "First hit p < 0.05 later followed by p=0.20.",
            }
        ]
    )
    assert hints[0].operator == "<"
    assert hints[0].value == 0.05


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "p = 0.04 in the intention-to-treat set."}
    snapshot = dict(paper)
    PValueHintExtractor().extract([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = PValueHintExtractor.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "EffectSizeHintExtractor" in doc
        or "SampleSizeHintExtractor" in doc
        or "Elicit" in doc
        or "Consensus" in doc
        or "PaperQA" in doc
    )
