"""Tests for FundingDisclosureFlagger."""

from retrieval.funding_disclosure import FundingDisclosureFlagger


def test_empty_papers_ok() -> None:
    assert FundingDisclosureFlagger().flag([]) == ()


def test_flags_nih_nsf_erc_and_funded_by_cues() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "This work was supported by the NIH under award R01LM012345.",
        },
        {
            "id": "p2",
            "acknowledgements": "We thank the NSF for grant NSF-IIS-1911234.",
        },
        {
            "doi": "10.1000/erc",
            "acknowledgment": "Funded by the ERC Starting Grant.",
        },
        {
            "paper_id": "p3",
            "funding": "This study was funded by Acme Foundation.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers on long documents without sponsor text.",
        },
    ]
    flags = FundingDisclosureFlagger().flag(papers)
    assert len(flags) == 5
    by_id = {flag.paper_id: flag for flag in flags}
    assert by_id["p1"].flagged is True
    assert any("NIH" in reason for reason in by_id["p1"].reasons)
    assert by_id["p2"].flagged is True
    assert any("NSF" in reason for reason in by_id["p2"].reasons)
    assert by_id["10.1000/erc"].flagged is True
    assert any("ERC" in reason for reason in by_id["10.1000/erc"].reasons)
    assert by_id["p3"].flagged is True
    assert any("funded by" in reason.lower() for reason in by_id["p3"].reasons)
    assert by_id["clear"].flagged is False
    assert by_id["clear"].reasons == ()


def test_detects_grant_number_patterns() -> None:
    flags = FundingDisclosureFlagger().flag(
        [
            {
                "paper_id": "g1",
                "abstract": "Support: grant number R21-HG009999 and U01CA123456.",
            }
        ]
    )
    assert flags[0].flagged is True
    assert any("grant" in reason.lower() for reason in flags[0].reasons)
    assert flags[0].matched_cues


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "Supported by NIH."}
    snapshot = dict(paper)
    FundingDisclosureFlagger().flag([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = FundingDisclosureFlagger.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "Crossref" in doc or "crossref_funder" in doc or "Scite" in doc or "Elicit" in doc
