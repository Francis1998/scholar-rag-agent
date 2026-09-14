"""Tests for ConflictOfInterestFlagger."""

from retrieval.conflict_of_interest import ConflictOfInterestFlagger


def test_empty_papers_ok() -> None:
    assert ConflictOfInterestFlagger().flag([]) == ()


def test_flags_coi_and_competing_interest_cues() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "The authors declare a conflict of interest with Acme Pharma.",
        },
        {
            "id": "p2",
            "acknowledgements": "Competing interests: J.S. is a consultant for BioCo.",
        },
        {
            "doi": "10.1000/disc",
            "disclosure": "Financial disclosure: stock options in MedTech.",
        },
        {
            "paper_id": "p3",
            "conflicts": "Author serves on the advisory board of DeviceInc.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers on long documents without disclosure text.",
        },
    ]
    flags = ConflictOfInterestFlagger().flag(papers)
    assert len(flags) == 5
    by_id = {flag.paper_id: flag for flag in flags}
    assert by_id["p1"].flagged is True
    assert any("conflict of interest" in reason.lower() for reason in by_id["p1"].reasons)
    assert by_id["p2"].flagged is True
    assert any("competing" in reason.lower() for reason in by_id["p2"].reasons)
    assert by_id["10.1000/disc"].flagged is True
    assert any("disclosure" in reason.lower() for reason in by_id["10.1000/disc"].reasons)
    assert by_id["p3"].flagged is True
    assert any("advisory" in reason.lower() for reason in by_id["p3"].reasons)
    assert by_id["clear"].flagged is False
    assert by_id["clear"].reasons == ()


def test_detects_no_conflicts_declaration() -> None:
    flags = ConflictOfInterestFlagger().flag(
        [
            {
                "paper_id": "none",
                "abstract": "The authors declare no conflicts of interest.",
            }
        ]
    )
    assert flags[0].flagged is True
    assert any("no conflict" in reason.lower() for reason in flags[0].reasons)
    assert flags[0].matched_cues


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "Conflict of interest: consulting fees."}
    snapshot = dict(paper)
    ConflictOfInterestFlagger().flag([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = ConflictOfInterestFlagger.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "FundingDisclosure" in doc
        or "funding_disclosure" in doc
        or "Scite" in doc
        or "Elicit" in doc
    )
