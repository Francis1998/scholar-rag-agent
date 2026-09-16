"""Tests for OpenDataAvailabilityFlagger."""

from retrieval.open_data_availability import OpenDataAvailabilityFlagger


def test_empty_papers_ok() -> None:
    assert OpenDataAvailabilityFlagger().flag([]) == ()


def test_flags_data_hosts_and_availability_phrases() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "Raw data are available upon request; open data release planned.",
        },
        {
            "id": "p2",
            "data_availability": "Deposited at zenodo.org under record 12345.",
        },
        {
            "doi": "10.1000/osf",
            "abstract": "Materials and data posted on OSF (osf.io/abcde).",
        },
        {
            "paper_id": "p3",
            "abstract": "Tables live in the supplementary data file.",
        },
        {
            "paper_id": "p4",
            "abstract": "Replication package at github.com/org/dataset-release.",
        },
        {
            "paper_id": "p5",
            "abstract": "Microarray files archived in Dryad and figshare.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without sharing underlying datasets.",
        },
    ]
    flags = OpenDataAvailabilityFlagger().flag(papers)
    assert len(flags) == 7
    by_id = {row.paper_id: row for row in flags}
    assert by_id["p1"].flagged
    assert by_id["p2"].flagged
    assert any("zenodo" in c.lower() for c in by_id["p2"].matched_cues)
    assert by_id["10.1000/osf"].flagged
    assert by_id["p3"].flagged
    assert any("supplement" in c.lower() for c in by_id["p3"].matched_cues)
    assert by_id["p4"].flagged
    assert any("github.com" in c.lower() for c in by_id["p4"].matched_cues)
    assert by_id["p5"].flagged
    assert by_id["clear"].flagged is False
    assert by_id["clear"].reasons == ()
    assert by_id["clear"].matched_cues == ()


def test_collects_multiple_open_data_cues() -> None:
    rows = OpenDataAvailabilityFlagger().flag(
        [
            {
                "paper_id": "multi",
                "abstract": (
                    "Data available on Zenodo; also mirrored at "
                    "github.com/lab/open-data with supplementary data."
                ),
            }
        ]
    )
    assert rows[0].flagged
    assert len(rows[0].matched_cues) >= 2


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "Data available at zenodo.org."}
    snapshot = dict(paper)
    OpenDataAvailabilityFlagger().flag([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = OpenDataAvailabilityFlagger.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "CodeAvailabilityBooster" in doc
        or "FundingDisclosureFlagger" in doc
        or "PreregistrationFlagDetector" in doc
        or "Elicit" in doc
        or "Consensus" in doc
        or "PaperQA" in doc
    )
