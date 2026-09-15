"""Tests for PreregistrationFlagDetector."""

from retrieval.preregistration_flag import PreregistrationFlagDetector


def test_empty_papers_ok() -> None:
    assert PreregistrationFlagDetector().detect([]) == ()


def test_detects_clinicaltrials_osf_isrctn_prereg() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "Registered at clinicaltrials.gov (NCT01234567).",
        },
        {
            "id": "p2",
            "methods": "Materials were posted on the Open Science Framework (osf.io).",
        },
        {
            "doi": "10.1000/isrctn",
            "registration": "ISRCTN12345678",
        },
        {
            "paper_id": "p3",
            "abstract": "This study was preregistered prior to data collection.",
        },
        {
            "paper_id": "p4",
            "abstract": "Protocol underwent pre-registration on an institutional registry.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without registry identifiers.",
        },
    ]
    flags = PreregistrationFlagDetector().detect(papers)
    assert len(flags) == 6
    by_id = {row.paper_id: row for row in flags}
    assert by_id["p1"].flagged
    assert "clinicaltrials.gov" in by_id["p1"].matched_cues or any(
        "NCT" in c for c in by_id["p1"].matched_cues
    )
    assert by_id["p2"].flagged
    assert by_id["10.1000/isrctn"].flagged
    assert any("ISRCTN" in c.upper() for c in by_id["10.1000/isrctn"].matched_cues)
    assert by_id["p3"].flagged
    assert by_id["p4"].flagged
    assert by_id["clear"].flagged is False
    assert by_id["clear"].reasons == ()
    assert by_id["clear"].matched_cues == ()


def test_collects_multiple_cues() -> None:
    rows = PreregistrationFlagDetector().detect(
        [
            {
                "paper_id": "multi",
                "abstract": (
                    "Preregistered on OSF and registered at clinicaltrials.gov as NCT09876543."
                ),
            }
        ]
    )
    assert rows[0].flagged
    assert len(rows[0].matched_cues) >= 2


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "NCT01234567 was the trial ID."}
    snapshot = dict(paper)
    PreregistrationFlagDetector().detect([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = PreregistrationFlagDetector.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "FundingDisclosureFlagger" in doc
        or "PrismaScreeningChecklist" in doc
        or "Elicit" in doc
        or "Consensus" in doc
        or "PaperQA" in doc
    )
