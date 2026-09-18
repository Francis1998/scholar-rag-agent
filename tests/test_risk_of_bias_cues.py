"""Tests for RiskOfBiasCueExtractor."""

from retrieval.risk_of_bias_cues import RiskOfBiasCueExtractor


def test_empty_papers_ok() -> None:
    assert RiskOfBiasCueExtractor().extract([]) == ()


def test_extracts_cochrane_rob_domains() -> None:
    papers = [
        {
            "paper_id": "p1",
            "abstract": "Patients were randomly assigned to arms.",
        },
        {
            "id": "p2",
            "methods": "Allocation concealment used sealed opaque envelopes.",
        },
        {
            "doi": "10.1000/rob",
            "abstract": "This was a double-blind placebo-controlled trial.",
        },
        {
            "paper_id": "p3",
            "results": "Incomplete outcome data were handled via ITT analysis.",
        },
        {
            "paper_id": "p4",
            "abstract": "Loss to follow-up was under 5% with few drop-outs.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without trial reporting cues.",
        },
    ]
    rows = RiskOfBiasCueExtractor().extract(papers)
    assert len(rows) == 6
    by_id = {row.paper_id: row for row in rows}
    assert by_id["p1"].flagged
    assert "randomization" in by_id["p1"].domains
    assert by_id["p2"].flagged
    assert "allocation_concealment" in by_id["p2"].domains
    assert by_id["10.1000/rob"].flagged
    assert "blinding" in by_id["10.1000/rob"].domains
    assert by_id["p3"].flagged
    assert "incomplete_outcome" in by_id["p3"].domains
    assert by_id["p4"].flagged
    assert "incomplete_outcome" in by_id["p4"].domains
    assert by_id["clear"].flagged is False
    assert by_id["clear"].domains == ()
    assert by_id["clear"].matched_cues == ()


def test_collects_multiple_rob_domains() -> None:
    rows = RiskOfBiasCueExtractor().extract(
        [
            {
                "paper_id": "multi",
                "abstract": (
                    "Participants were randomized with allocation concealment; "
                    "outcome assessors were blinded and attrition was low."
                ),
            }
        ]
    )
    assert rows[0].flagged
    assert len(rows[0].domains) >= 3
    assert "randomization" in rows[0].domains
    assert "allocation_concealment" in rows[0].domains
    assert "blinding" in rows[0].domains


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "abstract": "A randomized double-blind trial."}
    snapshot = dict(paper)
    RiskOfBiasCueExtractor().extract([paper])
    assert paper == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = RiskOfBiasCueExtractor.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "StudyLimitationCueExtractor" in doc
        or "ConflictOfInterestFlagger" in doc
        or "MethodExtractCard" in doc
        or "Elicit" in doc
        or "Consensus" in doc
        or "PaperQA" in doc
    )
