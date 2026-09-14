"""Tests for PrismaScreeningChecklist."""

from retrieval.prisma_screening import PrismaScreeningChecklist


def test_empty_papers_ok() -> None:
    assert PrismaScreeningChecklist().build([], inclusion=[], exclusion=[]) == ()


def test_builds_pending_rows_never_auto_decides() -> None:
    papers = [
        {"paper_id": "p1", "title": "RCT of drug X in adults", "abstract": "Randomized trial."},
        {"id": "p2", "title": "Case report of drug X", "abstract": "Single patient."},
    ]
    rows = PrismaScreeningChecklist().build(
        papers,
        inclusion=["randomized trial", "adults"],
        exclusion=["case report", "animal"],
    )
    assert len(rows) == 2
    assert all(row.decision == "pending" for row in rows)
    assert all(row.stage in {"title_abstract", "screening"} for row in rows)
    by_id = {row.paper_id: row for row in rows}
    assert by_id["p1"].inclusion_hits
    assert by_id["p2"].exclusion_hits
    # Advisory only — cues never flip decision away from pending.
    assert by_id["p1"].decision == "pending"
    assert by_id["p2"].decision == "pending"


def test_records_criteria_checklist_items() -> None:
    rows = PrismaScreeningChecklist().build(
        [
            {
                "paper_id": "p1",
                "title": "Meta-analysis of vaccines",
                "abstract": "Systematic review.",
            }
        ],
        inclusion=["meta-analysis"],
        exclusion=["protocol only"],
    )
    assert rows[0].checklist
    assert any("include" in item.lower() or "meta" in item.lower() for item in rows[0].checklist)
    assert rows[0].decision == "pending"


def test_never_mutates_input() -> None:
    paper = {"paper_id": "x", "title": "Study", "abstract": "Text"}
    inclusion = ["study"]
    exclusion = ["animal"]
    snap_paper, snap_inc, snap_exc = dict(paper), list(inclusion), list(exclusion)
    PrismaScreeningChecklist().build([paper], inclusion=inclusion, exclusion=exclusion)
    assert paper == snap_paper
    assert inclusion == snap_inc
    assert exclusion == snap_exc


def test_docstring_mentions_frontier_models_and_hitl_gap() -> None:
    doc = PrismaScreeningChecklist.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "HITL" in doc or "human" in doc.lower() or "never" in doc.lower()
    assert "PRISMA" in doc or "Elicit" in doc or "Covidence" in doc
