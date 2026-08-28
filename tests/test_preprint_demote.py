"""Tests for preprint soft-demotion postprocessing."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.preprint_demote import PreprintDemoter


def _result(
    chunk_id: str,
    score: float,
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=f"text for {chunk_id}",
            source="test",
            metadata={} if metadata is None else metadata,
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


@pytest.mark.parametrize(
    "alpha",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        PreprintDemoter(alpha=alpha)


def test_demote_prefers_non_preprints() -> None:
    preprint = _result("pre", score=1.0, metadata={"venue": "arXiv"})
    journal = _result("jour", score=0.0, metadata={"venue": "Nature"})
    demoter = PreprintDemoter(alpha=1.0)

    demoted = demoter.demote([preprint, journal])

    assert [result.chunk.chunk_id for result in demoted] == ["jour", "pre"]
    assert demoted[0].score == pytest.approx(1.0)
    assert demoted[1].score == pytest.approx(0.2)
    assert demoted[0].retriever == "preprint_demote"
    assert demoted[0].path == ["hybrid", "bm25"]
    assert preprint.score == 1.0  # originals not mutated


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("publication_type", "preprint"),
        ("type", "biorxiv"),
        ("venue", "medRxiv"),
        ("venue", "SSRN Working Paper"),
        ("publication_type", "arXiv preprint"),
    ],
)
def test_detects_preprint_markers_across_keys(key: str, value: str) -> None:
    result = _result("only", score=0.0, metadata={key: value})
    demoter = PreprintDemoter(alpha=1.0)

    demoted = demoter.demote([result])

    assert demoted[0].score == pytest.approx(0.2)


def test_non_preprint_metadata_keeps_full_demote_score() -> None:
    result = _result("only", score=0.0, metadata={"type": "journal-article"})
    demoter = PreprintDemoter(alpha=1.0)

    demoted = demoter.demote([result])

    assert demoted[0].score == pytest.approx(1.0)


def test_missing_metadata_treated_as_non_preprint() -> None:
    result = _result("only", score=0.0, metadata={})
    demoter = PreprintDemoter(alpha=1.0)

    demoted = demoter.demote([result])

    assert demoted[0].score == pytest.approx(1.0)


def test_formula_matches_one_minus_alpha_old_plus_alpha_demote() -> None:
    result = _result("only", score=0.8, metadata={"venue": "arXiv"})
    demoter = PreprintDemoter(alpha=0.25)

    demoted = demoter.demote([result])

    assert demoted[0].score == pytest.approx((1 - 0.25) * 0.8 + 0.25 * 0.2)


def test_default_alpha_is_quarter() -> None:
    demoter = PreprintDemoter()
    result = _result("only", score=1.0, metadata={"type": "preprint"})

    demoted = demoter.demote([result])

    assert demoted[0].score == pytest.approx((1 - 0.25) * 1.0 + 0.25 * 0.2)


def test_stable_ordering_for_tied_scores() -> None:
    first = _result("first", score=0.5, metadata={"venue": "Nature"})
    second = _result("second", score=0.5, metadata={"venue": "Science"})
    demoter = PreprintDemoter(alpha=0.0)

    demoted = demoter.demote([first, second])

    assert [result.chunk.chunk_id for result in demoted] == ["first", "second"]


def test_top_k_truncates_after_sort() -> None:
    results = [
        _result("a", score=0.9, metadata={"venue": "arXiv"}),
        _result("b", score=0.1, metadata={"venue": "Nature"}),
        _result("c", score=0.2, metadata={"type": "preprint"}),
    ]
    demoter = PreprintDemoter(alpha=1.0)

    demoted = demoter.demote(results, top_k=2)

    assert len(demoted) == 2
    assert demoted[0].chunk.chunk_id == "b"


def test_empty_input_returns_empty() -> None:
    assert PreprintDemoter().demote([]) == []
