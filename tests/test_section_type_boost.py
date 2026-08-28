"""Tests for section-type retrieval boosting."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.section_type_boost import SectionTypeBooster


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
        SectionTypeBooster(alpha=alpha)


def test_rejects_invalid_section_scores() -> None:
    with pytest.raises(ValueError, match="section_scores"):
        SectionTypeBooster(section_scores={"introduction": 1.5})


def test_boost_prefers_default_sections() -> None:
    intro = _result("intro", score=1.0, metadata={"section": "introduction"})
    results = _result("res", score=0.0, metadata={"section": "Results"})
    booster = SectionTypeBooster(alpha=1.0)

    boosted = booster.boost([intro, results])

    assert [result.chunk.chunk_id for result in boosted] == ["res", "intro"]
    assert boosted[0].score == pytest.approx(1.0)
    assert boosted[1].score == pytest.approx(0.2)
    assert boosted[0].retriever == "section_type_boost"
    assert boosted[0].path == ["hybrid", "bm25"]
    assert intro.score == 1.0  # originals not mutated


@pytest.mark.parametrize(
    "section",
    ["methods", "conclusion", "abstract", "RESULTS"],
)
def test_default_preferred_sections_score_one(section: str) -> None:
    result = _result("only", score=0.0, metadata={"section_type": section})
    booster = SectionTypeBooster(alpha=1.0)

    boosted = booster.boost([result])

    assert boosted[0].score == pytest.approx(1.0)


def test_reads_section_before_section_type() -> None:
    result = _result(
        "only",
        score=0.0,
        metadata={"section": "methods", "section_type": "introduction"},
    )
    booster = SectionTypeBooster(alpha=1.0)

    boosted = booster.boost([result])

    assert boosted[0].score == pytest.approx(1.0)


def test_custom_section_scores_overlay_defaults() -> None:
    result = _result("only", score=0.0, metadata={"section": "discussion"})
    booster = SectionTypeBooster(
        alpha=1.0,
        section_scores={"discussion": 0.8},
    )

    boosted = booster.boost([result])

    assert boosted[0].score == pytest.approx(0.8)


def test_formula_matches_one_minus_alpha_old_plus_alpha_section() -> None:
    result = _result("only", score=0.8, metadata={"section": "abstract"})
    booster = SectionTypeBooster(alpha=0.3)

    boosted = booster.boost([result])

    assert boosted[0].score == pytest.approx((1 - 0.3) * 0.8 + 0.3 * 1.0)


def test_stable_ordering_for_tied_scores() -> None:
    first = _result("first", score=0.5, metadata={"section": "methods"})
    second = _result("second", score=0.5, metadata={"section": "methods"})
    booster = SectionTypeBooster(alpha=0.0)

    boosted = booster.boost([first, second])

    assert [result.chunk.chunk_id for result in boosted] == ["first", "second"]


def test_top_k_truncates_after_sort() -> None:
    results = [
        _result("a", score=0.9, metadata={"section": "introduction"}),
        _result("b", score=0.1, metadata={"section": "results"}),
        _result("c", score=0.2, metadata={"section": "methods"}),
    ]
    booster = SectionTypeBooster(alpha=1.0)

    boosted = booster.boost(results, top_k=2)

    assert len(boosted) == 2
    assert boosted[0].chunk.chunk_id == "b"


def test_empty_input_returns_empty() -> None:
    assert SectionTypeBooster().boost([]) == []


def test_missing_section_uses_unknown_score() -> None:
    result = _result("only", score=0.0, metadata={})
    booster = SectionTypeBooster(alpha=1.0)

    boosted = booster.boost([result])

    assert boosted[0].score == pytest.approx(0.2)
