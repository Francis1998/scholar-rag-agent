"""Tests for language preference postprocessing."""

import pytest

from retrieval.language_prefer import LanguagePreferencer
from retrieval.models import Chunk, SearchResult


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
        LanguagePreferencer(alpha=alpha)


@pytest.mark.parametrize("mode", ["drop", "rank", ""])
def test_rejects_invalid_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        LanguagePreferencer(mode=mode)


def test_rejects_empty_preferred_languages() -> None:
    with pytest.raises(ValueError, match="preferred_languages"):
        LanguagePreferencer(preferred_languages=["", "  "])


def test_boost_prefers_english_by_default() -> None:
    other = _result("de", score=1.0, metadata={"language": "de"})
    english = _result("en", score=0.0, metadata={"language": "en"})
    preferencer = LanguagePreferencer(alpha=1.0, mode="boost")

    preferred = preferencer.prefer([other, english])

    assert [result.chunk.chunk_id for result in preferred] == ["en", "de"]
    assert preferred[0].score == pytest.approx(1.0)
    assert preferred[1].score == pytest.approx(0.0)
    assert preferred[0].retriever == "language_prefer"
    assert preferred[0].path == ["hybrid", "bm25"]
    assert other.score == 1.0  # originals not mutated


def test_reads_lang_key_and_custom_preferred() -> None:
    fr = _result("fr", score=0.0, metadata={"lang": "FR"})
    en = _result("en", score=0.0, metadata={"lang": "en"})
    preferencer = LanguagePreferencer(
        preferred_languages=["fr"],
        alpha=1.0,
    )

    preferred = preferencer.prefer([en, fr])

    assert preferred[0].chunk.chunk_id == "fr"
    assert preferred[0].score == pytest.approx(1.0)
    assert preferred[1].score == pytest.approx(0.0)


def test_formula_matches_one_minus_alpha_old_plus_alpha_lang() -> None:
    result = _result("only", score=0.8, metadata={"language": "en"})
    preferencer = LanguagePreferencer(alpha=0.3)

    preferred = preferencer.prefer([result])

    assert preferred[0].score == pytest.approx((1 - 0.3) * 0.8 + 0.3 * 1.0)


def test_filter_drops_non_preferred_when_any_match_exists() -> None:
    english = _result("en", score=0.2, metadata={"language": "en"})
    german = _result("de", score=0.9, metadata={"language": "de"})
    preferencer = LanguagePreferencer(mode="filter")

    preferred = preferencer.prefer([german, english])

    assert preferred == [english]
    assert preferred[0] is english


def test_filter_keeps_all_when_no_preferred_language() -> None:
    first = _result("first", score=0.4, metadata={"language": "de"})
    second = _result("second", score=0.6, metadata={})
    preferencer = LanguagePreferencer(mode="filter")

    preferred = preferencer.prefer([first, second])

    assert preferred == [first, second]
    assert preferred[0] is first


def test_stable_ordering_for_tied_boost_scores() -> None:
    first = _result("first", score=0.5, metadata={"language": "en"})
    second = _result("second", score=0.5, metadata={"language": "en"})
    preferencer = LanguagePreferencer(alpha=0.0)

    preferred = preferencer.prefer([first, second])

    assert [result.chunk.chunk_id for result in preferred] == ["first", "second"]


def test_top_k_truncates_after_preference() -> None:
    results = [
        _result("a", score=0.9, metadata={"language": "de"}),
        _result("b", score=0.1, metadata={"language": "en"}),
        _result("c", score=0.2, metadata={"language": "en"}),
    ]
    preferencer = LanguagePreferencer(alpha=1.0)

    preferred = preferencer.prefer(results, top_k=2)

    assert len(preferred) == 2
    assert preferred[0].chunk.chunk_id == "b"


def test_empty_input_returns_empty() -> None:
    assert LanguagePreferencer().prefer([]) == []


def test_multiple_preferred_languages() -> None:
    de = _result("de", score=0.0, metadata={"language": "de"})
    fr = _result("fr", score=0.0, metadata={"language": "fr"})
    en = _result("en", score=0.0, metadata={"language": "en"})
    preferencer = LanguagePreferencer(
        preferred_languages=["de", "fr"],
        alpha=1.0,
    )

    preferred = preferencer.prefer([en, de, fr])

    scores = {result.chunk.chunk_id: result.score for result in preferred}
    assert scores["de"] == pytest.approx(1.0)
    assert scores["fr"] == pytest.approx(1.0)
    assert scores["en"] == pytest.approx(0.0)
