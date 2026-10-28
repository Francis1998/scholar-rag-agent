"""BM25 tuning must fail early without changing supported scoring behavior."""

import math
import sys
from decimal import Decimal
from fractions import Fraction
from unittest.mock import AsyncMock, Mock

import pytest

from retrieval.models import Chunk
from retrieval.sparse import BM25Retriever


def corpus() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=chunk_id,
            document_id=chunk_id,
            title="BM25 tuning fixture",
            text=text,
            source="fixture",
        )
        for chunk_id, text in [
            ("short", "alpha"),
            ("repeat", "alpha alpha beta"),
            ("other", "beta beta"),
        ]
    ]


@pytest.mark.parametrize("parameter", ["k1", "b"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param("0.5", id="numeric-string"),
        pytest.param("private-payload", id="nonnumeric-string"),
        pytest.param(None, id="none"),
        pytest.param(0.5 + 0j, id="complex"),
        pytest.param(Decimal("0.5"), id="decimal-is-not-real"),
    ],
)
def test_rejects_non_real_tuning_at_construction(parameter: str, value: object) -> None:
    with pytest.raises(
        TypeError, match=rf"^{parameter} must be a real number \(excluding booleans\)$"
    ):
        BM25Retriever(**{parameter: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("parameter", ["k1", "b"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="positive-infinity"),
        pytest.param(-math.inf, id="negative-infinity"),
        pytest.param(10**400, id="oversized-integer"),
        pytest.param(Fraction(10**400), id="oversized-fraction"),
    ],
)
def test_rejects_nonfinite_or_unrepresentable_tuning(parameter: str, value: object) -> None:
    with pytest.raises(
        ValueError, match=rf"^{parameter} must be finite and representable as a float$"
    ):
        BM25Retriever(**{parameter: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        pytest.param("k1", -1, id="k1-zero-denominator"),
        pytest.param("k1", -2, id="k1-negative-score"),
        pytest.param("k1", math.nextafter(0.0, -math.inf), id="k1-below-zero"),
        pytest.param("k1", Fraction(-1, 10**400), id="k1-rounds-to-negative-zero"),
        pytest.param("b", -4, id="b-negative-score"),
        pytest.param("b", 3, id="b-zero-denominator"),
        pytest.param("b", math.nextafter(0.0, -math.inf), id="b-below-zero"),
        pytest.param("b", math.nextafter(1.0, math.inf), id="b-above-one"),
        pytest.param("b", Fraction(10**400 + 1, 10**400), id="b-rounds-to-one"),
    ],
)
def test_rejects_out_of_range_tuning(parameter: str, value: object) -> None:
    message = r"^k1 must be >= 0$" if parameter == "k1" else r"^b must be within \[0, 1\]$"
    with pytest.raises(ValueError, match=message):
        BM25Retriever(**{parameter: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(("parameter", "value"), [("k1", -1), ("b", 3)])
async def test_invalid_tuning_prevents_indexing_and_retrieval(
    parameter: str, value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_chunks = Mock()
    retrieve = AsyncMock(return_value=[])
    monkeypatch.setattr(BM25Retriever, "add_chunks", add_chunks)
    monkeypatch.setattr(BM25Retriever, "retrieve", retrieve)

    with pytest.raises(ValueError, match=rf"^{parameter} "):
        retriever = BM25Retriever(**{parameter: value})
        retriever.add_chunks(corpus())
        await retriever.retrieve("alpha")

    add_chunks.assert_not_called()
    retrieve.assert_not_called()


async def test_defaults_preserve_existing_scores_and_ranking() -> None:
    default = BM25Retriever()
    explicit = BM25Retriever(k1=1.5, b=0.75)
    for retriever in (default, explicit):
        retriever.add_chunks(corpus())

    results = await default.retrieve("alpha")

    assert results == await explicit.retrieve("alpha")
    assert [result.chunk.chunk_id for result in results] == ["short", "repeat", "other"]
    assert [result.score for result in results] == pytest.approx(
        [0.6064562958009492, 0.5784660052255207, 0.0]
    )


@pytest.mark.parametrize(
    ("k1", "b", "first", "first_weight", "second_weight"),
    [
        pytest.param(0, 0, "repeat", 1, 1, id="integer-zero"),
        pytest.param(0.0, 1.0, "repeat", 1, 1, id="zero-with-full-length-normalization"),
        pytest.param(2, 0, "repeat", 3 / 2, 1, id="no-length-normalization"),
        pytest.param(2, 1, "short", 3 / 2, 6 / 5, id="full-length-normalization"),
        pytest.param(0.6, 0.2, "repeat", 160 / 133, 80 / 77, id="custom-floats"),
        pytest.param(
            Fraction(3, 5),
            Fraction(1, 5),
            "repeat",
            160 / 133,
            80 / 77,
            id="real-fractions",
        ),
    ],
)
async def test_supported_tuning_preserves_scores(
    k1: float, b: float, first: str, first_weight: float, second_weight: float
) -> None:
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.add_chunks(corpus())

    results = await retriever.retrieve("alpha")

    second = "short" if first == "repeat" else "repeat"
    assert [result.chunk.chunk_id for result in results] == [first, second, "other"]
    # Two of three chunks match; the mean length is two tokens.
    idf = math.log(1.6)
    assert [result.score for result in results] == pytest.approx(
        [idf * first_weight, idf * second_weight, 0.0]
    )


@pytest.mark.parametrize(
    "k1",
    [
        pytest.param(-0.0, id="negative-zero"),
        pytest.param(math.nextafter(0.0, math.inf), id="smallest-positive-float"),
        pytest.param(sys.float_info.max, id="largest-finite-float"),
    ],
)
def test_finite_k1_is_not_clamped(k1: float) -> None:
    retriever = BM25Retriever(k1=k1, b=1)

    assert retriever._k1 == k1
    assert retriever._b == 1


@pytest.mark.parametrize(
    ("k1", "b"),
    [
        pytest.param(Fraction(3, 5), Fraction(1, 5), id="custom-real-fractions"),
        pytest.param(0, 1, id="zero-k1"),
        pytest.param(sys.float_info.max, 0, id="maximum-k1-no-normalization"),
        pytest.param(sys.float_info.max, 0.75, id="maximum-k1-default-normalization"),
        pytest.param(sys.float_info.max, 1, id="maximum-k1-full-normalization"),
    ],
)
async def test_document_replacement_preserves_valid_tuning(k1: float, b: float) -> None:
    short, repeated, other = corpus()
    original = short.model_copy(update={"chunk_id": "old", "text": "stale passage"})
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.add_chunks([original, repeated, other])
    before = await retriever.retrieve("alpha")

    with retriever.replacing_documents(frozenset({"short"}), [short]):
        assert await retriever.retrieve("alpha") == before

    fresh = BM25Retriever(k1=k1, b=b)
    fresh.add_chunks(corpus())
    results = await retriever.retrieve("alpha")

    assert all(math.isfinite(result.score) for result in results)
    assert results == await fresh.retrieve("alpha")
    assert {result.chunk.chunk_id for result in results} == {"short", "repeat", "other"}
    assert retriever._k1 == k1
    assert retriever._b == b
