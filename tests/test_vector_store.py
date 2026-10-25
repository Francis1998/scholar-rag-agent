"""The optional vector store owns indexed data and detaches search results."""

import math
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from retrieval.models import Chunk, SearchResult
from storage.vector_store import InMemoryVectorStore

INVALID_LIMITS = [
    pytest.param(-1, id="negative"),
    pytest.param(-(10**400), id="huge-negative"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(0.0, id="zero-float"),
    pytest.param(1.0, id="integral-float"),
    pytest.param(1.5, id="fractional-float"),
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="infinity"),
    pytest.param(-math.inf, id="negative-infinity"),
    pytest.param("1", id="numeric-string"),
    pytest.param("private-limit-value", id="private-string"),
    pytest.param(None, id="none"),
    pytest.param([], id="list"),
    pytest.param({}, id="dict"),
]


def make_chunk(chunk_id: str = "chunk") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="paper",
        title="Original",
        text="Original evidence",
        source="fixture",
        metadata={"tag": "original"},
    )


@pytest.mark.parametrize("limit", INVALID_LIMITS)
@pytest.mark.parametrize("populated", [False, True], ids=["empty", "populated"])
def test_search_rejects_invalid_limits(limit: object, populated: bool) -> None:
    store = InMemoryVectorStore()
    if populated:
        for chunk_id in ("z", "a", "m"):
            store.upsert(make_chunk(chunk_id), [1.0, 0.0])
    error = ValueError if isinstance(limit, int) and not isinstance(limit, bool) else TypeError
    message = (
        r"^limit must be >= 0$"
        if error is ValueError
        else r"^limit must be an integer \(excluding booleans\)$"
    )

    with pytest.raises(error, match=message):
        store.search([1.0, 0.0], limit=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [pytest.param(0, id="zero"), *INVALID_LIMITS])
@pytest.mark.parametrize("populated", [False, True], ids=["empty", "populated"])
def test_limit_is_handled_before_vector_or_chunk_work(limit: object, populated: bool) -> None:
    store = InMemoryVectorStore()
    if populated:
        store.upsert(make_chunk(), [1.0, 0.0])
    query = FailingVector([1.0, 0.0])
    error = ValueError if isinstance(limit, int) and not isinstance(limit, bool) else TypeError

    with (
        patch(
            "storage.vector_store.cosine_similarity",
            side_effect=AssertionError("search must not score vectors"),
        ) as score,
        patch.object(
            FailingVector,
            "__iter__",
            side_effect=AssertionError("search must not iterate the query"),
        ) as iterate,
        patch.object(
            Chunk,
            "model_copy",
            side_effect=AssertionError("search must not copy chunks"),
        ) as copy,
    ):
        if isinstance(limit, int) and not isinstance(limit, bool) and limit == 0:
            assert store.search(query, limit=limit) == []
        else:
            with pytest.raises(error, match=r"^limit must be "):
                store.search(query, limit=limit)  # type: ignore[arg-type]

    score.assert_not_called()
    iterate.assert_not_called()
    copy.assert_not_called()


@pytest.mark.parametrize("boundary", ["input", "result"])
@pytest.mark.parametrize(
    "field", ["chunk_id", "document_id", "title", "text", "source", "metadata"]
)
def test_chunk_mutations_do_not_change_indexed_evidence(boundary: str, field: str) -> None:
    store = InMemoryVectorStore()
    chunk = make_chunk()
    store.upsert(chunk, [1.0, 0.0])
    caller_chunk = chunk if boundary == "input" else store.search([1.0, 0.0])[0].chunk

    if field == "metadata":
        caller_chunk.metadata["tag"] = "mutated"
        caller_chunk.metadata["added"] = "caller-owned"
    else:
        setattr(caller_chunk, field, "mutated")

    for query, score in [([1.0, 0.0], 1.0), ([0.0, 1.0], 0.0)]:
        assert store.search(query) == [
            SearchResult(chunk=make_chunk(), score=score, retriever="vector_store")
        ]
    if boundary == "result":
        assert chunk == make_chunk()


@pytest.mark.parametrize("boundary", ["input", "result"])
def test_nested_metadata_is_detached(boundary: str) -> None:
    # Trusted model_copy updates can carry nested values without schema validation.
    chunk = make_chunk().model_copy(update={"metadata": {"nested": {"tags": ["original"]}}})
    expected = chunk.model_copy(deep=True)
    store = InMemoryVectorStore()
    store.upsert(chunk, [1.0, 0.0])
    caller_chunk = chunk if boundary == "input" else store.search([1.0, 0.0])[0].chunk
    nested = caller_chunk.metadata["nested"]
    assert isinstance(nested, dict)
    nested["tags"].append("mutated")

    assert store.search([1.0, 0.0])[0].chunk == expected
    if boundary == "result":
        assert chunk == expected


@pytest.mark.parametrize("replacement", [[0.0, 1.0], [], [1.0, 0.0, 0.0]])
def test_vector_value_and_length_mutations_do_not_change_scores(
    replacement: list[float],
) -> None:
    store = InMemoryVectorStore()
    vector = [1.0, 0.0]
    store.upsert(make_chunk(), vector)
    assert vector == [1.0, 0.0]

    vector[:] = replacement

    assert store.search([1.0, 0.0])[0].score == 1.0
    assert store.search([0.0, 1.0])[0].score == 0.0
    assert vector == replacement


def test_search_results_have_independent_identity_and_do_not_mutate_inputs() -> None:
    store = InMemoryVectorStore()
    chunk = make_chunk()
    store.upsert(chunk, [1.0, 0.0])
    first = store.search([1.0, 0.0])
    second = store.search([1.0, 0.0])
    expected = [SearchResult(chunk=make_chunk(), score=1.0, retriever="vector_store")]

    assert first == second == expected
    assert first[0] is not second[0]
    assert first[0].chunk is not chunk
    assert first[0].chunk is not second[0].chunk
    assert first[0].chunk.metadata is not chunk.metadata
    assert first[0].chunk.metadata is not second[0].chunk.metadata
    first[0].chunk.text = "mutated"
    first[0].chunk.metadata.clear()
    first[0].score = -999.0
    first[0].retriever = "caller"
    first[0].path.append("caller")
    first.clear()

    assert second == store.search([1.0, 0.0]) == expected
    assert chunk == make_chunk()


def test_repeated_id_replaces_both_snapshots_without_reordering_ties() -> None:
    store = InMemoryVectorStore()
    store.upsert(make_chunk("z"), [1.0, 0.0])
    store.upsert(make_chunk("a"), [0.0, 1.0])
    before = store.search([1.0, 0.0])[0]
    replacement = make_chunk("z").model_copy(
        update={"text": "Replacement evidence", "metadata": {"tag": "replacement"}}
    )
    expected = replacement.model_copy(deep=True)
    vector = [0.0, 1.0]
    store.upsert(replacement, vector)
    replacement.text = "later mutation"
    replacement.metadata.clear()
    vector.clear()

    for query, score in [([0.0, 1.0], 1.0), ([1.0, 0.0], 0.0)]:
        assert store.search(query) == [
            SearchResult(chunk=expected, score=score, retriever="vector_store"),
            SearchResult(chunk=make_chunk("a"), score=score, retriever="vector_store"),
        ]
    assert before == SearchResult(chunk=make_chunk("z"), score=1.0, retriever="vector_store")


@pytest.mark.parametrize("limit", [0, 1, 2, 10, 10**400])
@pytest.mark.parametrize(
    ("query", "ranking"),
    [
        ([1.0, 0.0], [("z", 1.0), ("a", 1.0), ("m", 0.0)]),
        ([0.0, 1.0], [("m", 1.0), ("z", 0.0), ("a", 0.0)]),
        ([-1.0, 0.0], [("m", 0.0), ("z", -1.0), ("a", -1.0)]),
    ],
)
def test_search_preserves_scores_insert_order_limits_and_query(
    query: list[float], ranking: list[tuple[str, float]], limit: int
) -> None:
    store = InMemoryVectorStore()
    store.upsert(make_chunk("z"), [1.0, 0.0])
    store.upsert(make_chunk("a"), [1.0, 0.0])
    store.upsert(make_chunk("m"), [0.0, 1.0])
    original_query = query.copy()

    assert store.search(query, limit=limit) == [
        SearchResult(chunk=make_chunk(chunk_id), score=score, retriever="vector_store")
        for chunk_id, score in ranking[:limit]
    ]
    assert query == original_query


def test_default_limit_returns_ten_results_in_insertion_order() -> None:
    store = InMemoryVectorStore()
    chunk_ids = [f"c{index:02d}" for index in reversed(range(12))]
    for chunk_id in chunk_ids:
        store.upsert(make_chunk(chunk_id), [1.0, 0.0])

    assert store.search([1.0, 0.0]) == [
        SearchResult(chunk=make_chunk(chunk_id), score=1.0, retriever="vector_store")
        for chunk_id in chunk_ids[:10]
    ]


def test_empty_store_and_dimension_errors_do_not_change_indexed_data() -> None:
    store = InMemoryVectorStore()
    assert store.search([]) == []
    assert store.search([1.0, 0.0]) == []
    store.upsert(make_chunk(), [1.0, 0.0])

    with pytest.raises(ValueError, match="zip"):
        store.search([1.0])

    assert store.search([1.0, 0.0]) == [
        SearchResult(chunk=make_chunk(), score=1.0, retriever="vector_store")
    ]


class FailingVector(list[float]):
    def __iter__(self) -> Iterator[float]:
        raise RuntimeError("vector snapshot failed")


@pytest.mark.parametrize("failure", ["chunk", "vector"])
def test_failed_snapshot_leaves_previous_chunk_and_vector_together(failure: str) -> None:
    store = InMemoryVectorStore()
    store.upsert(make_chunk(), [1.0, 0.0])
    replacement = make_chunk().model_copy(update={"text": "Replacement evidence"})

    with pytest.raises(RuntimeError, match=f"{failure} snapshot failed"):
        if failure == "chunk":
            with patch.object(
                Chunk, "model_copy", side_effect=RuntimeError("chunk snapshot failed")
            ):
                store.upsert(replacement, [0.0, 1.0])
        else:
            store.upsert(replacement, FailingVector([0.0, 1.0]))

    assert store.search([1.0, 0.0]) == [
        SearchResult(chunk=make_chunk(), score=1.0, retriever="vector_store")
    ]
