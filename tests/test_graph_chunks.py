"""Bounded graph chunk selection is independent of insertion and query plans."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from retrieval.models import Chunk, Entity
from storage.graph_store import SQLiteGraphStore

SELECTED_DOCUMENT = "Paper:10.1/'allowed-\u03b2'"


@pytest.mark.parametrize("scope", [None, (SELECTED_DOCUMENT,)], ids=["unscoped", "scoped"])
@pytest.mark.parametrize("limit", [0, 1, 2, 4, 10])
def test_chunks_for_entities_stable_across_insertion_and_planner_changes(
    tmp_path: Path, scope: tuple[str, ...] | None, limit: int
) -> None:
    prepared = [
        (
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title="Shared title",
                text="Shared evidence",
                source="fixture",
                metadata={"original_id": chunk_id},
            ),
            [Entity(name=name) for name in names],
        )
        for chunk_id, document_id, names in [
            ("z:late", SELECTED_DOCUMENT, ["Alpha", "Zeta", "ALPHA", "Zeta"]),
            ("m:middle", SELECTED_DOCUMENT, ["Alpha", "Zeta"]),
            ("A:quote'?", SELECTED_DOCUMENT, ["Zeta"]),
            ("a:case", SELECTED_DOCUMENT, ["Zeta"]),
            ("\u03b2:tail", SELECTED_DOCUMENT, ["Zeta"]),
            ("00:excluded", SELECTED_DOCUMENT.lower(), ["Alpha", "Zeta"]),
            ("!unmatched", SELECTED_DOCUMENT, ["Unrelated"]),
        ]
    ]
    expected = sorted(
        [
            chunk
            for chunk, _ in prepared
            if chunk.chunk_id != "!unmatched" and (scope is None or chunk.document_id in scope)
        ],
        key=lambda chunk: chunk.chunk_id,
    )[:limit]
    orders = [list(range(7)), list(reversed(range(7))), [3, 0, 6, 2, 5, 1, 4]]
    seeds = ["ZETA", "aLpHa"]
    observed: dict[str, list[Chunk]] = {}
    for number, order in enumerate(orders):
        database = tmp_path / f"graph-{number}.sqlite3"
        store = SQLiteGraphStore(database)
        for index in order:
            chunk, entities = prepared[index]
            store.add_mentions(chunk, entities if number == 0 else list(reversed(entities)))
        observed[f"{number}/initial"] = store.chunks_for_entities(
            seeds, limit=limit, document_ids=scope
        )
        observed[f"{number}/reopened"] = SQLiteGraphStore(database).chunks_for_entities(
            list(reversed(seeds)), limit=limit, document_ids=scope
        )

        # Real planner statistics may switch an indexed lookup to a mention-table scan.
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute("ANALYZE")
        observed[f"{number}/analyzed"] = SQLiteGraphStore(database).chunks_for_entities(
            seeds, limit=limit, document_ids=scope
        )
        for index in reversed(order):
            chunk, entities = prepared[index]
            store.replace_chunk(chunk, list(reversed(entities)), [])
        observed[f"{number}/replaced"] = SQLiteGraphStore(database).chunks_for_entities(
            list(reversed(seeds)), limit=limit, document_ids=scope
        )

    selected_ids = {
        label: tuple(chunk.chunk_id for chunk in chunks) for label, chunks in observed.items()
    }
    assert len(set(selected_ids.values())) == 1, selected_ids
    for chunks in observed.values():
        assert chunks == expected


@pytest.mark.parametrize("reverse_insertion", [False, True])
def test_chunks_for_entities_orders_across_documents_after_eligibility(
    tmp_path: Path, reverse_insertion: bool
) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    prepared = [
        ("00:excluded", SELECTED_DOCUMENT.lower(), ["Seed"]),
        ("!unmatched", SELECTED_DOCUMENT, ["Unrelated"]),
        ("z:late", SELECTED_DOCUMENT, ["Seed"]),
        ("a:first", "Selected", ["Seed", "Other", "SEED"]),
        ("b:second", SELECTED_DOCUMENT, ["Other"]),
    ]
    chunks: dict[str, Chunk] = {}
    for chunk_id, document_id, names in reversed(prepared) if reverse_insertion else prepared:
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title=chunk_id,
            text="Shared evidence",
            source="fixture",
        )
        chunks[chunk_id] = chunk
        store.add_mentions(chunk, [Entity(name=name) for name in names])

    scope = [SELECTED_DOCUMENT, "Selected", SELECTED_DOCUMENT, "missing"]
    seeds = ["seed", "Other", "SEED"]
    assert store.chunks_for_entities(seeds, limit=2, document_ids=scope) == [
        chunks["a:first"],
        chunks["b:second"],
    ]
    assert store.chunks_for_entities(seeds, limit=1, document_ids=scope) == [chunks["a:first"]]
    assert store.chunks_for_entities(seeds, document_ids=["missing"]) == []
    assert store.chunks_for_entities(["missing"], document_ids=scope) == []
    assert store.chunks_for_entities([], document_ids=scope) == []
