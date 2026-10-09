"""Graph neighbour budgets count stored entity identities, not display spellings."""

from pathlib import Path

import pytest

from retrieval.models import Chunk, Entity, EntityEdge
from storage.graph_store import PreparedGraphChunk, SQLiteGraphStore


def relationship(
    chunk_id: str,
    document_id: str,
    source: str,
    target: str,
    direction: str = "forward",
) -> PreparedGraphChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=chunk_id,
        text=f"{source} {target}",
        source="fixture",
    )
    pairs = {
        "forward": [(source, target)],
        "reverse": [(target, source)],
        "both": [(source, target), (target, source)],
    }[direction]
    return (
        chunk,
        [Entity(name=source), Entity(name=target)],
        [EntityEdge(source=left, target=right, chunk_id=chunk_id) for left, right in pairs],
    )


@pytest.mark.parametrize("direction", ["forward", "reverse", "both"])
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 10])
def test_neighbours_limit_counts_normalized_identities(
    tmp_path: Path, direction: str, limit: int
) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    for index, name in enumerate(["ALPHA", "Alpha", "Beta", "gamma"]):
        chunk, entities, edges = relationship(f"c{index}", f"d{index}", "Seed", name, direction)
        store.add_mentions(chunk, entities)
        store.add_edges(edges * 2)

    neighbours = store.neighbours(["sEeD", "SEED"], limit=limit)

    assert len({name.lower() for name in neighbours}) == min(limit, 3)
    assert neighbours == ["ALPHA", "Beta", "gamma"][:limit]


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["\u00c4PFEL", "\u00e4pfel", "Beta"], ["Beta", "\u00c4PFEL"]),
        (["\u0130ris", "i\u0307ris", "Zulu"], ["i\u0307ris", "Zulu"]),
        (["\u03a3igma", "\u03c3igma", "Zulu"], ["Zulu", "\u03a3igma"]),
        (["Stra\u00dfe", "STRASSE"], ["STRASSE", "Stra\u00dfe"]),
    ],
    ids=["non-ascii-lower", "expanded-lower", "greek-lower", "not-casefold"],
)
def test_neighbours_uses_stored_python_lower_keys(
    tmp_path: Path, names: list[str], expected: list[str]
) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    for index, name in enumerate(names):
        store.replace_chunk(
            *relationship(
                f"c{index}",
                f"d{index}",
                "\u00c9tude",
                name,
                "forward" if index % 2 else "reverse",
            )
        )

    assert store.neighbours(["\u00e9TUDE"], limit=10) == expected
    assert store.neighbours(["\u00e9TUDE"], limit=1) == expected[:1]


@pytest.mark.parametrize("reverse_insertion", [False, True])
def test_neighbours_are_stable_across_replacement_and_restart(
    tmp_path: Path, reverse_insertion: bool
) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    chunks = [
        relationship("c0", "d0", "Seed", "Alpha"),
        relationship("c1", "d1", "ALPHA", "OtherSeed"),
        relationship("c2", "d2", "Seed", "zulu"),
        relationship("c3", "d3", "ZULU", "OtherSeed"),
        relationship("c4", "d4", "beta", "Seed"),
        relationship("c5", "d5", "Seed", "Gamma"),
    ]
    if reverse_insertion:
        chunks.reverse()
    for prepared in chunks:
        store.replace_chunk(*prepared)
    expected = ["ALPHA", "beta", "Gamma", "ZULU"]
    seeds = ["sEeD", "OTHERSEED"]
    assert store.neighbours(seeds, limit=10) == expected
    assert store.neighbours(seeds, limit=2) == expected[:2]

    for prepared in reversed(chunks):
        store.replace_chunk(*prepared)
    reopened = SQLiteGraphStore(database)
    assert reopened.neighbours(list(reversed(seeds)), limit=10) == expected
    assert reopened.neighbours(seeds, limit=2) == expected[:2]

    reopened.replace_documents(
        frozenset(chunk.document_id for chunk, _, _ in chunks), list(reversed(chunks))
    )
    restarted = SQLiteGraphStore(database)
    assert restarted.neighbours(seeds, limit=10) == expected
    assert restarted.neighbours(seeds, limit=2) == expected[:2]


@pytest.mark.parametrize("direction", ["forward", "reverse", "both"])
def test_neighbours_scope_precedes_grouping_representative_and_limit(
    tmp_path: Path, direction: str
) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    selected_id = "Paper:10.1/'allowed-\u03b2'"
    for index, name in enumerate(["A0", "A1", "A2", "A3", "BETA", "ZETA"]):
        store.replace_chunk(
            *relationship(f"x{index}", selected_id.lower(), "Seed", name, direction)
        )
    store.add_edges([EntityEdge(source="Seed", target="A-orphan", chunk_id="missing")])
    for index, name in enumerate(["ALPHA", "Alpha", "Beta", "Zeta"]):
        document_id = selected_id if index % 2 else "Selected"
        store.replace_chunk(*relationship(f"c{index}", document_id, "Seed", name, direction))
    scope = [selected_id, "Selected"]

    assert store.neighbours(["seed"], limit=2) == ["A-orphan", "A0"]
    assert store.neighbours(["seed"], limit=2, document_ids=scope) == ["ALPHA", "Beta"]
    assert store.neighbours(["seed"], limit=10, document_ids=scope) == [
        "ALPHA",
        "Beta",
        "Zeta",
    ]
    assert store.neighbours(["seed"], limit=1, document_ids=scope) == ["ALPHA"]
    assert store.neighbours(["seed"], document_ids=["missing"]) == []


@pytest.mark.parametrize(
    ("seeds", "expected"),
    [([], []), (["missing"], []), (["seed"], ["Alpha", "camelCase"])],
    ids=["empty-seeds", "unknown-seeds", "unaliased-names"],
)
def test_neighbours_preserves_empty_results_and_unaliased_display_names(
    tmp_path: Path, seeds: list[str], expected: list[str]
) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    for index, name in enumerate(["Alpha", "camelCase"]):
        store.replace_chunk(*relationship(f"c{index}", f"d{index}", "Seed", name))

    assert store.neighbours(seeds) == expected
