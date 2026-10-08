"""Exact chunk identities retain only their current, atomically indexed graph."""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import NoReturn

import httpx
import pytest
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from ingestion.chunking import TextChunker
from ingestion.pipeline import IngestionPipeline
from retrieval.graph import GraphRAGBuilder, GraphRetriever, SpacyEntityExtractor
from retrieval.models import Chunk, Document, Entity, EntityEdge
from retrieval.multihop import MultiHopRetriever
from storage.graph_store import SQLiteGraphStore


class FixtureExtractor(SpacyEntityExtractor):
    def __init__(self) -> None:
        """Never import spaCy or load a model."""

    def extract(self, text: str) -> list[Entity]:
        return [Entity(name=word, label="FIXTURE") for word in text.split() if word[0].isupper()]


class FixedIdentityChunker(TextChunker):
    def chunk(self, document: Document) -> list[Chunk]:
        return [Chunk(chunk_id=f"{document.document_id}:0", **document.model_dump())]


def deny_network(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Graph indexing regressions must not use the network.")


@pytest.fixture(autouse=True)
def deterministic_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("retrieval.graph.SpacyEntityExtractor", FixtureExtractor)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_network)


def chunk(chunk_id: str, text: str, document_id: str = "selected") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=f"Title {chunk_id}",
        text=text,
        source="fixture",
        metadata={"revision": "1"},
    )


def graph_rows(database: Path) -> dict[str, list[tuple[object, ...]]]:
    """Compare complete graph payloads without the edges' internal sequence IDs."""
    queries = {
        "chunks": (
            "SELECT chunk_id, document_id, title, text, source, metadata "
            "FROM graph_chunks ORDER BY chunk_id"
        ),
        "mentions": (
            "SELECT chunk_id, entity_name, entity_key, label "
            "FROM entity_mentions ORDER BY chunk_id, entity_key"
        ),
        "edges": (
            "SELECT chunk_id, source_key, target_key, source_name, target_name, weight "
            "FROM entity_edges ORDER BY chunk_id, source_key, target_key"
        ),
    }
    with closing(sqlite3.connect(database)) as connection:
        return {name: connection.execute(query).fetchall() for name, query in queries.items()}


def test_replaying_chunk_keeps_edge_count_bounded_after_reopening(tmp_path: Path) -> None:
    database = tmp_path / "graph.sqlite3"
    original = chunk("stable", "Alpha Beta Gamma")
    GraphRAGBuilder(SQLiteGraphStore(database)).index_chunks([original])
    before = graph_rows(database)
    assert len(before["edges"]) == 3

    for _ in range(3):
        store = SQLiteGraphStore(database)
        GraphRAGBuilder(store).index_chunks([original, original])
        after = graph_rows(database)
        assert len(after["edges"]) == 3
        assert after == before
        assert store.neighbours(["aLpHa"]) == ["Beta", "Gamma"]


@pytest.mark.parametrize("reopen_legacy", [False, True], ids=["fresh", "legacy-reopen"])
@pytest.mark.parametrize(
    ("query", "index_name"),
    [
        (
            "EXPLAIN QUERY PLAN DELETE FROM entity_mentions WHERE chunk_id = ?",
            "idx_entity_mentions_chunk_id",
        ),
        (
            "EXPLAIN QUERY PLAN DELETE FROM entity_edges WHERE chunk_id = ?",
            "idx_entity_edges_chunk_id",
        ),
    ],
    ids=["mentions", "edges"],
)
def test_chunk_deletion_uses_indexes_without_rewriting_existing_data(
    tmp_path: Path, reopen_legacy: bool, query: str, index_name: str
) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    GraphRAGBuilder(store).index_chunks(
        [chunk("stable", "Alpha Beta"), chunk("other", "Alpha Gamma", "excluded")]
    )
    store.add_edges([EntityEdge(source="Alpha", target="Beta", chunk_id="stable", weight=0.25)])
    before = graph_rows(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        if reopen_legacy:
            connection.execute("DROP INDEX IF EXISTS idx_entity_mentions_chunk_id")
            connection.execute("DROP INDEX IF EXISTS idx_entity_edges_chunk_id")
        edge_ids = connection.execute("SELECT id FROM entity_edges ORDER BY id").fetchall()
    if reopen_legacy:
        SQLiteGraphStore(database)

    with closing(sqlite3.connect(database)) as connection:
        plans = [row[3] for row in connection.execute(query, ("stable",)).fetchall()]
        assert any("SEARCH" in plan and index_name in plan for plan in plans), plans
        assert connection.execute("SELECT id FROM entity_edges ORDER BY id").fetchall() == edge_ids
    assert graph_rows(database) == before


@pytest.mark.parametrize("reopen_legacy", [False, True], ids=["fresh", "legacy-reopen"])
@pytest.mark.parametrize(
    ("delete_sql", "remaining"),
    [
        (
            "DELETE FROM entity_mentions WHERE chunk_id IN "
            "(SELECT chunk_id FROM graph_chunks WHERE document_id = ?)",
            (1, 0, 2),
        ),
        (
            "DELETE FROM entity_edges WHERE chunk_id IN "
            "(SELECT chunk_id FROM graph_chunks WHERE document_id = ?)",
            (1, 2, 0),
        ),
        ("DELETE FROM graph_chunks WHERE document_id = ?", (0, 2, 2)),
    ],
    ids=["mentions-by-document", "edges-by-document", "chunks-by-document"],
)
def test_document_deletions_use_ownership_index_without_rewriting_data(
    tmp_path: Path, reopen_legacy: bool, delete_sql: str, remaining: tuple[int, int, int]
) -> None:
    database = tmp_path / "graph.sqlite3"
    document_id = "paper:10.1/'quoted-\u03b2'"
    store = SQLiteGraphStore(database)
    GraphRAGBuilder(store).index_chunks(
        [chunk("stable", "Alpha Beta", document_id), chunk("other", "Alpha Gamma", "excluded")]
    )
    store.add_edges([EntityEdge(source="Alpha", target="Beta", chunk_id="stable", weight=0.25)])
    before = graph_rows(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        if reopen_legacy:
            connection.execute("DROP INDEX IF EXISTS idx_graph_chunks_document_id")
        edge_ids = connection.execute("SELECT id FROM entity_edges ORDER BY id").fetchall()
    if reopen_legacy:
        SQLiteGraphStore(database)
    assert graph_rows(database) == before

    with closing(sqlite3.connect(database)) as connection:
        plans = [
            row[3]
            for row in connection.execute(
                "EXPLAIN QUERY PLAN " + delete_sql, (document_id,)
            ).fetchall()
        ]
        assert any("SEARCH" in plan and "idx_graph_chunks_document_id" in plan for plan in plans), (
            plans
        )
        assert not any("SCAN graph_chunks" in plan for plan in plans), plans
        assert connection.execute("SELECT id FROM entity_edges ORDER BY id").fetchall() == edge_ids
        connection.execute("BEGIN")
        connection.execute(delete_sql, (document_id,))
        counts_sql = (
            "SELECT (SELECT count(*) FROM graph_chunks WHERE chunk_id = ?), "
            "(SELECT count(*) FROM entity_mentions WHERE chunk_id = ?), "
            "(SELECT count(*) FROM entity_edges WHERE chunk_id = ?)"
        )
        assert connection.execute(counts_sql, ("stable",) * 3).fetchone() == remaining
        assert connection.execute(counts_sql, ("other",) * 3).fetchone() == (1, 2, 1)
        connection.rollback()
    assert graph_rows(database) == before


async def test_changed_chunk_replaces_old_entity_routes_and_payload(tmp_path: Path) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    builder = GraphRAGBuilder(store)
    original = chunk("stable", "Alpha Beta")
    current = original.model_copy(
        update={
            "document_id": "current-owner",
            "title": "Updated title",
            "text": "Gamma Delta",
            "source": "updated-fixture",
            "metadata": {"revision": "2"},
        }
    )
    builder.index_chunks([original])
    builder.index_chunks([current])

    direct = GraphRetriever(store)
    assert await direct.retrieve(["aLpHa", "bEtA"]) == []
    results = await direct.retrieve(["gAmMa"], document_ids=["current-owner"])
    assert [result.chunk for result in results] == [current]
    assert results[0].score == 1.0
    assert results[0].path == ["gAmMa"]
    assert await direct.retrieve(["Gamma"], document_ids=["selected"]) == []
    assert store.neighbours(["Alpha", "Beta"]) == []
    assert store.neighbours(["gAmMa"]) == ["Delta"]
    assert store.neighbours(["delta"]) == ["Gamma"]
    assert await MultiHopRetriever(store).retrieve("old", ["Alpha"]) == []
    assert [
        result.chunk for result in await MultiHopRetriever(store).retrieve("new", ["Gamma"])
    ] == [current]


async def test_reindex_removes_only_its_multihop_bridge_and_preserves_scope(tmp_path: Path) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    builder = GraphRAGBuilder(store)
    bridge = chunk("bridge", "Alpha Beta")
    builder.index_chunks(
        [
            chunk("head", "Alpha"),
            bridge,
            chunk("old-tail", "Beta"),
            chunk("new-tail", "Gamma"),
            chunk("excluded-bridge", "Alpha Beta", "excluded"),
        ]
    )
    before = graph_rows(database)
    current = bridge.model_copy(update={"text": "Alpha Gamma"})
    builder.index_chunks([current])

    retriever = MultiHopRetriever(store)
    results = await retriever.retrieve(
        "Alpha", ["aLpHa"], depth=2, limit=20, document_ids=["selected"]
    )
    assert {result.chunk.chunk_id: result.score for result in results} == {
        "head": 1.0,
        "bridge": 1.0,
        "new-tail": 0.5,
    }
    assert next(result for result in results if result.chunk.chunk_id == "new-tail").path == [
        "gamma"
    ]
    direct = await GraphRetriever(store).retrieve(["Beta"], document_ids=["selected"])
    assert [result.chunk.chunk_id for result in direct] == ["old-tail"]
    assert store.neighbours(["Alpha"], document_ids=["selected"]) == ["Gamma"]
    assert store.neighbours(["Beta"], document_ids=["selected"]) == []
    assert store.neighbours(["Gamma"], document_ids=["selected"]) == ["Alpha"]
    assert store.neighbours(["Alpha"], document_ids=["excluded"]) == ["Beta"]
    assert {result.chunk.chunk_id for result in await retriever.retrieve("Alpha", ["Alpha"])} == {
        "head",
        "bridge",
        "old-tail",
        "new-tail",
        "excluded-bridge",
    }
    for table, rows in graph_rows(database).items():
        assert [row for row in rows if row[0] != "bridge"] == [
            row for row in before[table] if row[0] != "bridge"
        ]


async def test_empty_entity_replacement_clears_links_but_keeps_current_chunk(
    tmp_path: Path,
) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    builder = GraphRAGBuilder(store)
    original = chunk("stable", "Alpha Beta")
    current = original.model_copy(update={"text": "no named entities"})
    builder.index_chunks([original])
    builder.index_chunks([current])

    rows = graph_rows(database)
    assert rows["mentions"] == []
    assert rows["edges"] == []
    assert len(rows["chunks"]) == 1
    assert rows["chunks"][0][3] == current.text
    assert await GraphRetriever(store).retrieve(["Alpha", "Beta"]) == []
    assert await MultiHopRetriever(store).retrieve("old", ["Alpha"]) == []
    assert store.neighbours(["Alpha", "Beta"]) == []


@pytest.mark.parametrize("final_text", ["Gamma Delta", "no named entities"])
def test_repeated_chunk_ids_in_one_batch_use_last_value(tmp_path: Path, final_text: str) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    builder = GraphRAGBuilder(store)
    unrelated = chunk("other", "Alpha Omega")
    original = chunk("stable", "Alpha Beta")
    current = original.model_copy(update={"text": final_text, "title": "Last value"})
    batch = [
        original,
        unrelated,
        original.model_copy(update={"text": "no named entities"}),
        current,
    ]
    builder.index_chunks(batch)

    assert store.chunks_for_entities(["Alpha"]) == [unrelated]
    assert store.chunks_for_entities(["Beta"]) == []
    assert store.chunks_for_entities(["Gamma"]) == (
        [current] if final_text == "Gamma Delta" else []
    )
    before = graph_rows(database)
    builder.index_chunks(batch)
    assert graph_rows(database) == before


@pytest.mark.parametrize("current_text", ["Gamma Delta", "no named entities"])
async def test_fixed_identity_pipeline_replacement_survives_app_restart(
    tmp_path: Path, current_text: str
) -> None:
    database = tmp_path / "ingestion.sqlite3"
    settings = offline_settings(database)
    container: AppContainer = create_app(settings).state.container
    pipeline = IngestionPipeline(
        container.document_store,
        container.hybrid_retriever,
        container.graph_builder,
        chunker=FixedIdentityChunker(),
    )
    original = Document(
        document_id="stable-document", title="Original", text="Alpha Beta", source="fixture"
    )
    old_chunks = pipeline.ingest_documents([original])
    current = original.model_copy(update={"text": current_text, "title": "Current"})
    current_chunks = pipeline.ingest_documents([current])
    assert [chunk.chunk_id for chunk in current_chunks] == [chunk.chunk_id for chunk in old_chunks]
    pipeline.ingest_documents([current])
    before = graph_rows(database)

    restarted: AppContainer = create_app(settings).state.container
    assert restarted.document_store.list_chunks() == current_chunks
    assert graph_rows(database) == before
    assert await GraphRetriever(restarted.graph_store).retrieve(["Alpha"]) == []
    assert await MultiHopRetriever(restarted.graph_store).retrieve("old", ["Alpha"]) == []
    expected = current_chunks if current_text == "Gamma Delta" else []
    assert restarted.graph_store.chunks_for_entities(["gamma"]) == expected
    restarted.graph_builder.index_chunks(current_chunks)
    assert graph_rows(database) == before


@pytest.mark.parametrize("stage", ["extraction", "edges"])
def test_graph_preparation_failure_preserves_previous_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    builder = GraphRAGBuilder(store)
    original = chunk("stable", "Alpha Beta")
    builder.index_chunks([original])
    before = graph_rows(database)

    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise ValueError("injected graph preparation failure")

    if stage == "extraction":
        monkeypatch.setattr(FixtureExtractor, "extract", fail)
    else:
        monkeypatch.setattr("retrieval.graph.EntityEdge", fail)
    with pytest.raises(ValueError, match="injected graph preparation failure"):
        builder.index_chunks([original.model_copy(update={"text": "Gamma Delta"})])
    assert graph_rows(database) == before


@pytest.mark.parametrize(
    "trigger",
    [
        "CREATE TRIGGER reject_chunk BEFORE INSERT ON graph_chunks "
        "WHEN NEW.text = 'Delta Gamma Omega' "
        "BEGIN SELECT RAISE(ABORT, 'injected insertion failure'); END",
        "CREATE TRIGGER reject_mention BEFORE INSERT ON entity_mentions "
        "WHEN NEW.entity_name = 'Omega' "
        "BEGIN SELECT RAISE(ABORT, 'injected insertion failure'); END",
        "CREATE TRIGGER reject_edge BEFORE INSERT ON entity_edges "
        "WHEN NEW.target_name = 'Omega' "
        "BEGIN SELECT RAISE(ABORT, 'injected insertion failure'); END",
    ],
    ids=["chunk", "mention", "edge"],
)
def test_failed_graph_insert_rolls_back_entire_chunk(tmp_path: Path, trigger: str) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    builder = GraphRAGBuilder(store)
    original = chunk("stable", "Alpha Beta")
    builder.index_chunks([original, chunk("unrelated", "Delta Gamma")])
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(trigger)
        before = list(connection.iterdump())

    with pytest.raises(sqlite3.IntegrityError, match="injected insertion failure"):
        builder.index_chunks([original.model_copy(update={"text": "Delta Gamma Omega"})])
    with closing(sqlite3.connect(database)) as connection:
        assert list(connection.iterdump()) == before
    assert store.chunks_for_entities(["Alpha"]) == [original]
    assert store.neighbours(["Beta"]) == ["Alpha"]


def test_replace_chunk_preserves_current_labels_weights_and_case(tmp_path: Path) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    original = chunk("stable", "Alpha Beta")
    GraphRAGBuilder(store).index_chunks([original])
    current = original.model_copy(update={"text": "ALPHA Gamma"})
    entities = [Entity(name="ALPHA", label="GENE"), Entity(name="Gamma", label="PATHWAY")]
    edges = [EntityEdge(source="ALPHA", target="Gamma", chunk_id="stable", weight=2.5)]

    store.replace_chunk(current, entities, edges)
    assert store.chunks_for_entities(["aLpHa"]) == [current]
    assert store.chunks_for_entities(["Beta"]) == []
    assert store.neighbours(["gAmMa"]) == ["ALPHA"]
    before = graph_rows(database)
    assert before["mentions"] == [
        ("stable", "ALPHA", "alpha", "GENE"),
        ("stable", "Gamma", "gamma", "PATHWAY"),
    ]
    assert before["edges"] == [("stable", "alpha", "gamma", "ALPHA", "Gamma", 2.5)]
    store.replace_chunk(current, entities, edges)
    assert graph_rows(database) == before


def test_replace_chunk_rejects_foreign_edge_ownership_before_opening_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    original = chunk("stable", "Alpha Beta")
    GraphRAGBuilder(store).index_chunks([original, chunk("other", "Gamma Delta")])
    before = graph_rows(database)
    current = original.model_copy(update={"text": "Gamma Delta"})

    def unexpected_connection(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Ownership must be validated before opening a transaction.")

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", unexpected_connection)
        with pytest.raises(ValueError, match="chunk_id"):
            store.replace_chunk(
                current,
                FixtureExtractor().extract(current.text),
                [
                    EntityEdge(source="Gamma", target="Delta", chunk_id="stable"),
                    EntityEdge(source="Gamma", target="Delta", chunk_id="other"),
                ],
            )
    assert graph_rows(database) == before


def test_low_level_graph_writes_remain_additive(tmp_path: Path) -> None:
    database = tmp_path / "graph.sqlite3"
    store = SQLiteGraphStore(database)
    original = chunk("stable", "Alpha")
    current = original.model_copy(update={"text": "Beta"})
    store.add_mentions(original, [Entity(name="Alpha")])
    store.add_mentions(current, [Entity(name="Beta")])
    edge = EntityEdge(source="Alpha", target="Beta", chunk_id="stable", weight=0.25)
    store.add_edges([edge])
    store.add_edges([edge])

    assert store.chunks_for_entities(["Alpha"]) == [current]
    assert store.chunks_for_entities(["Beta"]) == [current]
    assert len(graph_rows(database)["edges"]) == 2
