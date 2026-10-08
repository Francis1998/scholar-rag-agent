"""Complete-document ingestion must replace, not accumulate, current evidence."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Barrier
from typing import NoReturn
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from ingestion.chunking import TextChunker
from ingestion.pipeline import IngestionPipeline
from retrieval.dense import DenseRetriever
from retrieval.embeddings import HashEmbeddingModel
from retrieval.graph import GraphRAGBuilder, GraphRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Chunk, Document, Entity
from retrieval.multihop import MultiHopRetriever
from retrieval.sparse import BM25Retriever
from storage.document_store import SQLiteDocumentStore
from storage.graph_store import SQLiteGraphStore
from tests.test_graph_reindexing import FixtureExtractor, deny_network, graph_rows

OLD_TEXT = "Alpha Beta obsolete methodology."
NEW_TEXT = "Gamma Delta revised methodology."
Retriever = DenseRetriever | BM25Retriever | HybridRetriever


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("retrieval.graph.SpacyEntityExtractor", FixtureExtractor)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_network)


def document(document_id: str = "same-paper", text: str = OLD_TEXT) -> Document:
    return Document(
        document_id=document_id,
        title="Synthetic",
        text=text,
        source="fixture",
        metadata={"revision": "1"},
    )


def make_container(database: Path, *, separate_graph: bool = False) -> AppContainer:
    container = AppContainer(offline_settings(database))
    if separate_graph:
        container.graph_store = SQLiteGraphStore(database.with_suffix(".graph.sqlite3"))
        container.graph_builder = GraphRAGBuilder(container.graph_store)
        container.ingestion_pipeline = IngestionPipeline(
            container.document_store, container.hybrid_retriever, container.graph_builder
        )
    return container


def components(hybrid: HybridRetriever) -> tuple[Retriever, ...]:
    return hybrid._dense_retriever, hybrid._sparse_retriever, hybrid


def dump(database: Path) -> str:
    with closing(sqlite3.connect(database)) as connection:
        return "\n".join(connection.iterdump())


async def assert_current(container: AppContainer, expected: list[Chunk]) -> None:
    assert container.document_store.list_chunks() == sorted(expected, key=lambda c: c.chunk_id)
    fresh = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    fresh.add_chunks(expected)
    scopes = [None, ["same-paper"], ["other"], ["missing"]]
    for actual, reference in zip(
        components(container.hybrid_retriever), components(fresh), strict=True
    ):
        for query in ("Alpha", "Beta", "Gamma Delta", "background", "absent"):
            for scope in scopes:
                for limit in (0, 1, 20):
                    assert await actual.retrieve(
                        query, limit=limit, document_ids=scope
                    ) == await reference.retrieve(query, limit=limit, document_ids=scope)


@pytest.mark.parametrize("surface", ["documents", "dense", "bm25", "hybrid", "graph", "multihop"])
async def test_real_chunker_revision_removes_superseded_evidence(
    tmp_path: Path, surface: str
) -> None:
    container = make_container(tmp_path / "corpus.sqlite3")
    original = document()
    old = container.ingestion_pipeline.ingest_documents([original])
    current = container.ingestion_pipeline.ingest_documents([document(text=NEW_TEXT)])
    assert [chunk.chunk_id for chunk in old] == ["chunk-140dc63fdf0152ba"]
    assert [chunk.chunk_id for chunk in current] == ["chunk-114e37427eb733dd"]

    if surface == "documents":
        assert container.document_store.list_chunks() == current
    elif surface in ("dense", "bm25", "hybrid"):
        retriever = dict(
            zip(("dense", "bm25", "hybrid"), components(container.hybrid_retriever), strict=True)
        )[surface]
        assert [hit.chunk for hit in await retriever.retrieve("Alpha", limit=10)] == current
    elif surface == "graph":
        assert container.graph_store.chunks_for_entities(["Alpha", "Beta"]) == []
        assert container.graph_store.chunks_for_entities(["Gamma"]) == current
        assert container.graph_store.neighbours(["Alpha", "Beta"]) == []
    else:
        assert await MultiHopRetriever(container.graph_store).retrieve("Alpha", ["Alpha"]) == []
        assert [
            hit.chunk
            for hit in await MultiHopRetriever(container.graph_store).retrieve("Gamma", ["Gamma"])
        ] == current


@pytest.mark.parametrize("text", ["shorter", "", " \t\n\u2003"])
async def test_shorter_or_blank_revision_and_identical_replay_survive_restart(
    tmp_path: Path, text: str
) -> None:
    database = tmp_path / "corpus.sqlite3"
    container = make_container(database)
    original = document(text="Alpha Beta " * 200)
    untouched = document("other", "background Gamma evidence.")
    initial = container.ingestion_pipeline.ingest_documents([original, untouched])
    old = [chunk for chunk in initial if chunk.document_id == original.document_id]
    assert len(old) > 2
    current_document = original.model_copy(
        update={
            "text": original.text[:801] if text == "shorter" else text,
            "title": "Current title",
            "source": "revised-fixture",
            "metadata": {"revision": "2"},
        }
    )
    expected = TextChunker().chunk(current_document)
    assert len(expected) < len(old)
    if text == "shorter":
        assert expected[0].chunk_id == old[0].chunk_id
    else:
        assert expected == []
    retained = [chunk for chunk in initial if chunk.document_id == untouched.document_id]

    assert container.ingestion_pipeline.ingest_documents([current_document]) == expected
    await assert_current(container, [*expected, *retained])
    graph_before = graph_rows(database)
    assert container.ingestion_pipeline.ingest_documents([current_document]) == expected
    assert graph_rows(database) == graph_before
    for runtime in (container, make_container(database)):
        await assert_current(runtime, [*expected, *retained])
        summaries = runtime.document_catalog.list_documents().documents
        assert {row.document_id: row.chunk_count for row in summaries} == {
            original.document_id: len(expected),
            untouched.document_id: len(retained),
        }
        assert [
            row.chunk_id for row in runtime.document_chunks.list_chunks("same-paper").chunks
        ] == sorted(chunk.chunk_id for chunk in expected)
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT text, title, source, metadata FROM documents WHERE document_id = ?",
            (original.document_id,),
        ).fetchone()
    assert row == (
        current_document.text,
        current_document.title,
        current_document.source,
        json.dumps(current_document.metadata, sort_keys=True),
    )


@pytest.mark.parametrize("last_text", [NEW_TEXT, ""])
async def test_duplicate_document_ids_are_whole_document_last_value_wins(
    tmp_path: Path, last_text: str
) -> None:
    database = tmp_path / "corpus.sqlite3"
    container = make_container(database)
    untouched = document("other", "Alpha background evidence.")
    retained = container.ingestion_pipeline.ingest_documents([untouched])
    container.ingestion_pipeline.ingest_documents(
        [document(), document("paper:10.1/'\u03b2'", "Prior Evidence")]
    )
    last = document(text=last_text)
    other_revision = document("paper:10.1/'\u03b2'", "Gamma Background")
    batch = [document(text="Discarded " * 200), other_revision, last]
    expected = [*TextChunker().chunk(last), *TextChunker().chunk(other_revision)]

    assert container.ingestion_pipeline.ingest_documents(batch) == expected
    await assert_current(container, [*expected, *retained])
    assert container.graph_store.chunks_for_entities(["Discarded", "Prior"]) == []
    before = graph_rows(database)
    assert container.ingestion_pipeline.ingest_documents(batch) == expected
    assert container.ingestion_pipeline.ingest_documents([]) == []
    assert graph_rows(database) == before
    await assert_current(make_container(database), [*expected, *retained])


async def test_revision_removes_only_its_graph_bridge_and_preserves_scope(
    tmp_path: Path,
) -> None:
    database = tmp_path / "corpus.sqlite3"
    container = make_container(database)
    originals = [
        document("head", "Alpha"),
        document("bridge", "Alpha Beta"),
        document("old-tail", "Beta"),
        document("new-tail", "Gamma"),
        document("excluded", "Alpha Beta"),
    ]
    initial = container.ingestion_pipeline.ingest_documents(originals)
    old_bridge = next(chunk for chunk in initial if chunk.document_id == "bridge")
    before = graph_rows(database)
    [current] = container.ingestion_pipeline.ingest_documents([document("bridge", "Alpha Gamma")])
    assert current.chunk_id != old_bridge.chunk_id
    scope = ["head", "bridge", "old-tail", "new-tail"]

    for runtime in (container, make_container(database)):
        store = runtime.graph_store
        assert store.neighbours(["Alpha"], document_ids=scope) == ["Gamma"]
        assert store.neighbours(["Beta"], document_ids=scope) == []
        assert store.neighbours(["Alpha"], document_ids=["excluded"]) == ["Beta"]
        hits = await MultiHopRetriever(store).retrieve(
            "Alpha", ["Alpha"], depth=2, limit=20, document_ids=scope
        )
        assert {hit.chunk.document_id: hit.score for hit in hits} == {
            "head": 1.0,
            "bridge": 1.0,
            "new-tail": 0.5,
        }
        assert {
            hit.chunk.document_id for hit in await GraphRetriever(store).retrieve(["Beta"])
        } == {
            "old-tail",
            "excluded",
        }
    for table, rows in graph_rows(database).items():
        assert all(row[0] != old_bridge.chunk_id for row in rows)
        assert [row for row in rows if row[0] != current.chunk_id] == [
            row for row in before[table] if row[0] != old_bridge.chunk_id
        ]


def test_low_level_document_store_chunk_upsert_stays_incremental(tmp_path: Path) -> None:
    container = make_container(tmp_path / "corpus.sqlite3")
    [original] = container.ingestion_pipeline.ingest_documents([document()])
    [extra] = TextChunker().chunk(document(text=NEW_TEXT))
    store = container.document_store
    store.add_documents([], [extra])
    assert store.list_chunks() == sorted([original, extra], key=lambda c: c.chunk_id)
    changed = original.model_copy(update={"text": "Exact chunk update"})
    store.add_documents([], [changed])
    store.add_documents([], [])
    assert store.list_chunks() == sorted([changed, extra], key=lambda c: c.chunk_id)


@pytest.mark.parametrize("stage", ["chunking", "embedding", "extraction", "edges"])
@pytest.mark.parametrize("separate_graph", [False, True], ids=["shared-db", "separate-db"])
async def test_preparation_failure_leaves_entire_batch_and_live_indexes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, separate_graph: bool
) -> None:
    database = tmp_path / "corpus.sqlite3"
    graph_database = database.with_suffix(".graph.sqlite3") if separate_graph else database
    container = make_container(database, separate_graph=separate_graph)
    initial = container.ingestion_pipeline.ingest_documents(
        [document(), document("other", "Alpha background")]
    )
    before = {path: dump(path) for path in (database, graph_database)}
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected preparation failure")

    chunker = TextChunker()
    chunk = chunker.chunk
    embed = HashEmbeddingModel.embed
    extract = FixtureExtractor.extract

    def chunk_then_fail(value: Document) -> list[Chunk]:
        fail_second()
        return chunk(value)

    def embed_then_fail(self: HashEmbeddingModel, text: str) -> list[float]:
        fail_second()
        return embed(self, text)

    def extract_then_fail(self: FixtureExtractor, text: str) -> list[Entity]:
        fail_second()
        return extract(self, text)

    def edge_failure(*args: object, **kwargs: object) -> NoReturn:
        raise ValueError("injected preparation failure")

    with monkeypatch.context() as patch:
        if stage == "chunking":
            patch.setattr(chunker, "chunk", chunk_then_fail)
            container.ingestion_pipeline = IngestionPipeline(
                container.document_store,
                container.hybrid_retriever,
                container.graph_builder,
                chunker=chunker,
            )
        elif stage == "embedding":
            patch.setattr(HashEmbeddingModel, "embed", embed_then_fail)
        elif stage == "extraction":
            patch.setattr(FixtureExtractor, "extract", extract_then_fail)
        else:
            patch.setattr("retrieval.graph.EntityEdge", edge_failure)
        with pytest.raises(ValueError, match="injected preparation failure"):
            container.ingestion_pipeline.ingest_documents(
                [document(text=NEW_TEXT), document("new-document", "Theta Zeta")]
            )
    assert {path: dump(path) for path in before} == before
    await assert_current(container, initial)


SQL_FAILURES = [
    "CREATE TRIGGER fail_insert BEFORE INSERT ON documents "
    "WHEN NEW.document_id = 'new-document' "
    "BEGIN SELECT RAISE(ABORT, 'injected SQL failure'); END",
    "CREATE TRIGGER fail_insert BEFORE INSERT ON chunks "
    "WHEN NEW.document_id = 'new-document' "
    "BEGIN SELECT RAISE(ABORT, 'injected SQL failure'); END",
    "CREATE TRIGGER fail_insert BEFORE INSERT ON graph_chunks "
    "WHEN NEW.document_id = 'new-document' "
    "BEGIN SELECT RAISE(ABORT, 'injected SQL failure'); END",
    "CREATE TRIGGER fail_insert BEFORE INSERT ON entity_mentions "
    "WHEN NEW.entity_name = 'Zeta' "
    "BEGIN SELECT RAISE(ABORT, 'injected SQL failure'); END",
    "CREATE TRIGGER fail_insert BEFORE INSERT ON entity_edges "
    "WHEN NEW.target_name = 'Zeta' "
    "BEGIN SELECT RAISE(ABORT, 'injected SQL failure'); END",
]


@pytest.mark.parametrize(
    "trigger", SQL_FAILURES, ids=["document", "chunk", "graph", "mention", "edge"]
)
async def test_sql_failure_rolls_back_documents_graph_and_live_indexes(
    tmp_path: Path, trigger: str
) -> None:
    database = tmp_path / "corpus.sqlite3"
    container = make_container(database)
    initial = container.ingestion_pipeline.ingest_documents(
        [document(), document("other", "Alpha background")]
    )
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(trigger)
    before = dump(database)
    with pytest.raises(sqlite3.IntegrityError, match="injected SQL failure"):
        container.ingestion_pipeline.ingest_documents(
            [document(text=NEW_TEXT), document("new-document", "Theta Zeta")]
        )
    assert dump(database) == before
    await assert_current(container, initial)
    await assert_current(make_container(database), initial)


async def test_separately_wired_graph_sql_failure_rolls_back_both_stores(
    tmp_path: Path,
) -> None:
    database = tmp_path / "corpus.sqlite3"
    graph_database = database.with_suffix(".graph.sqlite3")
    container = make_container(database, separate_graph=True)
    initial = container.ingestion_pipeline.ingest_documents([document()])
    with closing(sqlite3.connect(graph_database)) as connection, connection:
        connection.execute(SQL_FAILURES[-1])
    before = dump(database), dump(graph_database)
    with pytest.raises(sqlite3.IntegrityError, match="injected SQL failure"):
        container.ingestion_pipeline.ingest_documents(
            [document(text=NEW_TEXT), document("new-document", "Theta Zeta")]
        )
    assert (dump(database), dump(graph_database)) == before
    await assert_current(container, initial)


async def test_commit_failure_keeps_previous_corpus_and_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "corpus.sqlite3"
    container = make_container(database)
    initial = container.ingestion_pipeline.ingest_documents([document()])
    before = dump(database)
    connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    class FailedCommit(sqlite3.Connection):
        def commit(self) -> None:
            raise sqlite3.OperationalError("injected commit failure")

    def failing_connect(path: str) -> sqlite3.Connection:
        connection = connect(path, factory=FailedCommit)
        opened.append(connection)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", failing_connect)
        with pytest.raises(sqlite3.OperationalError, match="injected commit failure"):
            container.ingestion_pipeline.ingest_documents([document(text=NEW_TEXT)])
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    assert dump(database) == before
    await assert_current(container, initial)


@pytest.mark.parametrize(
    "writer", ["dense", "bm25", "hybrid", "graph-builder", "documents", "graph-store"]
)
async def test_legacy_writer_override_requires_explicit_replacement_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    database = tmp_path / "corpus.sqlite3"
    container = make_container(database)
    initial = container.ingestion_pipeline.ingest_documents([document()])

    def reject(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("synthetic custom index rejection")

    class CustomDense(DenseRetriever):
        add_chunks = reject

    class CustomSparse(BM25Retriever):
        add_chunks = reject

    class CustomHybrid(HybridRetriever):
        add_chunks = reject

    class CustomBuilder(GraphRAGBuilder):
        index_chunks = reject

    class CustomDocuments(SQLiteDocumentStore):
        add_documents = reject

    class CustomGraph(SQLiteGraphStore):
        replace_chunk = reject

    hybrid = container.hybrid_retriever
    if writer == "dense":
        dense = CustomDense()
        DenseRetriever.add_chunks(dense, initial)
        hybrid._dense_retriever = dense
    elif writer == "bm25":
        sparse = CustomSparse()
        BM25Retriever.add_chunks(sparse, initial)
        hybrid._sparse_retriever = sparse
    elif writer == "hybrid":
        container.hybrid_retriever = CustomHybrid(
            hybrid._dense_retriever, hybrid._sparse_retriever, HyDEExpander()
        )
    elif writer == "graph-builder":
        container.graph_builder = CustomBuilder(container.graph_store)
    elif writer == "documents":
        container.document_store = CustomDocuments(database)
    else:
        container.graph_store = CustomGraph(database)
        container.graph_builder = GraphRAGBuilder(container.graph_store)
    pipeline = IngestionPipeline(
        container.document_store, container.hybrid_retriever, container.graph_builder
    )
    before = dump(database)
    connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def track_connect(path: str) -> sqlite3.Connection:
        connection = connect(path)
        opened.append(connection)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", track_connect)
        with pytest.raises(
            TypeError, match=r"replacing_documents|replace_documents|prepare_chunks"
        ):
            pipeline.ingest_documents([document(text=NEW_TEXT)])
    assert opened == []
    assert dump(database) == before
    await assert_current(container, initial)


async def test_separate_files_expose_commit_boundary_and_explicit_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "corpus.sqlite3"
    graph_database = database.with_suffix(".graph.sqlite3")
    container = make_container(database, separate_graph=True)
    initial = container.ingestion_pipeline.ingest_documents([document()])
    expected = TextChunker().chunk(document(text=NEW_TEXT))
    before = dump(database), dump(graph_database)
    connect = sqlite3.connect

    class FailedCommit(sqlite3.Connection):
        def commit(self) -> None:
            raise sqlite3.OperationalError("injected document commit failure")

    def fail_document_commit(path: str) -> sqlite3.Connection:
        factory = FailedCommit if Path(path) == database else sqlite3.Connection
        return connect(path, factory=factory)

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", fail_document_commit)
        with pytest.raises(sqlite3.OperationalError, match="injected document commit failure"):
            container.ingestion_pipeline.ingest_documents([document(text=NEW_TEXT)])
    assert dump(database) == before[0]
    assert dump(graph_database) != before[1]
    await assert_current(container, initial)
    assert container.graph_store.chunks_for_entities(["Gamma"]) == expected
    assert container.graph_store.chunks_for_entities(["Alpha"]) == []

    assert container.ingestion_pipeline.ingest_documents([document(text=NEW_TEXT)]) == expected
    await assert_current(container, expected)
    assert container.graph_store.chunks_for_entities(["Gamma"]) == expected


async def test_explicit_index_hooks_preserve_configured_models_and_untouched_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks: list[str] = []

    def reject_legacy(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("The custom writer explicitly opted in to a replacement hook.")

    class CustomDense(DenseRetriever):
        add_chunks = reject_legacy

        @contextmanager
        def replacing_documents(
            self, document_ids: frozenset[str], chunks: list[Chunk]
        ) -> Iterator[None]:
            hooks.append("dense")
            with super().replacing_documents(document_ids, chunks):
                yield

    class CustomSparse(BM25Retriever):
        add_chunks = reject_legacy

        @contextmanager
        def replacing_documents(
            self, document_ids: frozenset[str], chunks: list[Chunk]
        ) -> Iterator[None]:
            hooks.append("bm25")
            with super().replacing_documents(document_ids, chunks):
                yield

    class CustomHybrid(HybridRetriever):
        add_chunks = reject_legacy

        @contextmanager
        def replacing_documents(
            self, document_ids: frozenset[str], chunks: list[Chunk]
        ) -> Iterator[None]:
            hooks.append("hybrid")
            with super().replacing_documents(document_ids, chunks):
                yield

    container = make_container(tmp_path / "corpus.sqlite3")
    embedder = HashEmbeddingModel(dimensions=7)
    hybrid = CustomHybrid(CustomDense(embedder), CustomSparse(k1=0.6, b=0.2), HyDEExpander())
    pipeline = IngestionPipeline(container.document_store, hybrid, container.graph_builder)
    initial = pipeline.ingest_documents([document(), document("other", "background Gamma")])
    embedded: list[str] = []
    embed = embedder.embed

    def record(text: str) -> list[float]:
        embedded.append(text)
        return embed(text)

    monkeypatch.setattr(embedder, "embed", record)
    current = pipeline.ingest_documents([document(text=NEW_TEXT)])
    assert embedded == [NEW_TEXT]
    assert hooks == ["hybrid", "dense", "bm25"] * 2
    expected = [*current, *[chunk for chunk in initial if chunk.document_id == "other"]]
    fresh = HybridRetriever(
        DenseRetriever(HashEmbeddingModel(dimensions=7)),
        BM25Retriever(k1=0.6, b=0.2),
        HyDEExpander(),
    )
    fresh.add_chunks(expected)
    for actual, reference in zip(components(hybrid), components(fresh), strict=True):
        for scope in (None, ["same-paper"], ["other"]):
            assert await actual.retrieve("Gamma", document_ids=scope) == await reference.retrieve(
                "Gamma", document_ids=scope
            )


async def test_pipeline_snapshots_the_complete_document_batch_before_chunking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = make_container(tmp_path / "corpus.sqlite3")
    batch = [document(), document("other", "Gamma background")]
    expected = [chunk for value in batch for chunk in TextChunker().chunk(value)]
    chunker = TextChunker()
    chunk = chunker.chunk

    def chunk_and_mutate(value: Document) -> list[Chunk]:
        if batch:
            batch[-1].text = "Unindexed caller edit"
            batch[-1].metadata.clear()
            batch.clear()
        return chunk(value)

    monkeypatch.setattr(chunker, "chunk", chunk_and_mutate)
    pipeline = IngestionPipeline(
        container.document_store, container.hybrid_retriever, container.graph_builder, chunker
    )
    assert pipeline.ingest_documents(batch) == expected
    await assert_current(container, expected)


@pytest.mark.parametrize("invalid", ["document-owner", "duplicate-chunk", "foreign-chunk-id"])
async def test_invalid_chunker_output_cannot_replace_unrelated_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    database = tmp_path / "corpus.sqlite3"
    container = make_container(database)
    initial = container.ingestion_pipeline.ingest_documents(
        [document(), document("other", "Gamma background")]
    )
    unrelated = next(chunk for chunk in initial if chunk.document_id == "other")
    chunker = TextChunker()
    [candidate] = chunker.chunk(document(text=NEW_TEXT))
    if invalid == "document-owner":
        candidate.document_id = unrelated.document_id
    elif invalid == "foreign-chunk-id":
        candidate.chunk_id = unrelated.chunk_id
    output = [candidate, candidate] if invalid == "duplicate-chunk" else [candidate]
    monkeypatch.setattr(chunker, "chunk", lambda value: output)
    pipeline = IngestionPipeline(
        container.document_store, container.hybrid_retriever, container.graph_builder, chunker
    )
    before = dump(database)
    with pytest.raises(ValueError, match=r"chunk|document"):
        pipeline.ingest_documents([document(text=NEW_TEXT)])
    assert dump(database) == before
    await assert_current(container, initial)


async def test_returned_replacement_chunks_do_not_leak_mutable_index_state(
    tmp_path: Path,
) -> None:
    container = make_container(tmp_path / "corpus.sqlite3")
    container.ingestion_pipeline.ingest_documents([document()])
    supplied = document(text=NEW_TEXT)
    returned = container.ingestion_pipeline.ingest_documents([supplied])
    expected = TextChunker().chunk(document(text=NEW_TEXT))
    supplied.metadata.clear()
    supplied.text = "Unindexed caller edit"
    returned[0].chunk_id = "caller-owned"
    returned[0].document_id = "missing"
    returned[0].metadata.clear()
    returned[0].text = "Unindexed result edit"
    returned.clear()
    await assert_current(container, expected)
    assert container.graph_store.chunks_for_entities(["Gamma"]) == expected


def test_concurrent_replacements_finish_with_one_complete_current_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "corpus.sqlite3"
    container = make_container(database)
    retained = container.ingestion_pipeline.ingest_documents([document("other", "background")])
    container.ingestion_pipeline.ingest_documents([document()])
    rendezvous = Barrier(2)
    chunker = TextChunker()
    chunk = chunker.chunk

    def together(value: Document) -> list[Chunk]:
        result = chunk(value)
        rendezvous.wait(timeout=10)
        return result

    monkeypatch.setattr(chunker, "chunk", together)
    pipeline = IngestionPipeline(
        container.document_store, container.hybrid_retriever, container.graph_builder, chunker
    )
    revisions = [document(text=NEW_TEXT), document(text="Theta Zeta " * 200)]
    with ThreadPoolExecutor(max_workers=2) as workers:
        batches = list(workers.map(lambda value: pipeline.ingest_documents([value]), revisions))
    current = container.document_store.list_chunks()
    assert any(
        current == sorted([*batch, *retained], key=lambda c: c.chunk_id) for batch in batches
    )
    asyncio.run(assert_current(container, current))
    asyncio.run(assert_current(make_container(database), current))
    assert {row[0] for row in graph_rows(database)["chunks"]} == {
        chunk.chunk_id for chunk in current
    }


@pytest.mark.parametrize("revision_text", [NEW_TEXT, ""])
def test_current_readers_change_but_frozen_exports_reviews_and_annotations_do_not(
    tmp_path: Path, revision_text: str
) -> None:
    database = tmp_path / "corpus.sqlite3"
    app = create_app(offline_settings(database))
    container: AppContainer = app.state.container
    [original] = container.ingestion_pipeline.ingest_documents([document()])
    with TestClient(app) as client:
        created = client.post(
            "/collections", json={"name": "Synthetic revisions", "document_ids": ["same-paper"]}
        )
        assert created.status_code == 201, created.text
        collection = created.json()
        completed = client.post(
            "/query",
            json={"query": "Alpha methodology", "collection_id": collection["collection_id"]},
        )
        assert completed.status_code == 200
        result = completed.json()["result"]
        assert result["state"] == "DONE", result
        run = f"/runs/{result['run_id']}"
        exports = {
            fmt: client.get(f"{run}/export", params={"format": fmt}).content
            for fmt in ("json", "markdown", "html")
        }
        source = json.loads(exports["json"])["snapshot"]["sources"][0]
        assert source["chunk"] == original.model_dump()
        review = client.post(
            f"{run}/reviews",
            json={
                "review_id": str(uuid4()),
                "decision": "needs_revision",
                "comment": "Human note about the original synthetic passage.",
                "cited_chunk_ids": [original.chunk_id],
            },
        )
        assert review.status_code == 201, review.text
        annotation = client.post(
            f"{run}/annotations",
            json={
                "annotation_id": str(uuid4()),
                "document_id": original.document_id,
                "chunk_id": original.chunk_id,
                "source_text_sha256": source["text_sha256"],
                "start": 0,
                "end": 5,
                "quote": "Alpha",
                "note": "Original wording, not a current-corpus reference.",
            },
        )
        assert annotation.status_code == 201, annotation.text
        saved = {
            path: client.get(path).content
            for path in (f"{run}/reviews", f"{run}/annotations", f"{run}/events", "/runs")
        }
        expected = container.ingestion_pipeline.ingest_documents([document(text=revision_text)])
        for runtime in (container, make_container(database)):
            app.state.container = runtime
            catalog = client.get("/documents").json()["documents"]
            assert [(row["document_id"], row["chunk_count"]) for row in catalog] == [
                ("same-paper", len(expected))
            ]
            page = client.get("/documents/same-paper/chunks")
            assert page.status_code == 200, page.text
            assert [row["chunk_id"] for row in page.json()["chunks"]] == [
                chunk.chunk_id for chunk in expected
            ]
            explorer = client.get("/explore/document", params={"document_id": "same-paper"})
            assert explorer.status_code == 200
            assert OLD_TEXT not in explorer.text
            if revision_text:
                assert revision_text in explorer.text
            preview = client.post(
                "/retrieve",
                json={"query": "Gamma methodology", "collection_id": collection["collection_id"]},
            )
            assert preview.status_code == 200, preview.text
            assert [row["chunk"] for row in preview.json()["sources"]] == [
                chunk.model_dump() for chunk in expected
            ]
            assert client.get(f"/collections/{collection['collection_id']}").json() == collection
            for path, content in saved.items():
                assert client.get(path).content == content
            for fmt, content in exports.items():
                assert client.get(f"{run}/export", params={"format": fmt}).content == content
            drift = client.get(f"{run}/corpus-drift")
            assert drift.status_code == 200, drift.text
            assert drift.json()["counts"] == {
                "total": 1,
                "unchanged": 0,
                "changed": 0,
                "missing": 1,
            }
