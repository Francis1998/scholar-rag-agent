"""Core stores release real SQLite connections without relying on garbage collection."""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path

import pytest

from retrieval.models import Chunk, Document, Entity, EntityEdge
from storage.document_store import SQLiteDocumentStore
from storage.event_log import SQLiteEventLog
from storage.graph_store import SQLiteGraphStore

StoreFactory = type[SQLiteEventLog] | type[SQLiteDocumentStore] | type[SQLiteGraphStore]


class TrackedConnection(sqlite3.Connection):
    transaction_at_close: bool | None = None
    fail_commit = False

    def close(self) -> None:
        self.transaction_at_close = self.in_transaction
        super().close()

    def commit(self) -> None:
        if self.fail_commit:
            raise sqlite3.OperationalError("injected commit failure")
        super().commit()


@dataclass
class ConnectionTracker:
    connections: list[TrackedConnection] = field(default_factory=list)
    read_only: bool = False
    fail_commit: bool = False

    def assert_closed_since(self, start: int) -> None:
        connections = self.connections[start:]
        assert len(connections) == 1
        for connection in connections:
            assert connection.transaction_at_close is False
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> Iterator[ConnectionTracker]:
    original_connect = sqlite3.connect
    tracked = ConnectionTracker()

    def connect(database: str | Path) -> TrackedConnection:
        connection = original_connect(database, factory=TrackedConnection)
        assert isinstance(connection, TrackedConnection)
        connection.fail_commit = tracked.fail_commit
        if tracked.read_only:
            connection.execute("PRAGMA query_only = ON")
        tracked.connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        # Strong references make a missing close observable even on refcounting runtimes.
        yield tracked
    finally:
        for connection in tracked.connections:
            sqlite3.Connection.close(connection)


@dataclass
class Stores:
    events: SQLiteEventLog
    documents: SQLiteDocumentStore
    graph: SQLiteGraphStore
    document: Document
    chunk: Chunk

    def operations(self) -> dict[str, Callable[[], object]]:
        return {
            "append_event": lambda: self.events.append_event("agent", "run", "test", {"n": 1}),
            "list_events": lambda: self.events.list_events("run"),
            "add_documents": lambda: self.documents.add_documents([self.document], [self.chunk]),
            "list_chunks": self.documents.list_chunks,
            "add_mentions": lambda: self.graph.add_mentions(self.chunk, [Entity(name="Alpha")]),
            "add_edges": lambda: self.graph.add_edges(
                [EntityEdge(source="Alpha", target="Beta", chunk_id="chunk")]
            ),
            "chunks_for_entities": lambda: self.graph.chunks_for_entities(["Alpha"]),
            "neighbours": lambda: self.graph.neighbours(["Alpha"]),
        }


def seed_stores(database: Path) -> Stores:
    document = Document(document_id="doc", title="Title", text="Alpha Beta", source="fixture")
    chunk = Chunk(chunk_id="chunk", **document.model_dump())
    stores = Stores(
        SQLiteEventLog(database),
        SQLiteDocumentStore(database),
        SQLiteGraphStore(database),
        document,
        chunk,
    )
    stores.events.append_event("agent", "run", "test", {"n": 0})
    stores.documents.add_documents([document], [chunk])
    stores.graph.add_mentions(chunk, [Entity(name="Alpha")])
    stores.graph.add_edges([EntityEdge(source="Alpha", target="Beta", chunk_id="chunk")])
    return stores


@pytest.mark.parametrize("factory", [SQLiteEventLog, SQLiteDocumentStore, SQLiteGraphStore])
@pytest.mark.parametrize("read_only", [False, True])
def test_initialization_closes_connections(
    tmp_path: Path, tracker: ConnectionTracker, factory: StoreFactory, read_only: bool
) -> None:
    tracker.read_only = read_only
    if read_only:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            factory(tmp_path / "store.sqlite3")
    else:
        factory(tmp_path / "store.sqlite3")
    tracker.assert_closed_since(0)


@pytest.mark.parametrize(
    "operation",
    [
        "append_event",
        "list_events",
        "add_documents",
        "list_chunks",
        "add_mentions",
        "add_edges",
        "chunks_for_entities",
        "neighbours",
    ],
)
def test_successful_operations_close_connections(
    tmp_path: Path, tracker: ConnectionTracker, operation: str
) -> None:
    stores = seed_stores(tmp_path / "store.sqlite3")
    start = len(tracker.connections)
    result = stores.operations()[operation]()
    tracker.assert_closed_since(start)
    if operation in {"list_events", "list_chunks", "chunks_for_entities", "neighbours"}:
        assert result


@pytest.mark.parametrize(
    "operation", ["append_event", "add_documents", "add_mentions", "add_edges"]
)
@pytest.mark.parametrize("failure", ["readonly", "commit"])
def test_failed_writes_close_connections_and_preserve_data(
    tmp_path: Path, tracker: ConnectionTracker, operation: str, failure: str
) -> None:
    database = tmp_path / "store.sqlite3"
    stores = seed_stores(database)
    with closing(sqlite3.connect(database)) as connection:
        before = list(connection.iterdump())
    tracker.read_only = failure == "readonly"
    tracker.fail_commit = failure == "commit"
    start = len(tracker.connections)
    with pytest.raises(sqlite3.OperationalError, match=failure):
        stores.operations()[operation]()
    tracker.assert_closed_since(start)
    with closing(sqlite3.connect(database)) as connection:
        assert list(connection.iterdump()) == before


@pytest.mark.parametrize(
    ("operation", "drop_sql"),
    [
        ("list_events", "DROP TABLE agent_events"),
        ("list_chunks", "DROP TABLE chunks"),
        ("chunks_for_entities", "DROP TABLE graph_chunks"),
        ("neighbours", "DROP TABLE entity_edges"),
    ],
)
def test_failed_reads_close_connections(
    tmp_path: Path, tracker: ConnectionTracker, operation: str, drop_sql: str
) -> None:
    database = tmp_path / "store.sqlite3"
    stores = seed_stores(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(drop_sql)
        connection.commit()
    start = len(tracker.connections)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        stores.operations()[operation]()
    tracker.assert_closed_since(start)


@pytest.mark.parametrize(
    ("operation", "corrupt_sql"),
    [
        ("list_events", "UPDATE agent_events SET payload = 'invalid json'"),
        ("list_chunks", "UPDATE chunks SET metadata = 'invalid json'"),
        ("chunks_for_entities", "UPDATE graph_chunks SET metadata = 'invalid json'"),
    ],
)
def test_decode_failures_do_not_leave_connections_open(
    tmp_path: Path, tracker: ConnectionTracker, operation: str, corrupt_sql: str
) -> None:
    database = tmp_path / "store.sqlite3"
    stores = seed_stores(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(corrupt_sql)
        connection.commit()
    start = len(tracker.connections)
    with pytest.raises(JSONDecodeError):
        stores.operations()[operation]()
    tracker.assert_closed_since(start)


def test_serialization_failure_closes_event_connection(
    tmp_path: Path, tracker: ConnectionTracker
) -> None:
    events = SQLiteEventLog(tmp_path / "store.sqlite3")
    start = len(tracker.connections)
    with pytest.raises(TypeError, match="JSON serializable"):
        events.append_event("agent", "run", "test", {"invalid": object()})
    tracker.assert_closed_since(start)
    assert events.list_events() == []


@pytest.mark.parametrize(
    ("operation", "trigger_sql"),
    [
        (
            "add_documents",
            "CREATE TRIGGER reject_chunk BEFORE INSERT ON chunks WHEN NEW.chunk_id = 'bad' "
            "BEGIN SELECT RAISE(ABORT, 'injected constraint failure'); END",
        ),
        (
            "add_mentions",
            "CREATE TRIGGER reject_mention BEFORE INSERT ON entity_mentions "
            "WHEN NEW.entity_name = 'Denied' "
            "BEGIN SELECT RAISE(ABORT, 'injected constraint failure'); END",
        ),
        (
            "add_edges",
            "CREATE TRIGGER reject_edge BEFORE INSERT ON entity_edges "
            "WHEN NEW.target_name = 'Denied' "
            "BEGIN SELECT RAISE(ABORT, 'injected constraint failure'); END",
        ),
    ],
)
def test_partial_writes_roll_back_before_closing(
    tmp_path: Path, tracker: ConnectionTracker, operation: str, trigger_sql: str
) -> None:
    database = tmp_path / "store.sqlite3"
    stores = seed_stores(database)
    new_document = stores.document.model_copy(update={"document_id": "new"})
    new_chunk = stores.chunk.model_copy(update={"chunk_id": "new", "document_id": "new"})
    operations: dict[str, Callable[[], None]] = {
        "add_documents": lambda: stores.documents.add_documents(
            [new_document], [new_chunk, new_chunk.model_copy(update={"chunk_id": "bad"})]
        ),
        "add_mentions": lambda: stores.graph.add_mentions(
            new_chunk, [Entity(name="Allowed"), Entity(name="Denied")]
        ),
        "add_edges": lambda: stores.graph.add_edges(
            [
                EntityEdge(source="Alpha", target="Allowed", chunk_id="chunk"),
                EntityEdge(source="Alpha", target="Denied", chunk_id="chunk"),
            ]
        ),
    }
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(trigger_sql)
        connection.commit()
        before = list(connection.iterdump())
    start = len(tracker.connections)
    with pytest.raises(sqlite3.IntegrityError, match="injected constraint failure"):
        operations[operation]()
    tracker.assert_closed_since(start)
    with closing(sqlite3.connect(database)) as connection:
        assert list(connection.iterdump()) == before
