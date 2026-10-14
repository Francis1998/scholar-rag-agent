"""Bounded, exact, read-only duplicate candidates for persisted collections."""

import json
import socket
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.duplicate_paper_cluster import DuplicatePaperClusterer, PaperRef
from retrieval.models import Chunk, Document
from storage.document_store import SQLiteDocumentStore
from storage.paper_collections import CollectionError, PaperCollection, SQLitePaperCollections

if TYPE_CHECKING:
    from storage.collection_duplicates import CollectionDuplicateReport


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Duplicate inspection must not retrieve, generate, write, or use network.")


@dataclass
class DuplicateAPI:
    client: TestClient
    container: AppContainer
    database: Path
    collection: PaperCollection

    @property
    def route(self) -> str:
        return f"/collections/{self.collection.collection_id}/duplicates"

    def get(self, **params: object) -> httpx.Response:
        return self.client.get(
            self.route, params={"collection_revision": self.collection.revision, **params}
        )

    def report(self, **options: object) -> "CollectionDuplicateReport":
        return self.container.collection_duplicates.inspect(
            self.collection.collection_id,
            collection_revision=self.collection.revision,
            **options,
        )


@pytest.fixture
def duplicate_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[DuplicateAPI]:
    database = tmp_path / "duplicates ?# \u7814\u7a76.sqlite3"
    app = create_app(offline_settings(database))
    container = app.state.container
    for target, method in (
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
    ):
        monkeypatch.setattr(target, method, denied)
    documents = [
        Document(
            document_id=identifier,
            title=title,
            source="synthetic:duplicates",
            text="GraphRAG private-body-sentinel",
            metadata={**metadata, "private": "private-metadata-sentinel"},
        )
        for identifier, title, metadata in (
            ("b-doi", "Unrelated label B", {"paper_doi": "https://doi.org/10.0000/SYNTHETIC."}),
            ("a-doi", "Different label A", {"doi": " DOI:10.0000/synthetic "}),
            ("d-title", "Synthetic GraphRAG method notes", {}),
            ("c-title", "Synthetic GraphRAG method note", {"work_doi": "10.0000/other"}),
            ("z-alone", "Zoology", {}),
            ("blank-1", "", {}),
            ("blank-2", " \t ", {}),
            ("outside", "Must not inspect me", {"doi": "10.0000/synthetic"}),
        )
    ]
    chunks = [Chunk(chunk_id=f"chunk-{d.document_id}", **d.model_dump()) for d in documents]
    container.document_store.add_documents(documents, chunks)
    container.hybrid_retriever.add_chunks(chunks)
    collection = container.paper_collections.create(
        name="Synthetic candidate review", document_ids=[d.document_id for d in documents[:-1]]
    )
    with TestClient(app) as client:
        yield DuplicateAPI(client, container, database, collection)


def assert_private(response: httpx.Response) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def assert_error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert_private(response)
    assert set(response.json()) == {"detail"}
    assert set(response.json()["detail"]) == {"code", "message"}
    assert response.json()["detail"]["code"] == code
    assert "content-disposition" not in response.headers


def mutate(api: DuplicateAPI, sql: str, parameters: tuple[object, ...] = ()) -> None:
    with closing(sqlite3.connect(api.database)) as connection, connection:
        connection.execute(sql, parameters)


def test_api_capability_returns_complete_candidates(duplicate_api: DuplicateAPI) -> None:
    response = duplicate_api.get()
    assert response.status_code == 200, response.text
    assert_private(response)
    result = response.json()
    assert result["schema_version"] == "1.0"
    assert result["collection_id"] == duplicate_api.collection.collection_id
    assert result["collection_name"] == duplicate_api.collection.name
    assert result["collection_revision"] == 1
    assert result["title_threshold"] == 0.92
    assert result["inspected_count"] == 7
    assert result["document_ids"] == sorted(duplicate_api.collection.document_ids)
    assert [[p["document_id"] for p in g["members"]] for g in result["groups"]] == [
        ["a-doi", "b-doi"],
        ["c-title", "d-title"],
    ]
    assert result["groups"][0]["match_reasons"] == ["doi"]
    assert result["groups"][1]["match_reasons"] == ["title:0.984"]
    assert [p["document_id"] for p in result["singletons"]] == ["blank-1", "blank-2", "z-alone"]
    assert result["groups"][0]["members"][0]["doi"] == " DOI:10.0000/synthetic "
    assert result["groups"][0]["members"][1]["doi_field"] == "paper_doi"
    assert "private-body-sentinel" not in response.text
    assert "private-metadata-sentinel" not in response.text
    assert "outside" not in response.text
    assert "transitive" in " ".join(result["warnings"])
    assert "not proof" in " ".join(result["warnings"])


def test_python_capability_is_integrated_and_immutable(duplicate_api: DuplicateAPI) -> None:
    report = duplicate_api.report()
    assert report.to_json().encode() == duplicate_api.get().content
    assert report.to_markdown().encode() == duplicate_api.get(format="markdown").content
    assert isinstance(report.groups, tuple)
    assert isinstance(report.groups[0].members, tuple)
    assert isinstance(report.groups[0].match_reasons, tuple)
    assert isinstance(report.singletons, tuple)
    assert isinstance(report.document_ids, tuple)
    assert isinstance(report.warnings, tuple)
    with pytest.raises(ValidationError):
        report.collection_revision = 99
    with pytest.raises(ValidationError):
        report.groups[0].members[0].title = "Rewritten"


def test_reuses_the_existing_clusterer_without_losing_raw_metadata(
    duplicate_api: DuplicateAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = DuplicatePaperClusterer.cluster

    def cluster(self: DuplicatePaperClusterer, papers: list[PaperRef]) -> object:
        calls.append(papers)
        return original(self, papers)

    monkeypatch.setattr(DuplicatePaperClusterer, "cluster", cluster)
    report = duplicate_api.report(title_threshold=1.0)
    assert len(calls) == 1
    assert [p.document_id for p in calls[0]] == sorted(duplicate_api.collection.document_ids)
    assert calls[0][0].doi == " DOI:10.0000/synthetic "
    assert len(report.groups) == 1 and len(report.singletons) == 5


def test_membership_isolation_includes_corrupt_unselected_rows(duplicate_api: DuplicateAPI) -> None:
    before = duplicate_api.get().content
    mutate(
        duplicate_api,
        "UPDATE documents SET title = ?, metadata = ? WHERE document_id = 'outside'",
        ("x" * 100_000, "not-json" * 100_000),
    )
    assert duplicate_api.get().content == before
    assert duplicate_api.report().inspected_count == 7


def test_case_sensitive_identities_are_not_duplicate_evidence(duplicate_api: DuplicateAPI) -> None:
    api = duplicate_api
    identifiers = ["Paper", "paper", "paper!", "nul\x00id"]
    api.container.document_store.add_documents(
        [Document(document_id=i, title="", source="", text="private") for i in identifiers], []
    )
    api.collection = api.container.paper_collections.create(
        name="Exact raw identities", document_ids=identifiers
    )
    report = api.report(title_threshold=0.0)
    assert report.groups == ()
    assert [p.document_id for p in report.singletons] == sorted(identifiers)
    assert json.loads(report.to_json())["document_ids"] == sorted(identifiers)


def test_title_links_can_join_conflicting_dois_transitively(duplicate_api: DuplicateAPI) -> None:
    api = duplicate_api
    records = [
        ("a", "Synthetic research note", {"doi": "10.0000/a"}),
        ("b", "Totally unrelated", {"doi": "10.0000/a"}),
        ("c", "Synthetic research notes", {"doi": "10.0000/conflict"}),
    ]
    api.container.document_store.add_documents(
        [
            Document(document_id=i, title=t, source="", text="private", metadata=m)
            for i, t, m in records
        ],
        [],
    )
    api.collection = api.container.paper_collections.create(
        name="Transitive candidates", document_ids=["c", "b", "a"]
    )
    report = api.report()
    assert len(report.groups) == 1 and len(report.groups[0].members) == 3
    assert report.groups[0].match_reasons[0] == "doi"
    assert report.groups[0].match_reasons[1].startswith("title:")
    assert {p.doi for p in report.groups[0].members} == {"10.0000/a", "10.0000/conflict"}


@pytest.mark.parametrize("field", ["doi", "paper_doi", "work_doi"])
def test_doi_aliases_and_first_nonblank_precedence(duplicate_api: DuplicateAPI, field: str) -> None:
    api = duplicate_api
    metadata = {"doi": " ", "paper_doi": "", "work_doi": "", field: "HTTP://DOI.ORG/10.0000/X."}
    mutate(
        api,
        "UPDATE documents SET metadata = ? WHERE document_id = 'a-doi'",
        (json.dumps(metadata),),
    )
    mutate(
        api,
        "UPDATE documents SET metadata = ? WHERE document_id = 'b-doi'",
        (json.dumps({"doi": "10.0000/x", "paper_doi": "10.0000/conflicting-lower-priority"}),),
    )
    report = api.report()
    group = next(g for g in report.groups if g.members[0].document_id == "a-doi")
    assert group.match_reasons == ("doi",)
    assert group.members[0].doi_field == field
    assert group.members[0].doi == metadata[field]
    assert group.members[1].doi == "10.0000/x"


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"collection_revision": "0"},
        {"collection_revision": "1.0"},
        {"collection_revision": "1e0"},
        {"collection_revision": "true"},
        {"collection_revision": str(2**63)},
        {"collection_revision": "1", "title_threshold": "-0.01"},
        {"collection_revision": "1", "title_threshold": "1.01"},
        {"collection_revision": "1", "title_threshold": "nan"},
        {"collection_revision": "1", "title_threshold": "inf"},
        {"collection_revision": "1", "title_threshold": "true"},
        {"collection_revision": "1", "format": "html"},
        {"collection_revision": "1", "limit": "1"},
        {"collection_revision": "1", "document_ids": "outside"},
        [("collection_revision", "1"), ("collection_revision", "1")],
        [("collection_revision", "1"), ("format", "json"), ("format", "markdown")],
        [("collection_revision", "1"), ("title_threshold", ".92"), ("title_threshold", "1")],
    ],
)
def test_invalid_http_requests_fail_closed(
    duplicate_api: DuplicateAPI, params: object, caplog: pytest.LogCaptureFixture
) -> None:
    response = duplicate_api.client.get(duplicate_api.route, params=params)
    assert_error(response, 422, "invalid_duplicate_request")
    assert "invalid_duplicate_request" in caplog.text


@pytest.mark.parametrize("identifier", ["not-a-collection", "col_" + "A" * 32, " " * 36])
def test_invalid_collection_ids(duplicate_api: DuplicateAPI, identifier: str) -> None:
    response = duplicate_api.client.get(
        f"/collections/{quote(identifier, safe='')}/duplicates", params={"collection_revision": 1}
    )
    assert_error(response, 422, "invalid_duplicate_request")


@pytest.mark.parametrize(
    "options",
    [
        {"collection_revision": True},
        {"collection_revision": "1"},
        {"collection_revision": 0},
        {"collection_revision": 2**63},
        {"title_threshold": True},
        {"title_threshold": ".92"},
        {"title_threshold": float("nan")},
        {"title_threshold": float("inf")},
        {"title_threshold": -1},
        {"title_threshold": 2},
        {"collection_id": "invalid"},
        {"collection_id": "\ud800"},
    ],
)
def test_python_validates_before_storage(
    duplicate_api: DuplicateAPI, options: dict[str, object]
) -> None:
    args = {
        "collection_id": duplicate_api.collection.collection_id,
        "collection_revision": 1,
        **options,
    }
    with pytest.raises(CollectionError) as caught:
        duplicate_api.container.collection_duplicates.inspect(**args)
    assert caught.value.status_code == 422 and caught.value.code == "invalid_duplicate_request"


def test_missing_stale_and_invalid_memberships(duplicate_api: DuplicateAPI) -> None:
    api = duplicate_api
    missing = api.client.get(
        "/collections/col_" + "0" * 32 + "/duplicates", params={"collection_revision": 1}
    )
    assert_error(missing, 404, "collection_not_found")
    assert_error(api.get(collection_revision=2), 409, "collection_revision_conflict")
    mutate(api, "DELETE FROM documents WHERE document_id = 'z-alone'")
    assert_error(api.get(), 409, "collection_documents_missing")
    mutate(api, 'UPDATE paper_collections SET document_ids = \'["a-doi", "a-doi"]\'')
    assert_error(api.get(), 409, "invalid_collection_record")


@pytest.mark.parametrize(
    ("field", "value", "status"),
    [
        ("title", b"private-blob", 409),
        ("source", b"private-blob", 409),
        ("metadata", b"{}", 409),
        ("metadata", "private-invalid-json", 409),
        ("metadata", "[]", 409),
        ("metadata", "null", 409),
        ("metadata", '{"doi": 42}', 409),
        pytest.param("metadata", '{"doi":' + "9" * 5000 + "}", 409, id="oversized-integer"),
        ("metadata", '{"paper_doi": null}', 409),
        ("metadata", '{"work_doi": {}}', 409),
        ("metadata", '{"doi":"one","doi":"two"}', 409),
        ("metadata", '{"x": NaN}', 409),
        ("metadata", '{"x": "\\ud800"}', 409),
        ("metadata", '{"doi": "valid", "paper_doi": 1}', 409),
        ("title", "x" * 513, 413),
        ("source", "x" * 513, 413),
        ("metadata", json.dumps({"doi": "x" * 513}), 413),
        ("metadata", json.dumps({"doi": "valid", "work_doi": "x" * 513}), 413),
        ("metadata", "{}" + " " * 16383, 413),
    ],
)
def test_invalid_or_oversized_stored_values_are_never_truncated(
    duplicate_api: DuplicateAPI,
    field: str,
    value: object,
    status: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sql = {
        "title": "UPDATE documents SET title = ? WHERE document_id = 'z-alone'",
        "source": "UPDATE documents SET source = ? WHERE document_id = 'z-alone'",
        "metadata": "UPDATE documents SET metadata = ? WHERE document_id = 'z-alone'",
    }[field]
    mutate(duplicate_api, sql, (value,))
    code = "duplicate_input_too_large" if status == 413 else "invalid_duplicate_document"
    assert_error(duplicate_api.get(), status, code)
    assert "private-" not in caplog.text


@pytest.mark.parametrize("column", ["title", "source", "metadata"])
def test_invalid_encoded_text_is_a_record_error(duplicate_api: DuplicateAPI, column: str) -> None:
    statements = {
        "title": "UPDATE documents SET title = CAST(X'FF' AS TEXT) WHERE document_id = 'a-doi'",
        "source": "UPDATE documents SET source = CAST(X'FF' AS TEXT) WHERE document_id = 'a-doi'",
        "metadata": (
            "UPDATE documents SET metadata = CAST(X'FF' AS TEXT) WHERE document_id = 'a-doi'"
        ),
    }
    mutate(duplicate_api, statements[column])
    assert_error(duplicate_api.get(), 409, "invalid_duplicate_document")


def test_sizes_are_checked_before_decoding_and_json_parsing(
    duplicate_api: DuplicateAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from storage import collection_duplicates

    api = duplicate_api
    mutate(
        api,
        "UPDATE documents SET title = CAST(? AS TEXT) WHERE document_id = 'z-alone'",
        (b"\xff" + b"x" * 2048,),
    )
    assert_error(api.get(), 413, "duplicate_input_too_large")
    mutate(
        api,
        "UPDATE documents SET title = '', metadata = ? WHERE document_id = 'z-alone'",
        ("{}" + " " * 16382,),
    )
    assert api.get().status_code == 200
    mutate(
        api,
        "UPDATE documents SET metadata = CAST(? AS TEXT) WHERE document_id = 'a-doi'",
        (b"\xff" + b"x" * 16384,),
    )
    parsed = []
    original = collection_duplicates.json.loads

    def loads(value: str, **kwargs: object) -> object:
        if "object_pairs_hook" in kwargs:
            parsed.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(collection_duplicates.json, "loads", loads)
    assert_error(api.get(), 413, "duplicate_input_too_large")
    assert parsed == []


def test_no_side_effects_or_unbounded_body_chunk_reads_and_deterministic_restart(
    duplicate_api: DuplicateAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = duplicate_api
    original_connect = sqlite3.connect
    read_connections = []
    statements = []

    def authorize(action: int, table: str, column: str, database: str, trigger: str) -> int:
        if action == sqlite3.SQLITE_READ:
            assert table in {"documents", "paper_collections", "json_each"}
            assert not (table == "documents" and column == "text")
        assert action not in {
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_CREATE_TABLE,
            sqlite3.SQLITE_DROP_TABLE,
        }
        return sqlite3.SQLITE_OK

    def readonly_connect(database: str, **kwargs: object) -> sqlite3.Connection:
        assert database.endswith("?mode=ro") and kwargs["uri"] is True
        connection = original_connect(database, **kwargs)
        connection.set_authorizer(authorize)
        connection.set_trace_callback(statements.append)
        read_connections.append(connection)
        return connection

    before = api.database.read_bytes()
    with monkeypatch.context() as patch:
        for target, method in (
            (api.container.runner, "run"),
            (api.container.runner, "preview"),
            (api.container.llm, "generate"),
            (api.container.hybrid_retriever, "retrieve"),
            (api.container.document_store, "list_chunks"),
            (api.container.document_store, "add_documents"),
            (api.container.document_catalog, "list_documents"),
            (api.container.event_log, "append_event"),
            (api.container.event_log, "append_transition"),
            (api.container.paper_collections, "create"),
            (api.container.paper_collections, "replace"),
            (api.container.paper_screening, "submit"),
        ):
            patch.setattr(target, method, denied)
        patch.setattr(sqlite3, "connect", readonly_connect)
        report = api.report()
        expected = {format: api.get(format=format).content for format in ("json", "markdown")}
    assert len(read_connections) == 3
    assert sum(statement == "BEGIN" for statement in statements) == 3
    assert api.database.read_bytes() == before
    restarted = create_app(offline_settings(api.database))
    with TestClient(restarted) as client:
        for format, content in expected.items():
            response = client.get(api.route, params={"collection_revision": 1, "format": format})
            assert response.content == content
    assert report == restarted.state.container.collection_duplicates.inspect(
        api.collection.collection_id, collection_revision=1
    )
    assert api.database.read_bytes() == before
    assert api.container.event_log.list_events() == []


def test_single_read_snapshot_survives_concurrent_membership_and_metadata_changes(
    duplicate_api: DuplicateAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = duplicate_api
    expected = api.get().content
    mutate(api, "PRAGMA journal_mode = WAL")
    original = SQLitePaperCollections.resolve_in_snapshot
    calls = []

    def resolve(connection: sqlite3.Connection, collection_id: str) -> PaperCollection:
        assert connection.row_factory is sqlite3.Row and connection.in_transaction
        collection = original(connection, collection_id)
        calls.append(collection)
        with closing(sqlite3.connect(api.database, timeout=1)) as writer, writer:
            writer.execute("UPDATE documents SET title = 'Changed', metadata = '{}'")
            writer.execute(
                "UPDATE paper_collections SET name = 'Changed', revision = 2, document_ids = ?",
                (json.dumps(["outside"]),),
            )
        return collection

    with monkeypatch.context() as patch:
        patch.setattr(SQLitePaperCollections, "resolve_in_snapshot", staticmethod(resolve))
        assert api.get().content == expected
    assert len(calls) == 1
    assert_error(api.get(), 409, "collection_revision_conflict")
    after = api.get(collection_revision=2).json()
    assert after["collection_name"] == "Changed"
    assert after["document_ids"] == ["outside"]
    assert after["singletons"][0]["title"] == "Changed"


def test_literal_markdown_preserves_inert_labels_and_complete_byte_cap(
    duplicate_api: DuplicateAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from storage import collection_duplicates

    api = duplicate_api
    label = '<script>alert("fixture")</script>\n``````\n[x](https://invalid.example)\n\u7814\u7a76'
    mutate(
        api,
        "UPDATE documents SET title = ?, source = ?, metadata = ?",
        (
            label,
            label,
            json.dumps({"doi": label}),
        ),
    )
    api.collection = api.container.paper_collections.replace(
        api.collection.collection_id,
        name="Review [x](https://invalid.example) <img src=x>",
        document_ids=api.collection.document_ids,
        expected_revision=1,
    )
    report = api.report()
    markdown = api.get(format="markdown")
    assert markdown.status_code == 200
    assert markdown.headers["content-type"] == "text/markdown; charset=utf-8"
    assert markdown.content == report.to_markdown().encode("utf-8")
    assert "```````json\n" in markdown.text
    literal = markdown.text.split("```````json\n", 1)[1].rsplit("```````\n", 1)[0]
    assert json.loads(literal) == json.loads(report.to_json())
    larger_size = max(len(report.to_json().encode()), len(markdown.content))
    assert larger_size <= collection_duplicates.MAX_RESPONSE_BYTES == 262144
    monkeypatch.setattr(collection_duplicates, "MAX_RESPONSE_BYTES", larger_size)
    assert api.get().status_code == api.get(format="markdown").status_code == 200
    monkeypatch.setattr(collection_duplicates, "MAX_RESPONSE_BYTES", larger_size - 1)
    for format in ("json", "markdown"):
        assert_error(api.get(format=format), 413, "duplicate_report_too_large")
    with pytest.raises(CollectionError, match="no partial"):
        api.report()


def test_hundred_papers_are_complete_and_actual_oversized_report_fails(
    duplicate_api: DuplicateAPI,
) -> None:
    api = duplicate_api
    ids = [f"paper-{i:03}" for i in reversed(range(100))]
    api.container.document_store.add_documents(
        [Document(document_id=i, title="", source="", text="private") for i in ids], []
    )
    api.collection = api.container.paper_collections.create(name="100 papers", document_ids=ids)
    report = api.report()
    assert report.inspected_count == len(report.singletons) == 100
    assert report.document_ids == tuple(sorted(ids)) and report.groups == ()
    with closing(sqlite3.connect(api.database)) as connection, connection:
        connection.executemany(
            "UPDATE documents SET title = ?, source = ?, metadata = ? WHERE document_id = ?",
            [("\U0001f4da" * 512, "\U0001f4da" * 512, "{}", i) for i in ids],
        )
    assert_error(api.get(), 413, "duplicate_report_too_large")
    mutate(api, "UPDATE paper_collections SET document_ids = ?", (json.dumps([*ids, "outside"]),))
    assert_error(api.get(), 409, "invalid_collection_record")


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
def test_exact_unicode_size_boundaries_without_clipping(tmp_path: Path, encoding: str) -> None:
    from storage.collection_duplicates import SQLiteCollectionDuplicates

    database = tmp_path / "unicode.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            {
                "UTF-8": "PRAGMA encoding = 'UTF-8'",
                "UTF-16le": "PRAGMA encoding = 'UTF-16le'",
                "UTF-16be": "PRAGMA encoding = 'UTF-16be'",
            }[encoding]
        )
        connection.execute("CREATE TABLE marker (value TEXT)")
    label = "\U0001f4da" * 512
    SQLiteDocumentStore(database).add_documents(
        [
            Document(
                document_id="unicode",
                title=label,
                source=label,
                text="private",
                metadata={"doi": label},
            )
        ],
        [],
    )
    collection = SQLitePaperCollections(database).create(name="Unicode", document_ids=["unicode"])
    service = SQLiteCollectionDuplicates(database)
    report = service.inspect(collection.collection_id, collection_revision=1)
    assert (
        report.singletons[0].title
        == report.singletons[0].source
        == report.singletons[0].doi
        == label
    )
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE documents SET title = ?", (label + "x",))
    with pytest.raises(CollectionError) as caught:
        service.inspect(collection.collection_id, collection_revision=1)
    assert caught.value.status_code == 413


def test_storage_unavailable_is_sanitized_and_never_creates_database(
    duplicate_api: DuplicateAPI, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from storage.collection_duplicates import SQLiteCollectionDuplicates

    missing = tmp_path / "must-not-create.sqlite3"
    service = SQLiteCollectionDuplicates(missing)
    with pytest.raises(CollectionError) as caught:
        service.inspect(duplicate_api.collection.collection_id, collection_revision=1)
    assert caught.value.code == "duplicate_storage_unavailable" and not missing.exists()
    mutate(duplicate_api, "DROP TABLE documents")
    response = duplicate_api.get()
    assert_error(response, 503, "duplicate_storage_unavailable")
    assert "no such table" not in response.text + caplog.text
    assert "duplicate_storage_unavailable" in caplog.text


def test_openapi_and_explicit_human_selected_preview(duplicate_api: DuplicateAPI) -> None:
    api = duplicate_api
    operation = api.client.get("/openapi.json").json()["paths"][
        "/collections/{collection_id}/duplicates"
    ]["get"]
    parameters = {parameter["name"]: parameter for parameter in operation["parameters"]}
    assert parameters["collection_revision"]["required"] is True
    assert parameters["title_threshold"]["schema"]["minimum"] == 0
    assert parameters["title_threshold"]["schema"]["maximum"] == 1
    assert parameters["format"]["schema"]["enum"] == ["json", "markdown"]
    assert {"200", "404", "409", "413", "422", "503"} <= set(operation["responses"])
    api.report()
    assert api.container.event_log.list_events() == []
    selected_ids = ["a-doi", "c-title"]
    response = api.client.post(
        "/retrieve", json={"query": "GraphRAG", "document_ids": selected_ids}
    )
    assert response.status_code == 200
    assert {s["chunk"]["document_id"] for s in response.json()["sources"]} == set(selected_ids)
    assert api.container.paper_collections.get(api.collection.collection_id) == api.collection
