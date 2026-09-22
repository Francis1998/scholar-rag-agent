"""Regression-first coverage for bounded, persisted chunk evidence inspection."""

import base64
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document


def no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Chunk inspection must not retrieve, generate, mutate, or use HTTP.")


@dataclass
class ChunkAPI:
    app: FastAPI
    client: TestClient
    container: AppContainer
    path: Path

    def seed(
        self,
        document_id: str = "selected",
        chunk_ids: tuple[str, ...] = ("c", "a", "b"),
        *,
        text: str = "Synthetic stored evidence.",
    ) -> None:
        document = Document(
            document_id=document_id,
            title="Synthetic methods",
            source="synthetic:chunks",
            text="UNEXPOSED_DOCUMENT_BODY",
            metadata={"private": "UNEXPOSED_METADATA"},
        )
        self.container.document_store.add_documents(
            [document],
            [
                Chunk(
                    chunk_id=identifier,
                    document_id=document_id,
                    title=document.title,
                    source=document.source,
                    text=text,
                    metadata={**document.metadata, "chunk_index": str(index)},
                )
                for index, identifier in enumerate(chunk_ids)
            ],
        )

    def get(self, document_id: str = "selected", **params: str | int) -> httpx.Response:
        return self.client.get(f"/documents/{quote(document_id, safe='')}/chunks", params=params)

    def page(self, document_id: str = "selected", **params: str | int) -> dict[str, Any]:
        response = self.get(document_id, **params)
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "UNEXPOSED_METADATA" not in response.text
        assert "UNEXPOSED_DOCUMENT_BODY" not in response.text
        result: dict[str, Any] = response.json()
        assert set(result) == {"document_id", "chunks", "next_cursor"}
        assert result["document_id"] == document_id
        return result


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ChunkAPI]:
    path = tmp_path / "corpus.sqlite3"
    app = create_app(offline_settings(path))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_work)
    with TestClient(app) as client:
        yield ChunkAPI(app, client, app.state.container, path)


def test_stored_chunks_are_inspectable_without_running_the_agent(
    api: ChunkAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.seed()
    before = api.path.read_bytes()
    for target, method in (
        (api.container.document_store, "list_chunks"),
        (api.container.document_store, "add_documents"),
        (api.container.hybrid_retriever, "retrieve"),
        (api.container.hybrid_retriever, "add_chunks"),
        (api.container.llm, "generate"),
        (api.container.runner, "run"),
        (api.container.event_log, "append_event"),
        (api.container.event_log, "list_events"),
    ):
        monkeypatch.setattr(target, method, no_work)
    page = api.page()
    assert [row["chunk_id"] for row in page["chunks"]] == ["a", "b", "c"]
    assert [row["chunk_index"] for row in page["chunks"]] == [1, 2, 0]
    assert page["next_cursor"] is None
    assert page["chunks"][0] == {
        "chunk_id": "a",
        "document_id": "selected",
        "chunk_index": 1,
        "title": "Synthetic methods",
        "title_truncated": False,
        "source": "synthetic:chunks",
        "source_truncated": False,
        "text": "Synthetic stored evidence.",
        "text_truncated": False,
    }
    assert api.path.read_bytes() == before


def test_known_empty_document_is_distinct_from_unknown_and_orphan_rows(api: ChunkAPI) -> None:
    api.seed("empty", ())
    assert api.page("empty") == {"document_id": "empty", "chunks": [], "next_cursor": None}
    api.seed("orphan", ("orphan-chunk",))
    with sqlite3.connect(api.path) as connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", ("orphan",))
    for identifier in ("missing", "orphan"):
        response = api.get(identifier)
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "document_not_found"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"


def test_first_middle_and_final_pages_are_exclusive_and_exactly_scoped(api: ChunkAPI) -> None:
    api.seed("selected", ("f", "d", "c", "b", "a"))
    api.seed("excluded", ("aa", "bb", "cc"), text="EXCLUDED_EVIDENCE")
    first = api.page(limit=2)
    assert [row["chunk_id"] for row in first["chunks"]] == ["a", "b"]
    assert isinstance(first["next_cursor"], str)
    second = api.page(limit=2, cursor=first["next_cursor"])
    third = api.page(limit=2, cursor=second["next_cursor"])
    assert [row["chunk_id"] for row in second["chunks"]] == ["c", "d"]
    assert [row["chunk_id"] for row in third["chunks"]] == ["f"]
    assert third["next_cursor"] is None
    rows = first["chunks"] + second["chunks"] + third["chunks"]
    assert len({row["chunk_id"] for row in rows}) == 5
    assert {row["document_id"] for row in rows} == {"selected"}
    assert "EXCLUDED_EVIDENCE" not in json.dumps(rows)
    assert api.page(limit=2, cursor=second["next_cursor"]) == third
    response = api.get("excluded", cursor=first["next_cursor"])
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_chunk_cursor"


def test_reingestion_and_inserts_behind_cursor_do_not_repeat_rows(api: ChunkAPI) -> None:
    api.seed()
    first = api.page(limit=1)
    api.seed(chunk_ids=("a", "a0", "d"))
    second = api.page(limit=10, cursor=first["next_cursor"])
    assert [row["chunk_id"] for row in second["chunks"]] == ["a0", "b", "c", "d"]
    assert second["next_cursor"] is None


@pytest.mark.parametrize(
    "identifier",
    ["paper-\u03b2", "https://doi.org/10.1/'quoted'", "' OR 1=1 --", "percent%?#", "nul\x00id"],
)
def test_imported_ids_remain_exact_not_hash_parsed_or_normalized(
    api: ChunkAPI, identifier: str
) -> None:
    api.seed(identifier, ("chunk-\u7814", "chunk-'quoted'"))
    first = api.page(identifier, limit=1)
    second = api.page(identifier, cursor=first["next_cursor"])
    assert [row["chunk_id"] for row in first["chunks"] + second["chunks"]] == [
        "chunk-'quoted'",
        "chunk-\u7814",
    ]
    assert api.get(identifier + "-different").status_code == 404


@pytest.mark.parametrize("identifier", ["", " ", " leading", "trailing ", "a" * 129])
def test_invalid_document_ids_fail_explicitly(api: ChunkAPI, identifier: str) -> None:
    assert api.get(identifier).status_code == 422


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("limit", "0"),
        ("limit", "-1"),
        ("limit", "101"),
        ("limit", "true"),
        ("limit", "1.5"),
        ("limit", ""),
        ("cursor", ""),
        ("cursor", "not-a-cursor"),
        ("cursor", "a" * 4097),
        ("cursor", "===="),
        ("cursor", "e30"),
    ],
)
def test_invalid_limits_and_cursors_fail_explicitly(
    api: ChunkAPI, parameter: str, value: str
) -> None:
    api.seed()
    response = api.get(**{parameter: value})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 2, "document_id": "selected", "chunk_id": "a"},
        {"version": True, "document_id": "selected", "chunk_id": "a"},
        {"version": 1.0, "document_id": "selected", "chunk_id": "a"},
        {"version": 1, "document_id": "selected", "chunk_id": ""},
        {"version": 1, "document_id": "selected", "chunk_id": 42},
        {"version": 1, "document_id": "selected", "chunk_id": "a", "extra": True},
        {"version": 1, "document_id": "selected", "chunk_id": "a" * 257},
    ],
)
def test_structurally_invalid_cursor_is_not_a_silent_empty_page(
    api: ChunkAPI, payload: dict[str, object]
) -> None:
    api.seed()
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    cursor = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    response = api.get(cursor=cursor)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_chunk_cursor"


def test_limits_and_unicode_nul_text_prefixes_are_bounded(api: ChunkAPI) -> None:
    text = "a\x00" + "\U0001f52c" * 10000
    api.seed(chunk_ids=tuple(f"c-{index:03}" for index in range(101)), text=text)
    with sqlite3.connect(api.path) as connection:
        connection.execute(
            "UPDATE chunks SET title = ?, source = ?",
            ("\u7814" * 301, "s" * 513),
        )
    assert len(api.page()["chunks"]) == 20
    first = api.page(limit=100)
    assert len(first["chunks"]) == 100
    assert len(api.page(limit=100, cursor=first["next_cursor"])["chunks"]) == 1
    for chunk in first["chunks"]:
        assert chunk["text"] == text[:4000]
        assert chunk["text_truncated"] is True
        assert chunk["title"] == "\u7814" * 300
        assert chunk["title_truncated"] is True
        assert chunk["source"] == "s" * 512
        assert chunk["source_truncated"] is True


def test_exact_text_boundary_and_empty_text_are_not_reported_as_truncated(api: ChunkAPI) -> None:
    api.seed(chunk_ids=("a",), text="\U0001f52c" * 4000)
    assert api.page()["chunks"][0]["text_truncated"] is False
    api.seed(chunk_ids=("a",), text="")
    chunk = api.page()["chunks"][0]
    assert chunk["text"] == ""
    assert chunk["text_truncated"] is False


@pytest.mark.parametrize("metadata", ["{}", '{"private": "UNEXPOSED_METADATA"}'])
def test_absent_chunk_index_is_none_not_a_fabricated_ordinal(api: ChunkAPI, metadata: str) -> None:
    api.seed(chunk_ids=("opaque-73",))
    with sqlite3.connect(api.path) as connection:
        connection.execute("UPDATE chunks SET metadata = ?", (metadata,))
    assert api.page()["chunks"][0]["chunk_index"] is None


@pytest.mark.parametrize(
    "metadata",
    [
        "not-json",
        "[]",
        '{"chunk_index": null}',
        '{"chunk_index": true}',
        '{"chunk_index": 1.5}',
        '{"chunk_index": "-1"}',
        '{"chunk_index": "01"}',
        '{"chunk_index": "1\\u0000invalid"}',
        '{"chunk_index": "not-an-index"}',
        '{"chunk_index": "9223372036854775808"}',
    ],
)
def test_invalid_projected_metadata_fails_without_private_payload(
    api: ChunkAPI, metadata: str
) -> None:
    api.seed()
    with sqlite3.connect(api.path) as connection:
        connection.execute("UPDATE chunks SET metadata = ? WHERE chunk_id = 'b'", (metadata,))
    response = api.get(limit=1)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_chunk_record"
    assert "UNEXPOSED" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    "assignment",
    [
        "chunk_id = ''",
        "chunk_id = NULL",
        "chunk_id = zeroblob(100)",
        "title = zeroblob(100)",
        "source = zeroblob(100)",
        "text = zeroblob(100)",
        "text = CAST(X'80' AS TEXT)",
    ],
)
def test_invalid_stored_chunk_fields_are_explicit_conflicts(api: ChunkAPI, assignment: str) -> None:
    api.seed(chunk_ids=("a",))
    with sqlite3.connect(api.path) as connection:
        connection.execute(f"UPDATE chunks SET {assignment}")  # noqa: S608
    response = api.get()
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_chunk_record"


def test_reads_survive_restart_and_do_not_change_existing_query_evidence(api: ChunkAPI) -> None:
    response = api.client.post(
        "/ingest/text",
        json={
            "title": "Synthetic selected note",
            "text": "GraphRAG connects synthetic evidence. " * 40,
            "source": "synthetic:restart",
        },
    )
    response.raise_for_status()
    identifier = response.json()["document_id"]
    api.seed("excluded", ("excluded-chunk",), text="EXCLUDED_EVIDENCE")
    before = api.page(identifier)
    api.app.state.container = AppContainer(offline_settings(api.path))
    assert api.page(identifier) == before
    assert api.app.state.container.event_log.list_events() == []
    catalog = api.client.get("/documents", params={"source": "synthetic:restart"}).json()
    assert catalog["documents"][0]["document_id"] == identifier
    query = api.client.post(
        "/query", json={"query": "GraphRAG evidence", "document_ids": [identifier]}
    )
    query.raise_for_status()
    run = query.json()["result"]
    assert run["state"] == "DONE"
    bundle = api.client.get(f"/runs/{run['run_id']}/export")
    bundle.raise_for_status()
    assert {row["chunk"]["document_id"] for row in bundle.json()["snapshot"]["sources"]} == {
        identifier
    }
    events = api.app.state.container.event_log.list_events()
    assert api.page(identifier) == before
    assert api.app.state.container.event_log.list_events() == events
    assert api.client.get(f"/runs/{run['run_id']}/export").content == bundle.content
    assert "EXCLUDED_EVIDENCE" not in bundle.text


def test_python_reader_is_strict_read_only_and_index_backed(api: ChunkAPI, tmp_path: Path) -> None:
    from storage.document_chunks import _NEXT_PAGE_SQL, SQLiteDocumentChunks

    api.seed()
    reader = SQLiteDocumentChunks(api.path)
    for value in (0, 101, True, 1.5, "2"):
        with pytest.raises(ValidationError):
            reader.list_chunks("selected", limit=value)  # type: ignore[arg-type]
    for value in ("", " ", 42, None):
        with pytest.raises(ValidationError):
            reader.list_chunks(value)  # type: ignore[arg-type]
    before = api.path.read_bytes()
    assert len(reader.list_chunks("selected").chunks) == 3
    assert api.path.read_bytes() == before
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        SQLiteDocumentChunks(missing).list_chunks("selected")
    assert not missing.exists()
    with closing(sqlite3.connect(api.path)) as connection:
        plan = connection.execute(
            "EXPLAIN QUERY PLAN " + _NEXT_PAGE_SQL,
            {
                "document_id": "selected",
                "cursor": "a",
                "fetch_limit": 21,
                "identity_bytes": 1028,
                "title_bytes": 1204,
                "source_bytes": 2052,
                "text_bytes": 16004,
            },
        ).fetchall()
    assert any("idx_chunks_document_chunk" in row[3] for row in plan)
    assert any("document_id=? AND chunk_id>?" in row[3] for row in plan)
    assert all("TEMP B-TREE" not in row[3] for row in plan)


@pytest.mark.parametrize("encoding", ["UTF-16le", "UTF-16be"])
def test_standalone_reader_handles_utf16_without_mutation(tmp_path: Path, encoding: str) -> None:
    from storage.document_chunks import SQLiteDocumentChunks
    from storage.document_store import SQLiteDocumentStore

    path = tmp_path / "utf16.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA encoding = '{encoding}'")
        connection.execute("CREATE TABLE marker (value TEXT)")
    store = SQLiteDocumentStore(path)
    text = "\U0001f52c\x00\u7814" * 3000
    store.add_documents(
        [Document(document_id="paper-\u03b2", title="Synthetic", text="private", source="local")],
        [
            Chunk(
                chunk_id="chunk-\u7814",
                document_id="paper-\u03b2",
                title="Synthetic",
                text=text,
                source="local",
            )
        ],
    )
    before = path.read_bytes()
    chunk = SQLiteDocumentChunks(path).list_chunks("paper-\u03b2").chunks[0]
    assert chunk.text == text[:4000] and chunk.text_truncated
    assert chunk.chunk_index is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("index", [0, 7, 9223372036854775807, "0", "7", "9223372036854775807"])
def test_imported_ordinals_are_preserved_without_defining_order(
    api: ChunkAPI, index: int | str
) -> None:
    api.seed()
    with sqlite3.connect(api.path) as connection:
        connection.execute("UPDATE chunks SET metadata = ?", (json.dumps({"chunk_index": index}),))
    first = api.page(limit=2)
    second = api.page(limit=2, cursor=first["next_cursor"])
    chunks = first["chunks"] + second["chunks"]
    assert [chunk["chunk_id"] for chunk in chunks] == ["a", "b", "c"]
    assert [chunk["chunk_index"] for chunk in chunks] == [int(index)] * 3


def test_cursor_survives_removed_boundary_and_exhausts_without_repeats(api: ChunkAPI) -> None:
    api.seed()
    first = api.page(limit=2)
    with sqlite3.connect(api.path) as connection:
        connection.execute("DELETE FROM chunks WHERE chunk_id IN ('b', 'c')")
    assert api.page(cursor=first["next_cursor"]) == {
        "document_id": "selected",
        "chunks": [],
        "next_cursor": None,
    }


def test_maximum_unicode_id_cursor_round_trips_without_truncation(api: ChunkAPI) -> None:
    document_id = "\U0001f52c" * 128
    identifiers = ("\u7814" * 255 + "a", "\u7814" * 255 + "b")
    api.seed(document_id, identifiers)
    first = api.page(document_id, limit=1)
    assert len(first["next_cursor"]) <= 4096
    second = api.page(document_id, cursor=first["next_cursor"])
    assert [chunk["chunk_id"] for chunk in first["chunks"] + second["chunks"]] == list(identifiers)
    assert second["next_cursor"] is None


def test_invalid_long_chunk_identity_fails_instead_of_being_truncated(api: ChunkAPI) -> None:
    api.seed(chunk_ids=("c" * 257,))
    assert api.get().status_code == 409


def test_storage_failures_are_not_disguised_as_empty_evidence(api: ChunkAPI) -> None:
    api.seed()
    with sqlite3.connect(api.path) as connection:
        connection.execute("DROP TABLE chunks")
    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.get("/documents/selected/chunks")
    assert response.status_code == 500


def test_openapi_describes_exact_response_bounds_and_errors(api: ChunkAPI) -> None:
    schema = api.client.get("/openapi.json").json()
    operation = schema["paths"]["/documents/{document_id}/chunks"]["get"]
    assert {"200", "404", "409", "422"} <= set(operation["responses"])
    parameters = {parameter["name"]: parameter["schema"] for parameter in operation["parameters"]}
    assert parameters["document_id"]["maxLength"] == 128
    assert parameters["limit"]["type"] == "integer"
    assert parameters["limit"]["minimum"] == 1
    assert parameters["limit"]["maximum"] == 100
    assert parameters["limit"]["default"] == 20
    models = schema["components"]["schemas"]
    assert models["DocumentChunksPage"]["properties"]["chunks"]["maxItems"] == 100
    assert models["StoredChunk"]["properties"]["text"]["maxLength"] == 4000
