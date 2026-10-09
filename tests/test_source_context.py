"""Regression-first coverage for exact, bounded, current-corpus source context."""

import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from retrieval.models import Chunk, Document
from tests.test_corpus_explorer import ExplorerAPI, no_work
from tests.test_corpus_explorer import api as api


def context(
    api: ExplorerAPI,
    document_id: str = "selected",
    chunk_id: str = "chunk-10",
    **params: str | int,
) -> httpx.Response:
    return api.client.get(
        "/documents/context",
        params={"document_id": document_id, "chunk_id": chunk_id, **params},
    )


def assert_error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "SECRET" not in response.text


def test_numeric_source_window_is_not_lexicographic_chunk_pagination(api: ExplorerAPI) -> None:
    api.seed(chunk_ids=tuple(f"chunk-{index}" for index in range(14)))
    api.seed("excluded", ("excluded-anchor",), text="EXCLUDED_EVIDENCE")
    response = context(api)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    result = response.json()
    assert set(result) == {
        "document_id",
        "anchor_chunk_id",
        "anchor_chunk_index",
        "before",
        "after",
        "returned_before",
        "returned_after",
        "has_more_before",
        "has_more_after",
        "chunks",
    }
    assert result["document_id"] == "selected"
    assert result["anchor_chunk_id"] == "chunk-10"
    assert result["anchor_chunk_index"] == 10
    assert result["before"] == result["after"] == 2
    assert result["returned_before"] == result["returned_after"] == 2
    assert result["has_more_before"] and result["has_more_after"]
    assert [chunk["chunk_id"] for chunk in result["chunks"]] == [
        "chunk-8",
        "chunk-9",
        "chunk-10",
        "chunk-11",
        "chunk-12",
    ]
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [8, 9, 10, 11, 12]
    assert [chunk["is_anchor"] for chunk in result["chunks"]] == [False, False, True, False, False]
    assert {chunk["document_id"] for chunk in result["chunks"]} == {"selected"}
    assert "EXCLUDED_EVIDENCE" not in response.text and "UNEXPOSED_" not in response.text


@pytest.mark.parametrize(
    ("anchor", "before", "after", "expected", "more_before", "more_after"),
    [
        (0, 5, 5, list(range(6)), False, True),
        (13, 5, 5, list(range(8, 14)), True, False),
        (7, 5, 5, list(range(2, 13)), True, True),
        (1, 0, 0, [1], True, True),
        (0, 0, 0, [0], False, True),
        (13, 0, 0, [13], True, False),
        (10, 1, 0, [9, 10], True, True),
        (10, 0, 3, [10, 11, 12, 13], True, False),
    ],
)
def test_counts_are_neighbors_not_inclusive_radius_and_do_not_fill_from_other_side(
    api: ExplorerAPI,
    anchor: int,
    before: int,
    after: int,
    expected: list[int],
    more_before: bool,
    more_after: bool,
) -> None:
    api.seed(chunk_ids=tuple(f"chunk-{index}" for index in range(14)))
    result = context(api, chunk_id=f"chunk-{anchor}", before=before, after=after).json()
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == expected
    assert result["returned_before"] == sum(index < anchor for index in expected)
    assert result["returned_after"] == sum(index > anchor for index in expected)
    assert result["has_more_before"] is more_before
    assert result["has_more_after"] is more_after


def test_gaps_integer_imports_and_one_chunk_preserve_recorded_order(api: ExplorerAPI) -> None:
    api.seed(chunk_ids=("opaque-a", "opaque-z", "opaque-b"))
    with sqlite3.connect(api.path) as connection:
        for identifier, ordinal in (
            ("opaque-a", 9223372036854775807),
            ("opaque-z", 0),
            ("opaque-b", 7),
        ):
            connection.execute(
                "UPDATE chunks SET metadata = ? WHERE chunk_id = ?",
                (json.dumps({"chunk_index": ordinal}), identifier),
            )
    result = context(api, chunk_id="opaque-b").json()
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [0, 7, 9223372036854775807]
    assert not result["has_more_before"] and not result["has_more_after"]
    api.seed("single", ("only",))
    result = context(api, "single", "only", before=5, after=5).json()
    assert len(result["chunks"]) == 1 and result["chunks"][0]["is_anchor"]
    assert result["returned_before"] == result["returned_after"] == 0
    assert not result["has_more_before"] and not result["has_more_after"]


@pytest.mark.parametrize(
    ("document_id", "chunk_id", "code"),
    [
        ("missing-SECRET", "a", "document_not_found"),
        ("selected", "missing-SECRET", "chunk_not_found"),
        ("selected", "other-anchor", "chunk_not_found"),
        ("empty", "a", "chunk_not_found"),
        ("orphan", "orphan-anchor", "document_not_found"),
    ],
)
def test_missing_and_wrong_owner_anchors_are_explicit(
    api: ExplorerAPI, document_id: str, chunk_id: str, code: str
) -> None:
    api.seed()
    api.seed("other", ("other-anchor",))
    api.seed("empty", ())
    api.seed("orphan", ("orphan-anchor",))
    with sqlite3.connect(api.path) as connection:
        connection.execute("DELETE FROM documents WHERE document_id = 'orphan'")
    assert_error(context(api, document_id, chunk_id), 404, code)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"document_id": "selected"},
        {"chunk_id": "a"},
        {"document_id": "", "chunk_id": "a"},
        {"document_id": " selected", "chunk_id": "a"},
        {"document_id": "selected ", "chunk_id": "a"},
        {"document_id": "d" * 129, "chunk_id": "a"},
        {"document_id": "selected", "chunk_id": ""},
        {"document_id": "selected", "chunk_id": "c" * 257},
        *(
            {"document_id": "selected", "chunk_id": "a", name: value}
            for name in ("before", "after")
            for value in ("-1", "6", "1.5", "true", "", "SECRET")
        ),
    ],
)
def test_invalid_query_is_sanitized_and_noncaching(
    api: ExplorerAPI, params: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    response = api.client.get("/documents/context", params=params)
    assert_error(response, 422, "invalid_source_context_request")
    assert "invalid_source_context_request" in caplog.text
    assert "SECRET" not in "\n".join(
        record.message for record in caplog.records if record.name.startswith("api.")
    )


@pytest.mark.parametrize(
    "metadata",
    [
        "{}",
        "[]",
        "null",
        "SECRET-not-json",
        '{"chunk_index": null}',
        '{"chunk_index": true}',
        '{"chunk_index": 1.5}',
        '{"chunk_index": -1}',
        '{"chunk_index": "01"}',
        '{"chunk_index": "+1"}',
        '{"chunk_index": " 1"}',
        '{"chunk_index": "1\\u0000SECRET"}',
        '{"chunk_index": "\\u0661"}',
        '{"chunk_index": "9223372036854775808"}',
        '{"chunk_index": 9223372036854775808}',
        '{"chunk_index": "1", "chunk_index": "2"}',
        '{"chunk_index": "1", "chunk_\\u0069ndex": "1"}',
        '{"nested": {"chunk_index": "1"}}',
        '{"chunk_index": "2", "private": NaN}',
    ],
)
def test_every_same_document_order_is_validated_even_outside_the_window(
    api: ExplorerAPI, metadata: str
) -> None:
    api.seed(chunk_ids=tuple(f"chunk-{index}" for index in range(14)))
    with sqlite3.connect(api.path) as connection:
        connection.execute("UPDATE chunks SET metadata = ? WHERE chunk_id = 'chunk-0'", (metadata,))
    assert_error(context(api, before=0, after=0), 409, "invalid_source_order")


def test_duplicate_ordinals_anywhere_are_not_tiebroken_by_chunk_id(api: ExplorerAPI) -> None:
    api.seed(chunk_ids=tuple(f"chunk-{index}" for index in range(14)))
    with sqlite3.connect(api.path) as connection:
        connection.execute(
            "UPDATE chunks SET metadata = '{\"chunk_index\": \"1\"}' WHERE chunk_id = 'chunk-0'"
        )
    assert_error(context(api, before=0, after=0), 409, "ambiguous_source_order")


@pytest.mark.parametrize(
    "assignment",
    [
        "chunk_id = ''",
        "chunk_id = NULL",
        "chunk_id = zeroblob(100)",
        "chunk_id = CAST(X'80' AS TEXT)",
        "chunk_id = printf('%0300d', 1)",
        "metadata = zeroblob(100)",
        "metadata = CAST(X'80' AS TEXT)",
    ],
)
def test_invalid_ordering_identity_and_metadata_reject_whole_window(
    api: ExplorerAPI, assignment: str
) -> None:
    api.seed()
    with sqlite3.connect(api.path) as connection:
        connection.execute(f"UPDATE chunks SET {assignment} WHERE chunk_id = 'c'")  # noqa: S608
    assert_error(context(api, chunk_id="a", before=0, after=0), 409, "invalid_source_order")


@pytest.mark.parametrize(
    "assignment",
    [
        "title = zeroblob(100)",
        "source = zeroblob(100)",
        "text = zeroblob(100)",
        "text = CAST(X'80' AS TEXT)",
    ],
)
def test_invalid_selected_evidence_never_becomes_a_partial_success(
    api: ExplorerAPI, assignment: str
) -> None:
    api.seed()
    with sqlite3.connect(api.path) as connection:
        connection.execute(f"UPDATE chunks SET {assignment} WHERE chunk_id = 'b'")  # noqa: S608
    assert_error(context(api, chunk_id="a"), 409, "invalid_chunk_record")
    assert context(api, chunk_id="a", before=0, after=0).status_code == 200


@pytest.mark.parametrize(
    "identifier",
    [
        ".",
        "..",
        "paper/../other",
        "https://doi.org/x/../y",
        "quotes'\"?%#&+\\",
        "nul\x00id",
        "line\r\nid",
        "\U0001f52c" * 128,
    ],
)
def test_query_selectors_preserve_exact_unicode_slash_dot_and_control_ids(
    api: ExplorerAPI, identifier: str
) -> None:
    chunk_id = identifier + "/chunk?\u7814"
    api.seed(identifier, (chunk_id,))
    result = context(api, identifier, chunk_id).json()
    assert result["document_id"] == identifier
    assert result["anchor_chunk_id"] == chunk_id
    assert result["chunks"][0]["chunk_id"] == chunk_id
    assert_error(context(api, identifier, chunk_id + "-different"), 404, "chunk_not_found")


def test_stored_prefixes_keep_provenance_and_truncation_without_normalization(
    api: ExplorerAPI,
) -> None:
    text = "\n\r\n\u7814\U0001f52c\x00" * 1000
    api.seed(title="t" * 301, source="\U0001f52c" * 513, text=text)
    result = context(api, chunk_id="a").json()
    for chunk in result["chunks"]:
        assert chunk["text"] == text[:4000] and chunk["text_truncated"]
        assert chunk["title"] == "t" * 300 and chunk["title_truncated"]
        assert chunk["source"] == "\U0001f52c" * 512 and chunk["source_truncated"]
    api.seed(text="\U0001f52c" * 4000)
    assert not context(api, chunk_id="a", before=0, after=0).json()["chunks"][0]["text_truncated"]
    api.seed(text="")
    chunk = context(api, chunk_id="a", before=0, after=0).json()["chunks"][0]
    assert chunk["text"] == "" and not chunk["text_truncated"]


def test_metadata_and_document_count_caps_are_fixed_and_exact(api: ExplorerAPI) -> None:
    from storage.source_context import MAX_CONTEXT_DOCUMENT_CHUNKS, MAX_CONTEXT_METADATA_BYTES

    assert MAX_CONTEXT_DOCUMENT_CHUNKS == 2048
    assert MAX_CONTEXT_METADATA_BYTES == 8192
    api.seed(chunk_ids=tuple(f"chunk-{index}" for index in range(MAX_CONTEXT_DOCUMENT_CHUNKS)))
    raw = '{"chunk_index":"0","private":"' + "x" * (MAX_CONTEXT_METADATA_BYTES - 32) + '"}'
    assert len(raw.encode()) == MAX_CONTEXT_METADATA_BYTES
    with sqlite3.connect(api.path) as connection:
        connection.execute("UPDATE chunks SET metadata = ? WHERE chunk_id = 'chunk-0'", (raw,))
    assert context(api, before=0, after=0).status_code == 200
    with sqlite3.connect(api.path) as connection:
        connection.execute(
            "UPDATE chunks SET metadata = ? WHERE chunk_id = 'chunk-0'", (raw + " ",)
        )
    assert_error(context(api, before=0, after=0), 409, "source_context_metadata_too_large")
    api.seed(chunk_ids=("extra",))
    assert_error(context(api, before=0, after=0), 409, "source_context_document_too_large")


def test_sql_bounds_discovery_and_projects_only_selected_evidence(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from storage.source_context import SQLiteSourceContext

    api.seed(chunk_ids=tuple(f"chunk-{index}" for index in range(14)))
    api.seed("excluded", ("excluded",))
    with sqlite3.connect(api.path) as connection:
        connection.execute("UPDATE chunks SET text = zeroblob(8000000) WHERE chunk_id = 'chunk-0'")
        connection.execute(
            "UPDATE chunks SET metadata = zeroblob(8000000) WHERE document_id = 'excluded'"
        )
        connection.execute(
            "UPDATE documents SET text = zeroblob(8000000), metadata = zeroblob(8000000)"
        )
    real_connect = sqlite3.connect
    queries: list[tuple[str, Mapping[str, object]]] = []
    opened: list[str] = []
    projected: list[sqlite3.Row] = []

    def row_factory(cursor: sqlite3.Cursor, row: tuple[object, ...]) -> sqlite3.Row:
        assert all(not isinstance(value, bytes) or len(value) <= 16004 for value in row)
        result = sqlite3.Row(cursor, row)
        projected.append(result)
        return result

    class ObservedConnection(sqlite3.Connection):
        def execute(
            self, sql: str, parameters: Mapping[str, object] | tuple[object, ...] = (), /
        ) -> sqlite3.Cursor:
            if isinstance(parameters, dict):
                queries.append((sql, parameters))
            self.row_factory = row_factory
            return super().execute(sql, parameters)

    def connect(database: str, *, uri: bool = False) -> sqlite3.Connection:
        assert database.endswith("?mode=ro") and uri is True
        opened.append(database)
        return real_connect(database, factory=ObservedConnection, uri=uri)

    monkeypatch.setattr(sqlite3, "connect", connect)
    result = SQLiteSourceContext(api.path).read("selected", "chunk-10", before=1, after=1)
    assert [chunk.chunk_index for chunk in result.chunks] == [9, 10, 11]
    assert len(opened) == 1
    discovery = [(sql, params) for sql, params in queries if "metadata_bytes" in params]
    assert len(discovery) == 1
    assert discovery[0][1]["fetch_limit"] == 2049 and "LIMIT" in discovery[0][0]
    evidence = [(sql, params) for sql, params in queries if "chunk_ids" in params]
    assert len(evidence) == 1
    assert json.loads(evidence[0][1]["chunk_ids"]) == ["chunk-9", "chunk-10", "chunk-11"]
    assert evidence[0][1]["fetch_limit"] <= 11
    assert "LIMIT" in evidence[0][0] and "substr(CAST(text AS BLOB)" in evidence[0][0]
    assert len(projected) <= 14 + 3 + 3
    assert all("metadata" not in row.keys() for row in projected)  # noqa: SIM118
    with closing(real_connect(api.path)) as connection:
        for sql, params in queries:
            plan = connection.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
            assert any("INDEX" in row[3] for row in plan), plan
            assert all("SCAN chunks" not in row[3] for row in plan)


def test_one_read_snapshot_survives_a_concurrent_replacement(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from storage.source_context import SQLiteSourceContext

    api.seed(chunk_ids=("c0", "c1", "c2"))
    with sqlite3.connect(api.path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    real_connect = sqlite3.connect
    replaced = False

    class ConcurrentConnection(sqlite3.Connection):
        def execute(
            self, sql: str, parameters: Mapping[str, object] | tuple[object, ...] = (), /
        ) -> sqlite3.Cursor:
            nonlocal replaced
            if isinstance(parameters, dict) and "chunk_ids" in parameters and not replaced:
                replaced = True
                with closing(real_connect(api.path)) as writer, writer:
                    writer.execute("UPDATE chunks SET text = 'NEW EVIDENCE' WHERE chunk_id = 'c1'")
                    writer.execute("DELETE FROM chunks WHERE chunk_id = 'c0'")
            return super().execute(sql, parameters)

    def connect(database: str, *, uri: bool = False) -> sqlite3.Connection:
        return real_connect(database, factory=ConcurrentConnection, uri=uri)

    monkeypatch.setattr(sqlite3, "connect", connect)
    reader = SQLiteSourceContext(api.path)
    first = reader.read("selected", "c1")
    assert replaced
    assert [chunk.chunk_id for chunk in first.chunks] == ["c0", "c1", "c2"]
    assert first.chunks[1].text == "Synthetic stored evidence."
    second = reader.read("selected", "c1")
    assert [chunk.chunk_id for chunk in second.chunks] == ["c1", "c2"]
    assert second.chunks[0].text == "NEW EVIDENCE"


def test_python_read_is_strict_and_never_creates_a_missing_database(
    api: ExplorerAPI, tmp_path: Path
) -> None:
    from storage.source_context import SourceContextError, SQLiteSourceContext

    api.seed()
    reader = SQLiteSourceContext(api.path)
    for field in ("before", "after"):
        for value in (True, 1.5, "2", -1, 6, None):
            with pytest.raises(ValidationError):
                reader.read("selected", "a", **{field: value})  # type: ignore[arg-type]
    for doc, chunk in (
        (None, "a"),
        ("selected", None),
        ("selected", ""),
        ("\ud800", "a"),
        ("selected", "\ud800"),
    ):
        with pytest.raises(ValidationError):
            reader.read(doc, chunk)  # type: ignore[arg-type]
    before = api.path.read_bytes()
    result = reader.read("selected", "a")
    assert result.model_dump(mode="json") == context(api, chunk_id="a").json()
    assert api.path.read_bytes() == before
    missing = tmp_path / "never-created.sqlite3"
    with pytest.raises(SourceContextError) as error:
        SQLiteSourceContext(missing).read("selected", "a")
    assert error.value.code == "source_context_storage_unavailable"
    assert error.value.status_code == 503 and not missing.exists()


@pytest.mark.parametrize("encoding", ["UTF-16le", "UTF-16be"])
def test_python_reader_preserves_utf16_prefixes_and_order(tmp_path: Path, encoding: str) -> None:
    from storage.document_store import SQLiteDocumentStore
    from storage.source_context import SQLiteSourceContext

    path = tmp_path / "utf16.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA encoding = '{encoding}'")
        connection.execute("CREATE TABLE marker (value TEXT)")
    store = SQLiteDocumentStore(path)
    store.add_documents(
        [Document(document_id="\u7814", title="Synthetic", text="PRIVATE", source="local")],
        [
            Chunk(
                chunk_id=f"c{index}",
                document_id="\u7814",
                title="Synthetic",
                text=("\U0001f52c\x00\u7814" * 2000),
                source="local",
                metadata={"chunk_index": str(index)},
            )
            for index in range(12)
        ],
    )
    before = path.read_bytes()
    result = SQLiteSourceContext(path).read("\u7814", "c10")
    assert [chunk.chunk_index for chunk in result.chunks] == [8, 9, 10, 11]
    assert result.chunks[2].text == ("\U0001f52c\x00\u7814" * 2000)[:4000]
    assert result.chunks[2].text_truncated
    assert path.read_bytes() == before


def test_real_ingestion_restart_and_reads_leave_retrieval_events_and_exports_unchanged(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingested = api.client.post(
        "/ingest/text",
        json={
            "title": "Synthetic context",
            "source": "synthetic:context",
            "text": "GraphRAG evidence. " * 150,
        },
    )
    ingested.raise_for_status()
    document_id = ingested.json()["document_id"]
    chunk_id = ingested.json()["chunk_ids"][1]
    query = api.client.post(
        "/query", json={"query": "GraphRAG evidence", "document_ids": [document_id]}
    )
    query.raise_for_status()
    export_url = f"/runs/{query.json()['result']['run_id']}/export"
    bundle = api.client.get(export_url).content
    first = context(api, document_id, chunk_id).content
    api.app.state.container = create_app(offline_settings(api.path)).state.container
    container = api.app.state.container
    before = api.path.read_bytes()
    events = container.event_log.list_events()
    catalog = api.client.get("/documents").content
    chunks = container.document_chunks.list_chunks(document_id)
    for target, method in (
        (container.document_store, "list_chunks"),
        (container.document_store, "add_documents"),
        (container.hybrid_retriever, "retrieve"),
        (container.hybrid_retriever, "add_chunks"),
        (container.runner._executor._reranker, "rerank"),
        (container.llm, "generate"),
        (container.runner, "run"),
        (container.runner, "preview"),
        (container.event_log, "append_event"),
        (container.ingestion_pipeline, "ingest_documents"),
    ):
        monkeypatch.setattr(target, method, no_work)
    assert context(api, document_id, chunk_id).content == first
    assert container.source_context.read(document_id, chunk_id).model_dump(
        mode="json"
    ) == json.loads(first)
    assert api.client.get("/documents").content == catalog
    assert container.document_chunks.list_chunks(document_id) == chunks
    assert api.client.get(export_url).content == bundle
    assert container.event_log.list_events() == events
    assert api.path.read_bytes() == before


def test_stale_anchor_after_ingestion_replacement_is_not_guessed(api: ExplorerAPI) -> None:
    request = {
        "title": "Synthetic context",
        "source": "synthetic:context",
        "text": "GraphRAG old evidence. " * 100,
    }
    first = api.client.post("/ingest/text", json=request).json()
    assert context(api, first["document_id"], first["chunk_ids"][0]).status_code == 200
    second = api.container.ingestion_pipeline.ingest_documents(
        [
            Document(
                document_id=first["document_id"],
                title=request["title"],
                source=request["source"],
                text="Changed source evidence.",
            )
        ]
    )
    assert_error(context(api, first["document_id"], first["chunk_ids"][0]), 404, "chunk_not_found")
    assert context(api, first["document_id"], second[0].chunk_id).status_code == 200


def test_storage_failure_is_sanitized_and_not_empty_context(api: ExplorerAPI) -> None:
    api.seed()
    with sqlite3.connect(api.path) as connection:
        connection.execute("DROP TABLE chunks")
    assert_error(context(api, chunk_id="a"), 503, "source_context_storage_unavailable")


def test_openapi_exposes_bounded_typed_context_and_existing_defaults(api: ExplorerAPI) -> None:
    schema = api.client.get("/openapi.json").json()
    operation = schema["paths"]["/documents/context"]["get"]
    assert {"200", "404", "409", "422", "503"} <= set(operation["responses"])
    parameters = {item["name"]: item for item in operation["parameters"]}
    for name in ("document_id", "chunk_id"):
        assert parameters[name]["in"] == "query" and parameters[name]["required"]
    assert parameters["document_id"]["schema"]["maxLength"] == 128
    assert parameters["chunk_id"]["schema"]["maxLength"] == 256
    for name in ("before", "after"):
        assert parameters[name]["schema"]["minimum"] == 0
        assert parameters[name]["schema"]["maximum"] == 5
        assert parameters[name]["schema"]["default"] == 2
    models = schema["components"]["schemas"]
    assert models["SourceContext"]["properties"]["chunks"]["maxItems"] == 11
    assert models["ContextChunk"]["properties"]["text"]["maxLength"] == 4000
    assert models["ContextChunk"]["properties"]["chunk_index"]["type"] == "integer"
    assert api.client.post("/documents/context").status_code == 405
    assert api.client.get("/documents").json()["next_cursor"] is None
