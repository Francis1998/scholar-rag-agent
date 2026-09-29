"""Acceptance and persistence contracts for generation-free saved-evidence drift."""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import NoReturn

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.evidence import text_digest
from api.application import create_app
from retrieval.models import Chunk, Document, SearchResult
from storage.corpus_drift import (
    MAX_CURRENT_BYTES,
    MAX_CURRENT_TEXT_BYTES,
    MAX_EVENT_BYTES,
    MAX_RUN_BYTES,
    MAX_RUN_EVENTS,
    ChunkDigests,
    CorpusDriftError,
    CorpusDriftReport,
    SQLiteCorpusDrift,
)
from storage.evidence_export import EvidenceExportError
from tests.test_run_comparison import ComparisonAPI, _save, _source


def _no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Drift inspection must not retrieve, generate, write, or use HTTP")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _no_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _no_work)


@pytest.fixture
def drift_api(tmp_path: Path) -> Iterator[ComparisonAPI]:
    path = tmp_path / "drift ?# \u7814\u7a76.sqlite3"
    application = create_app(offline_settings(path))
    with TestClient(application) as client:
        yield ComparisonAPI(application, client, application.state.container, path)


def _seed(api: ComparisonAPI, sources: list[SearchResult], run_id: str = "saved") -> None:
    _save(api, run_id, sources)
    api.container.document_store.add_documents(
        [Document(**source.chunk.model_dump(exclude={"chunk_id"})) for source in sources],
        [source.chunk for source in sources],
    )


def _report(api: ComparisonAPI, run_id: str = "saved") -> CorpusDriftReport:
    response = api.client.get(f"/runs/{run_id}/corpus-drift")
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    return CorpusDriftReport.model_validate_json(response.content)


def _error(api: ComparisonAPI, status: int, code: str, run_id: str = "saved") -> httpx.Response:
    response = api.client.get(f"/runs/{run_id}/corpus-drift")
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert set(response.json()) == {"detail"}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    return response


def _mutate(api: ComparisonAPI, sql: str, parameters: tuple[object, ...] = ()) -> None:
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(sql, parameters)


def _dump(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        return "\n".join(connection.iterdump())


def test_completed_query_has_an_unchanged_corpus_drift_report(tmp_path: Path) -> None:
    application = create_app(offline_settings(tmp_path / "drift.sqlite3"))
    with TestClient(application) as client:
        ingested = client.post(
            "/ingest/text",
            json={
                "title": "Synthetic evidence",
                "text": "GraphRAG connects synthetic research entities, not scientific findings.",
                "source": "synthetic:corpus-drift",
            },
        )
        assert ingested.status_code == 200
        query = client.post("/query", json={"query": "What does GraphRAG connect?"})
        assert query.status_code == 200
        run = query.json()["result"]
        assert run["state"] == "DONE"
        response = client.get(f"/runs/{run['run_id']}/corpus-drift")
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["run_id"] == run["run_id"]
        assert report["schema_version"] == "1.0"
        assert report["counts"] == {"total": 1, "unchanged": 1, "changed": 0, "missing": 0}
        assert report["sources"][0]["status"] == "unchanged"
        assert report["sources"][0]["changed_fields"] == []
        assert report["sources"][0]["frozen"] == report["sources"][0]["current"]
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"


def test_same_id_reingestion_compares_all_fields_in_frozen_order(
    drift_api: ComparisonAPI,
) -> None:
    sources = [
        _source("z", text="cafe\u0301 \u7814\u7a76\n"),
        _source("a"),
        _source("m"),
    ]
    _seed(drift_api, sources)
    replacement = sources[0].chunk.model_copy(
        update={
            "title": "Updated synthetic title",
            "text": "caf\u00e9 \u7814\u7a76\n",
            "source": "synthetic:replacement",
            "metadata": {"private-key": "synthetic-private-metadata"},
        }
    )
    drift_api.container.document_store.add_documents(
        [Document(**replacement.model_dump(exclude={"chunk_id"}))], [replacement]
    )
    report = _report(drift_api)
    assert report.counts.model_dump() == {"total": 3, "unchanged": 2, "changed": 1, "missing": 0}
    assert report.has_drift
    assert [source.chunk_id for source in report.sources] == ["z", "a", "m"]
    assert [source.rank for source in report.sources] == [1, 2, 3]
    first = report.sources[0]
    assert first.changed_fields == ["text", "title", "source", "metadata"]
    assert first.frozen == ChunkDigests.from_chunk(sources[0].chunk)
    assert first.current == ChunkDigests.from_chunk(replacement)
    assert first.missing_reason is None
    assert "synthetic-private-metadata" not in report.model_dump_json()
    assert "Updated synthetic title" not in report.model_dump_json()


@pytest.mark.parametrize("field", ["text", "title", "source", "metadata"])
def test_individual_fields_report_only_the_changed_field(
    drift_api: ComparisonAPI, field: str
) -> None:
    source = _source("one")
    _seed(drift_api, [source])
    changed = source.chunk.model_dump()
    changed[field] = {"value": "updated"} if field == "metadata" else "different"
    drift_api.container.document_store.add_documents([], [source.chunk.model_validate(changed)])
    result = _report(drift_api).sources[0]
    assert result.status == "changed"
    assert result.changed_fields == [field]


def test_canonical_metadata_ignores_spacing_order_and_unicode_escaping(
    drift_api: ComparisonAPI,
) -> None:
    _seed(drift_api, [_source("one", metadata={"z": "\u7814\u7a76", "a": "caf\u00e9"})])
    original = _report(drift_api)
    _mutate(
        drift_api,
        "UPDATE chunks SET metadata = ?",
        (' { "a" : "caf\\u00e9", "z": "\\u7814\\u7a76" } ',),
    )
    assert _report(drift_api) == original


def test_missing_documents_chunks_and_reassigned_ids_are_distinct(
    drift_api: ComparisonAPI,
) -> None:
    _seed(
        drift_api,
        [_source("document-gone"), _source("chunk-gone"), _source("reowned"), _source("same")],
    )
    _mutate(drift_api, "DELETE FROM documents WHERE document_id = 'doc-document-gone'")
    _mutate(drift_api, "DELETE FROM chunks WHERE chunk_id = 'chunk-gone'")
    _mutate(
        drift_api,
        "UPDATE chunks SET document_id = 'private-other-document' WHERE chunk_id = 'reowned'",
    )
    report = _report(drift_api)
    assert report.counts.missing == 3
    assert [source.missing_reason for source in report.sources] == [
        "document_missing",
        "chunk_missing",
        "chunk_reassigned",
        None,
    ]
    for source in report.sources[:3]:
        assert source.status == "missing"
        assert source.current is None
        assert source.changed_fields == []
    assert "private-other-document" not in report.model_dump_json()
    _mutate(drift_api, "DELETE FROM chunks WHERE chunk_id = 'document-gone'")
    assert _report(drift_api) == report


def test_current_only_rows_document_bodies_and_indexes_are_out_of_scope(
    drift_api: ComparisonAPI,
) -> None:
    _seed(drift_api, [_source("one")])
    before = _report(drift_api)
    _mutate(drift_api, "UPDATE documents SET title = 'changed', text = 'new document body'")
    _mutate(
        drift_api,
        "INSERT INTO chunks VALUES ('unselected', 'unselected', 'other', ?, 'other', 'bad json')",
        ("x" * (MAX_CURRENT_BYTES + 1),),
    )
    _mutate(drift_api, "DROP TABLE graph_chunks")
    assert _report(drift_api) == before


def test_reports_are_read_only_and_exports_survive_mutation_and_restart(
    drift_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = drift_api
    _seed(api, [_source("changed"), _source("missing"), _source("unchanged")])
    exports = {
        fmt: api.client.get("/runs/saved/export", params={"format": fmt}).content
        for fmt in ("json", "markdown")
    }
    events = api.container.event_log.list_events()
    _mutate(api, "UPDATE chunks SET text = 'updated synthetic text' WHERE chunk_id = 'changed'")
    _mutate(api, "DELETE FROM chunks WHERE chunk_id = 'missing'")
    restarted = create_app(offline_settings(api.path))
    for container in (api.container, restarted.state.container):
        for component, method in (
            (container.runner, "run"),
            (container.runner, "preview"),
            (container.llm, "generate"),
            (container.hybrid_retriever, "retrieve"),
            (container.document_store, "list_chunks"),
            (container.document_store, "add_documents"),
            (container.graph_store, "chunks_for_entities"),
            (container.event_log, "append_event"),
            (container.event_log, "append_transition"),
        ):
            monkeypatch.setattr(component, method, _no_work)
    before_dump, before_bytes = _dump(api.path), api.path.read_bytes()
    first = _report(api)
    with TestClient(restarted) as client:
        response = client.get("/runs/saved/corpus-drift")
        assert response.status_code == 200
        assert response.json() == first.model_dump(mode="json")
        for fmt, before in exports.items():
            assert client.get("/runs/saved/export", params={"format": fmt}).content == before
    assert api.container.event_log.list_events() == events
    assert _dump(api.path) == before_dump
    assert api.path.read_bytes() == before_bytes
    assert SQLiteCorpusDrift(api.path).report("saved") == first


def test_one_consistent_read_snapshot_even_during_concurrent_reingestion(
    drift_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(drift_api, [_source("first"), _source("second")])
    _mutate(drift_api, "PRAGMA journal_mode = WAL")
    original = SQLiteCorpusDrift._current
    calls = 0

    def read(
        connection: sqlite3.Connection,
        encoding: str,
        chunk_id: str,
        remaining_bytes: int,
        remaining_text_bytes: int,
    ) -> tuple[Chunk | None, int, int]:
        nonlocal calls
        result = original(connection, encoding, chunk_id, remaining_bytes, remaining_text_bytes)
        calls += 1
        if calls == 1:
            _mutate(drift_api, "UPDATE chunks SET title = 'concurrently replaced'")
            _mutate(drift_api, "DELETE FROM documents WHERE document_id = 'doc-second'")
        return result

    monkeypatch.setattr(SQLiteCorpusDrift, "_current", staticmethod(read))
    first = _report(drift_api)
    assert first.counts.unchanged == 2
    assert not first.has_drift
    second = _report(drift_api)
    assert second.counts.changed == second.counts.missing == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE chunks SET text = X'80'",
        "UPDATE chunks SET text = CAST(X'80' AS TEXT)",
        "UPDATE chunks SET title = X'80'",
        "UPDATE chunks SET source = X'80'",
        "UPDATE chunks SET document_id = X'80'",
        "UPDATE chunks SET metadata = X'80'",
        "UPDATE chunks SET metadata = 'not-json'",
        "UPDATE chunks SET metadata = '[]'",
        "UPDATE chunks SET metadata = 'null'",
        """UPDATE chunks SET metadata = '{"x": true}'""",
        """UPDATE chunks SET metadata = '{"x": 1}'""",
        """UPDATE chunks SET metadata = '{"x": NaN}'""",
        """UPDATE chunks SET metadata = '{"x": {}}'""",
        """UPDATE chunks SET metadata = '{"x": "one", "x": "two"}'""",
        """UPDATE chunks SET metadata = '{"x": "\\ud800"}'""",
    ],
)
def test_corrupt_selected_chunks_fail_instead_of_becoming_changes_or_missing(
    drift_api: ComparisonAPI, mutation: str
) -> None:
    _seed(drift_api, [_source("one")])
    _mutate(drift_api, mutation)
    _error(drift_api, 409, "invalid_corpus_record")
    _mutate(drift_api, "DELETE FROM documents")
    _error(drift_api, 409, "invalid_corpus_record")


def test_corruption_on_later_rank_never_returns_partial_success(drift_api: ComparisonAPI) -> None:
    _seed(drift_api, [_source("one"), _source("two")])
    _mutate(drift_api, "UPDATE chunks SET metadata = 'bad' WHERE chunk_id = 'two'")
    _error(drift_api, 409, "invalid_corpus_record")


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("DELETE FROM agent_events WHERE event_type = 'evidence_snapshot'", "snapshot_unavailable"),
        ("DELETE FROM agent_events WHERE id = 8", "run_incomplete"),
        (
            """UPDATE agent_events SET payload =
               '{"from_state":"ANSWERING","to_state":"ERROR","payload":{}}' WHERE id = 8""",
            "run_failed",
        ),
        ("UPDATE agent_events SET payload = 'bad json' WHERE id = 5", "invalid_run_record"),
        ("UPDATE agent_events SET payload = X'80' WHERE id = 5", "invalid_run_record"),
        ("UPDATE agent_events SET timestamp = CAST(X'80' AS TEXT)", "invalid_run_record"),
        (
            """UPDATE agent_events SET payload = json_set(payload, '$.schema_version', '2.0')
               WHERE event_type = 'evidence_snapshot'""",
            "invalid_run_record",
        ),
        (
            """UPDATE agent_events SET payload = json_set(
                   payload, '$.sources[0].text_sha256', 'bad')
               WHERE event_type = 'evidence_snapshot'""",
            "invalid_run_record",
        ),
    ],
)
def test_saved_run_errors_reuse_exporter_semantics(
    drift_api: ComparisonAPI, mutation: str, code: str
) -> None:
    _seed(drift_api, [_source("one")])
    _mutate(drift_api, mutation)
    _error(drift_api, 409, code)


def test_unknown_and_invalid_run_ids_have_explicit_private_errors(drift_api: ComparisonAPI) -> None:
    _error(drift_api, 404, "run_not_found", "unknown")
    _error(drift_api, 422, "invalid_drift_request", "a" * 257)
    for value in ("", "a" * 257, "\ud800"):
        with pytest.raises(CorpusDriftError) as exc:
            drift_api.container.corpus_drift.report(value)
        assert exc.value.code == "invalid_drift_request"


@pytest.mark.parametrize("table", ["agent_events", "documents", "chunks"])
def test_operational_errors_are_503_not_empty_results(
    drift_api: ComparisonAPI, table: str, caplog: pytest.LogCaptureFixture
) -> None:
    _seed(drift_api, [_source("one")])
    statements = {
        "agent_events": "DROP TABLE agent_events",
        "documents": "DROP TABLE documents",
        "chunks": "DROP TABLE chunks",
    }
    _mutate(drift_api, statements[table])
    with caplog.at_level(logging.WARNING, logger="api.corpus_drift"):
        response = _error(drift_api, 503, "drift_storage_unavailable")
    assert str(drift_api.path) not in response.text + caplog.text
    assert "no such table" not in response.text + caplog.text
    assert "Corpus drift failed: drift_storage_unavailable" in caplog.text


def test_readonly_service_never_creates_missing_storage(tmp_path: Path) -> None:
    path = tmp_path / "missing ?#.sqlite3"
    service = SQLiteCorpusDrift(path)
    assert not path.exists()
    with pytest.raises(CorpusDriftError) as exc:
        service.report("saved")
    assert exc.value.code == "drift_storage_unavailable"
    assert not path.exists()


def test_actual_locked_database_fails_explicitly(
    drift_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(drift_api, [_source("one")])
    connect = sqlite3.connect

    def immediate(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        return connect(database, uri=uri, timeout=0)

    with closing(connect(drift_api.path)) as writer:
        writer.execute("BEGIN EXCLUSIVE")
        monkeypatch.setattr(sqlite3, "connect", immediate)
        _error(drift_api, 503, "drift_storage_unavailable")
        writer.rollback()


def test_zero_and_fifty_sources_have_honest_bounded_counts(drift_api: ComparisonAPI) -> None:
    _seed(drift_api, [], "empty")
    empty = _report(drift_api, "empty")
    assert empty.counts.model_dump() == {"total": 0, "unchanged": 0, "changed": 0, "missing": 0}
    assert empty.sources == [] and not empty.has_drift
    _seed(drift_api, [_source(f"source-{index}") for index in range(50)])
    full = _report(drift_api)
    assert full.counts.total == full.counts.unchanged == 50
    assert [source.rank for source in full.sources] == list(range(1, 51))
    _mutate(
        drift_api,
        """UPDATE agent_events SET payload = json_insert(
               payload, '$.sources[#]', json_extract(payload, '$.sources[0]'))
           WHERE run_id = 'saved' AND event_type = 'evidence_snapshot'""",
    )
    _error(drift_api, 409, "invalid_run_record")


@pytest.mark.parametrize("length", [256, 257])
def test_frozen_identifiers_are_bounded_without_truncation(
    drift_api: ComparisonAPI, length: int
) -> None:
    identifier = "x" * length
    _seed(drift_api, [_source(identifier, document_id=identifier)])
    if length == 256:
        report = _report(drift_api)
        assert report.sources[0].chunk_id == report.sources[0].document_id == identifier
    else:
        _error(drift_api, 409, "invalid_run_record")


@pytest.mark.parametrize("sources", [1, 2])
def test_exact_current_utf8_text_budget_is_aggregate(
    drift_api: ComparisonAPI, sources: int
) -> None:
    _seed(drift_api, [_source(str(index)) for index in range(sources)])
    text = "\u00e9" * (MAX_CURRENT_TEXT_BYTES // 2 // sources)
    _mutate(drift_api, "UPDATE chunks SET text = ?", (text,))
    assert _report(drift_api).counts.changed == sources
    _mutate(drift_api, "UPDATE chunks SET text = text || 'x' WHERE chunk_id = '0'")
    _error(drift_api, 409, "drift_limit_exceeded")


def test_exact_current_record_byte_budget_includes_metadata(drift_api: ComparisonAPI) -> None:
    _seed(drift_api, [_source("one")])
    _mutate(drift_api, "UPDATE chunks SET metadata = '{}'")
    with closing(sqlite3.connect(drift_api.path)) as connection:
        row = connection.execute(
            "SELECT chunk_id, document_id, title, text, source, metadata FROM chunks"
        ).fetchone()
    used = sum(len(value.encode("utf-8")) for value in row)
    _mutate(drift_api, "UPDATE chunks SET metadata = ?", ("{}" + " " * (MAX_CURRENT_BYTES - used),))
    assert _report(drift_api).counts.changed == 1
    _mutate(drift_api, "UPDATE chunks SET metadata = metadata || ' '")
    _error(drift_api, 409, "drift_limit_exceeded")


def test_exact_event_count_cap_includes_post_completion_events(drift_api: ComparisonAPI) -> None:
    _seed(drift_api, [_source("one")])
    for _ in range(MAX_RUN_EVENTS - 8):
        drift_api.container.event_log.append_event("synthetic-agent", "saved", "diagnostic", {})
    assert _report(drift_api).counts.unchanged == 1
    drift_api.container.event_log.append_event("synthetic-agent", "saved", "diagnostic", {})
    _error(drift_api, 409, "drift_limit_exceeded")


def test_exact_single_event_byte_cap(drift_api: ComparisonAPI) -> None:
    _seed(drift_api, [_source("one")])
    event_id = drift_api.container.event_log.append_event(
        "synthetic-agent", "saved", "diagnostic", {}
    )
    _mutate(
        drift_api,
        "UPDATE agent_events SET payload = ? WHERE id = ?",
        ("{}" + " " * (MAX_EVENT_BYTES - 2), event_id),
    )
    assert _report(drift_api).counts.unchanged == 1
    _mutate(drift_api, "UPDATE agent_events SET payload = payload || ' ' WHERE id = ?", (event_id,))
    _error(drift_api, 409, "drift_limit_exceeded")


def test_exact_aggregate_event_byte_cap(drift_api: ComparisonAPI) -> None:
    _seed(drift_api, [_source("one")])
    ids = [
        drift_api.container.event_log.append_event("synthetic-agent", "saved", "diagnostic", {})
        for _ in range(8)
    ]
    with closing(sqlite3.connect(drift_api.path)) as connection, connection:
        used = connection.execute(
            """SELECT sum(length(CAST(timestamp AS BLOB)) + length(CAST(agent_id AS BLOB))
                    + length(CAST(run_id AS BLOB)) + length(CAST(event_type AS BLOB))
                    + length(CAST(payload AS BLOB))) FROM agent_events"""
        ).fetchone()[0]
        remaining = MAX_RUN_BYTES - used
        for event_id in ids:
            added = min(remaining, MAX_EVENT_BYTES - 2)
            connection.execute(
                "UPDATE agent_events SET payload = ? WHERE id = ?",
                ("{}" + " " * added, event_id),
            )
            remaining -= added
        assert remaining == 0
    assert _report(drift_api).counts.unchanged == 1
    _mutate(drift_api, "UPDATE agent_events SET payload = payload || ' ' WHERE id = ?", (ids[-1],))
    _error(drift_api, 409, "drift_limit_exceeded")


def test_oversized_rows_are_rejected_before_full_payload_fetch(
    drift_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(drift_api, [_source("one")])
    _mutate(drift_api, "UPDATE chunks SET metadata = ?", ("x" * (MAX_CURRENT_BYTES + 1),))
    queries: list[str] = []
    connections: list[str] = []
    connect = sqlite3.connect

    def traced(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        connections.append(database)
        assert uri is True
        assert database.endswith("?mode=ro")
        connection = connect(database, uri=uri, timeout=timeout)
        connection.set_trace_callback(queries.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced)
    _error(drift_api, 409, "drift_limit_exceeded")
    assert len(connections) == 1
    assert queries.count("BEGIN") == 1
    assert not any("SELECT CAST(chunk_id AS BLOB)" in query for query in queries)
    assert not any("SELECT *" in query for query in queries)


@pytest.mark.parametrize("encoding", ["UTF-16le", "UTF-16be"])
def test_sqlite_encoding_does_not_change_utf8_digests(tmp_path: Path, encoding: str) -> None:
    path = tmp_path / "utf16.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "PRAGMA encoding = 'UTF-16le'"
            if encoding == "UTF-16le"
            else "PRAGMA encoding = 'UTF-16be'"
        )
        connection.execute("CREATE TABLE encoding_anchor (id INTEGER)")
    application = create_app(offline_settings(path))
    with TestClient(application) as client:
        api = ComparisonAPI(application, client, application.state.container, path)
        _seed(api, [_source("one", text="caf\u00e9 \u7814\u7a76 \U0001f52c")])
        report = _report(api)
        assert report.sources[0].frozen.text_sha256 == text_digest(
            "caf\u00e9 \u7814\u7a76 \U0001f52c"
        )
        assert report.counts.unchanged == 1
        _mutate(api, "UPDATE chunks SET text = ?", ("\u00e9" * (MAX_CURRENT_TEXT_BYTES // 2),))
        assert _report(api).counts.changed == 1
        _mutate(api, "UPDATE chunks SET text = text || 'x'")
        _error(api, 409, "drift_limit_exceeded")


def test_errors_never_echo_saved_secrets(drift_api: ComparisonAPI) -> None:
    _seed(drift_api, [_source("one")])
    private = "synthetic-private-api-token"
    _mutate(drift_api, "UPDATE chunks SET metadata = ?", (f'{{"secret":{{"key":"{private}"}}}}',))
    response = _error(drift_api, 409, "invalid_corpus_record")
    assert private not in response.text
    with pytest.raises(CorpusDriftError) as exc:
        drift_api.container.corpus_drift.report("saved")
    assert private not in str(exc.value)


def test_openapi_exposes_bounded_readonly_contract(drift_api: ComparisonAPI) -> None:
    schema = drift_api.client.get("/openapi.json").json()
    route = schema["paths"]["/runs/{run_id}/corpus-drift"]
    assert set(route) == {"get"}
    assert {"200", "404", "409", "422", "503"} <= set(route["get"]["responses"])
    assert route["get"]["parameters"][0]["schema"]["maxLength"] == 256
    report = schema["components"]["schemas"]["CorpusDriftReport"]
    assert report["properties"]["sources"]["maxItems"] == 50
    source = schema["components"]["schemas"]["SourceDrift"]["properties"]
    assert source["rank"]["maximum"] == 50
    assert source["changed_fields"]["maxItems"] == 4
    assert set(source["status"]["enum"]) == {"unchanged", "changed", "missing"}


def test_reusable_python_service_uses_the_authoritative_exporter(drift_api: ComparisonAPI) -> None:
    _seed(drift_api, [_source("one")])
    assert drift_api.container.corpus_drift.report("saved") == _report(drift_api)
    _mutate(
        drift_api,
        "UPDATE agent_events SET payload = json_set(payload, '$.to_state', 'ERROR') WHERE id = 8",
    )
    with pytest.raises(EvidenceExportError) as exc:
        SQLiteCorpusDrift(drift_api.path).report("saved")
    assert exc.value.code == "run_failed"
