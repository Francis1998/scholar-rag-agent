"""Integrated exact-span annotations, isolated from mutable corpus and agent activity."""

import json
import logging
import socket
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from agent.evidence import EvidenceSnapshot, text_digest
from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from storage.evidence_annotations import (
    MAX_RECORD_TEXT_BYTES,
    MAX_RESPONSE_BYTES,
    AnnotationError,
    AnnotationPage,
    AnnotationSubmission,
    SavedEvidenceAnnotation,
    SQLiteEvidenceAnnotations,
)
from storage.evidence_export import MAX_EVENT_BYTES, MAX_RUN_EVENTS, EvidenceExportError
from tests.test_run_comparison import ComparisonAPI, _save, _source

TEXT = (
    "Caf\u00e9 \U0001f52c e\u0301. GraphRAG connects evidence. Again: GraphRAG connects evidence."
)
QUOTE = "GraphRAG connects evidence"
NOTE = "  Human opinion: inspect this exact occurrence, not a verified conclusion.\n"
PRIVATE = "private-note-or-storage-diagnostic"
URL = "/runs/saved/annotations"


def _no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Annotations must not perform agent, corpus, or network work")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    for target, name in (
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
    ):
        monkeypatch.setattr(target, name, _no_work)


@pytest.fixture
def annotation_api(tmp_path: Path) -> Iterator[ComparisonAPI]:
    path = tmp_path / "annotations ?# \u7814\u7a76.sqlite3"
    app = create_app(offline_settings(path))
    with TestClient(app) as client:
        api = ComparisonAPI(app, client, app.state.container, path)
        _save(api, "saved", [_source("chunk/one", document_id="paper/one", text=TEXT)])
        yield api


def _body(**changes: object) -> dict[str, Any]:
    start = TEXT.rindex(QUOTE)
    return {
        "annotation_id": str(uuid4()),
        "document_id": "paper/one",
        "chunk_id": "chunk/one",
        "source_text_sha256": text_digest(TEXT),
        "start": start,
        "end": start + len(QUOTE),
        "quote": QUOTE,
        "note": NOTE,
        **changes,
    }


def _created(api: ComparisonAPI, body: dict[str, Any] | None = None) -> httpx.Response:
    response = api.client.post(URL, json=_body() if body is None else body)
    assert response.status_code == 201, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    return response


def _error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert set(response.json()) == {"detail"}
    assert set(response.json()["detail"]) == {"code", "message"}
    assert response.json()["detail"]["code"] == code
    assert PRIVATE not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def _mutate(api: ComparisonAPI, sql: str, params: tuple[object, ...] = ()) -> None:
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(sql, params)


def _events(api: ComparisonAPI) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(api.path)) as connection:
        return connection.execute("SELECT * FROM agent_events ORDER BY id").fetchall()


def _forbid_work(container: AppContainer, monkeypatch: pytest.MonkeyPatch) -> None:
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
        (container.event_log, "list_events"),
    ):
        monkeypatch.setattr(component, method, _no_work)


def test_real_ingest_query_export_and_exact_annotation_contract(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = annotation_api
    ingested = api.client.post(
        "/ingest/text",
        json={"title": "Synthetic note", "source": "synthetic:annotations", "text": TEXT},
    )
    assert ingested.status_code == 200
    query = api.client.post("/query", json={"query": "What does GraphRAG connect?"})
    assert query.status_code == 200 and set(query.json()) == {"result"}
    result = query.json()["result"]
    assert result["state"] == "DONE"
    export_url = f"/runs/{result['run_id']}/export"
    exports = {
        fmt: api.client.get(export_url, params={"format": fmt}).content
        for fmt in ("json", "markdown")
    }
    source = json.loads(exports["json"])["snapshot"]["sources"][0]
    assert source["chunk"]["text"] == TEXT
    assert json.loads(exports["json"])["generation"]["provider"] == "fake"
    history = api.client.get("/runs").content
    events = _events(api)
    body = _body(
        document_id=source["chunk"]["document_id"],
        chunk_id=source["chunk"]["chunk_id"],
        source_text_sha256=source["text_sha256"],
    )
    path = f"/runs/{result['run_id']}/annotations"
    with monkeypatch.context() as guard:
        _forbid_work(api.container, guard)
        assert api.client.get(path).json() == {"annotations": [], "next_cursor": None}
        before = datetime.now(UTC).replace(microsecond=0)
        created = api.client.post(path, json=body)
        assert created.status_code == 201, created.text
        saved = SavedEvidenceAnnotation.model_validate_json(created.content)
        assert before <= saved.created_at <= datetime.now(UTC)
        assert saved.start == TEXT.rindex(QUOTE) != TEXT.index(QUOTE)
        assert TEXT[saved.start : saved.end] == saved.quote
        assert saved.note == NOTE
        assert set(created.json()) == {*body, "schema_version", "sequence", "run_id", "created_at"}
        assert saved.schema_version == "1.0" and saved.sequence > 0
        assert created.json()["created_at"].endswith("Z")
        assert created.headers["content-type"] == "application/json"
        assert created.content == saved.model_dump_json().encode("utf-8")
        replay = api.client.post(path, json=body)
        assert replay.status_code == 200 and replay.content == created.content
        listed = api.client.get(path)
        assert listed.json() == {"annotations": [created.json()], "next_cursor": None}
        assert listed.content == api.container.evidence_annotations.list_annotations(
            result["run_id"]
        ).model_dump_json().encode("utf-8")
    assert _events(api) == events
    assert api.client.get("/runs").content == history
    for fmt, data in exports.items():
        assert api.client.get(export_url, params={"format": fmt}).content == data


def test_repeated_quotes_unicode_codepoints_and_immutable_records(
    annotation_api: ComparisonAPI,
) -> None:
    api = annotation_api
    first_start = TEXT.index(QUOTE)
    first = _created(api, _body(start=first_start, end=first_start + len(QUOTE)))
    second = _created(api)
    symbol_start = TEXT.index("\U0001f52c")
    symbol = _created(api, _body(start=symbol_start, end=symbol_start + 1, quote="\U0001f52c"))
    combining_start = TEXT.index("\u0301")
    combining = _created(api, _body(start=combining_start, end=combining_start + 1, quote="\u0301"))
    assert [row["quote"] for row in api.client.get(URL).json()["annotations"]] == [
        "\u0301",
        "\U0001f52c",
        QUOTE,
        QUOTE,
    ]
    assert first.json()["start"] != second.json()["start"]
    assert second.json()["sequence"] > first.json()["sequence"]
    assert symbol.json()["end"] - symbol.json()["start"] == 1
    saved = SavedEvidenceAnnotation.model_validate_json(combining.content)
    with pytest.raises(ValidationError, match="frozen"):
        saved.note = "Replacement"
    byte_offset = len(TEXT[:symbol_start].encode("utf-8"))
    _error(
        api.client.post(
            URL, json=_body(start=byte_offset, end=byte_offset + 4, quote="\U0001f52c")
        ),
        422,
        "invalid_annotation_selector",
    )
    start = TEXT.index("e\u0301")
    _error(
        api.client.post(URL, json=_body(start=start, end=start + 2, quote="\u00e9")),
        422,
        "invalid_annotation_selector",
    )


@pytest.mark.parametrize(
    "change",
    [
        {"annotation_id": "invalid"},
        {"annotation_id": 1},
        {"annotation_id": None},
        {"document_id": ""},
        {"document_id": True},
        {"chunk_id": ""},
        {"chunk_id": 1},
        {"chunk_id": "x" * 257},
        {"document_id": "x" * 257},
        {"source_text_sha256": "a" * 63},
        {"source_text_sha256": "A" * 64},
        {"source_text_sha256": 0},
        {"start": -1},
        {"start": True},
        {"start": 1.0},
        {"start": "1"},
        {"end": False},
        {"end": 2.0},
        {"end": "2"},
        {"end": 2**63},
        {"end": 0},
        {"start": 5, "end": 5},
        {"quote": ""},
        {"quote": "x" * 1001},
        {"quote": 42},
        {"note": ""},
        {"note": " \t\n\u3000"},
        {"note": "bad\x00note"},
        {"note": "x" * 1001},
        {"note": True},
        {"note": None},
        {"author": "No authenticated author"},
        {"schema_version": "2.0"},
        {"sequence": 1},
        {"created_at": "2026-01-01T00:00:00Z"},
        {"run_id": "another"},
    ],
)
def test_invalid_requests_are_strict_sanitized_and_never_appended(
    annotation_api: ComparisonAPI, change: dict[str, object]
) -> None:
    body = _body(**change)
    with pytest.raises(ValidationError):
        AnnotationSubmission.model_validate(body)
    _error(annotation_api.client.post(URL, json=body), 422, "invalid_annotation_request")
    assert annotation_api.client.get(URL).json() == {"annotations": [], "next_cursor": None}


@pytest.mark.parametrize("field", list(AnnotationSubmission.model_fields))
def test_every_submission_field_is_required(annotation_api: ComparisonAPI, field: str) -> None:
    body = _body()
    del body[field]
    _error(annotation_api.client.post(URL, json=body), 422, "invalid_annotation_request")


@pytest.mark.parametrize("field", ["note", "quote", "document_id", "chunk_id"])
def test_surrogates_and_malformed_json_are_sanitized(
    annotation_api: ComparisonAPI, field: str
) -> None:
    for content in (json.dumps(_body(**{field: "\ud800"})).encode(), b'{"note":', b"\x80"):
        response = annotation_api.client.post(
            URL, content=content, headers={"Content-Type": "application/json"}
        )
        _error(response, 422, "invalid_annotation_request")


@pytest.mark.parametrize(
    ("change", "status", "code"),
    [
        ({"chunk_id": "not-in-this-run"}, 404, "annotation_source_not_found"),
        ({"chunk_id": "CHUNK/ONE"}, 404, "annotation_source_not_found"),
        ({"document_id": "wrong-paper"}, 422, "invalid_annotation_selector"),
        ({"source_text_sha256": "a" * 64}, 422, "invalid_annotation_selector"),
        ({"start": 0, "end": 1}, 422, "invalid_annotation_selector"),
        ({"end": len(TEXT) + 1}, 422, "invalid_annotation_selector"),
        ({"quote": QUOTE + " "}, 422, "invalid_annotation_selector"),
        ({"quote": QUOTE.lower()}, 422, "invalid_annotation_selector"),
    ],
)
def test_bad_exact_selectors(
    annotation_api: ComparisonAPI, change: dict[str, object], status: int, code: str
) -> None:
    api = annotation_api
    payload = AnnotationSubmission.model_validate(_body(**change))
    _error(api.client.post(URL, json=payload.model_dump(mode="json")), status, code)
    with pytest.raises(AnnotationError) as error:
        api.container.evidence_annotations.create("saved", payload)
    assert (error.value.status_code, error.value.code) == (status, code)
    assert api.client.get(URL).json()["annotations"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"note": NOTE + " "},
        {"quote": QUOTE + "."},
        {"start": 0},
        {"end": len(TEXT)},
        {"source_text_sha256": "a" * 64},
        {"document_id": "no-such-document"},
        {"chunk_id": "no-such-chunk"},
    ],
)
def test_same_uuid_changed_payload_conflicts_without_reselecting_or_overwriting(
    annotation_api: ComparisonAPI, change: dict[str, object]
) -> None:
    api = annotation_api
    body = _body()
    first = _created(api, body)
    retry = api.client.post(URL, json={**body, "annotation_id": body["annotation_id"].upper()})
    assert retry.status_code == 200 and retry.content == first.content
    _error(api.client.post(URL, json={**body, **change}), 409, "annotation_id_conflict")
    assert api.client.get(URL).json()["annotations"] == [first.json()]


def test_uuid_scope_pagination_default_bounds_and_insert_stability(
    annotation_api: ComparisonAPI,
) -> None:
    api = annotation_api
    store = api.container.evidence_annotations
    payload = AnnotationSubmission.model_validate(_body())
    oldest, created = store.create("saved", payload)
    assert created
    _save(api, "other", [_source("chunk/one", document_id="paper/one", text=TEXT)])
    other, created = store.create("other", payload)
    assert created and other.sequence > oldest.sequence and other.run_id == "other"
    for _ in range(100):
        store.create("saved", AnnotationSubmission.model_validate(_body()))
    assert len(api.client.get(URL).json()["annotations"]) == 20
    first_response = api.client.get(URL, params={"limit": 100})
    assert first_response.status_code == 200
    first = AnnotationPage.model_validate_json(first_response.content)
    assert len(first.annotations) == 100
    assert first.next_cursor == first.annotations[-1].sequence
    assert all(
        left.sequence > right.sequence
        for left, right in zip(first.annotations, first.annotations[1:], strict=False)
    )
    latest, _ = store.create("saved", AnnotationSubmission.model_validate(_body()))
    second_response = api.client.get(URL, params={"limit": 100, "cursor": first.next_cursor})
    second = AnnotationPage.model_validate_json(second_response.content)
    assert second.annotations == [oldest] and second.next_cursor is None
    assert latest.sequence > first.annotations[0].sequence
    assert store.list_annotations("saved", cursor=oldest.sequence).annotations == []
    assert store.list_annotations("other").annotations == [other]
    assert store.list_annotations("saved", limit=1, cursor=2**63 - 1).annotations == [latest]


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "limit=-1",
        "limit=1.0",
        "limit=1.5",
        "limit=true",
        "limit=+1",
        "limit=1e0",
        "limit=%EF%BC%91",
        "cursor=0",
        "cursor=-1",
        "cursor=1.0",
        "cursor=1.5",
        "cursor=true",
        "cursor=",
        f"cursor={2**63}",
        "limit=1&limit=2",
        "cursor=1&cursor=2",
        "unexpected=private-note-or-storage-diagnostic",
    ],
)
def test_http_page_parameters_are_strict(annotation_api: ComparisonAPI, query: str) -> None:
    _error(annotation_api.client.get(f"{URL}?{query}"), 422, "invalid_annotation_request")
    _error(
        annotation_api.client.post(f"{URL}?{query}", json=_body()),
        422,
        "invalid_annotation_request",
    )


@pytest.mark.parametrize(
    "options",
    [
        {"limit": True},
        {"limit": False},
        {"limit": 1.0},
        {"limit": "1"},
        {"limit": 0},
        {"limit": 101},
        {"cursor": True},
        {"cursor": 1.0},
        {"cursor": "1"},
        {"cursor": 0},
        {"cursor": 2**63},
    ],
)
def test_python_page_validation_precedes_io(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    store = annotation_api.container.evidence_annotations
    monkeypatch.setattr(sqlite3, "connect", _no_work)
    with pytest.raises(AnnotationError) as error:
        store.list_annotations("saved", **options)
    assert (error.value.status_code, error.value.code) == (422, "invalid_annotation_request")


@pytest.mark.parametrize("run_id", ["", None, True, 42, "x" * 257, "\ud800"])
def test_python_run_validation_precedes_io(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch, run_id: object
) -> None:
    store = annotation_api.container.evidence_annotations
    monkeypatch.setattr(sqlite3, "connect", _no_work)
    with pytest.raises(AnnotationError) as error:
        store.list_annotations(run_id)  # type: ignore[arg-type]
    assert error.value.status_code == 422
    with pytest.raises(AnnotationError) as error:
        store.create(run_id, AnnotationSubmission.model_validate(_body()))  # type: ignore[arg-type]
    assert error.value.status_code == 422


@pytest.mark.parametrize("field", ["start", "end", "quote", "note"])
def test_python_revalidates_unchecked_model_copies(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    payload = AnnotationSubmission.model_validate(_body()).model_copy(update={field: True})
    monkeypatch.setattr(sqlite3, "connect", _no_work)
    with pytest.raises(AnnotationError) as error:
        annotation_api.container.evidence_annotations.create("saved", payload)
    assert error.value.status_code == 422


def test_exact_imported_identities_are_not_normalized_or_restricted(
    annotation_api: ComparisonAPI,
) -> None:
    api = annotation_api
    identities = [
        "Paper",
        "paper",
        " Paper ",
        ".",
        "-",
        "../paper/a?x#y",
        "\u7814\u7a76/\U0001f52c",
        "control\x00and\nnewline",
        "\u7814" * 256,
    ]
    sources = [_source(identity, document_id=identity, text=TEXT) for identity in identities]
    _save(api, " imported run ", sources)
    path = f"/runs/{quote(' imported run ', safe='')}/annotations"
    for identifier in identities:
        body = _body(document_id=identifier, chunk_id=identifier)
        response = api.client.post(path, json=body)
        assert response.status_code == 201, response.text
        assert response.json()["document_id"] == response.json()["chunk_id"] == identifier
        assert api.client.post(path, json=body).content == response.content
    listed = api.client.get(path)
    assert listed.status_code == 200
    assert [row["chunk_id"] for row in listed.json()["annotations"]] == identities[::-1]


def test_exact_quote_and_note_limits_and_whitespace_quote(annotation_api: ComparisonAPI) -> None:
    api = annotation_api
    text = "\U0001f52c" * 1000 + " \t\n\x00"
    _save(api, "bounds", [_source("chunk/one", document_id="paper/one", text=text)])
    body = _body(
        start=0,
        end=1000,
        quote=text[:1000],
        source_text_sha256=text_digest(text),
        note="\U0001f52c" * 1000,
    )
    response = api.client.post("/runs/bounds/annotations", json=body)
    assert response.status_code == 201
    assert response.json()["quote"] == response.json()["note"] == "\U0001f52c" * 1000
    assert len(response.content) < MAX_RESPONSE_BYTES
    whitespace = _body(
        start=1000, end=len(text), quote=text[1000:], source_text_sha256=text_digest(text)
    )
    assert api.client.post("/runs/bounds/annotations", json=whitespace).status_code == 201


def test_current_corpus_edits_deletion_restart_and_no_runtime_activity(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = annotation_api
    source = _source("chunk/one", document_id="paper/one", text=TEXT)
    api.container.document_store.add_documents(
        [Document(**source.chunk.model_dump(exclude={"chunk_id"}))], [source.chunk]
    )
    body = _body()
    first = _created(api, body)
    before = api.client.get(URL).content
    events = _events(api)
    _mutate(api, "UPDATE chunks SET text = 'Replaced current text'")
    assert api.client.get(URL).content == before
    _mutate(api, "DELETE FROM chunks")
    _mutate(api, "DELETE FROM documents")
    restarted = create_app(offline_settings(api.path))
    _forbid_work(api.container, monkeypatch)
    _forbid_work(restarted.state.container, monkeypatch)
    unchanged_bytes = api.path.read_bytes()
    with TestClient(restarted) as client:
        assert client.get(URL).content == before
        replay = client.post(URL, json=body)
        assert replay.status_code == 200 and replay.content == first.content
    assert api.path.read_bytes() == unchanged_bytes
    _mutate(api, "DROP TABLE chunks")
    _mutate(api, "DROP TABLE documents")
    _mutate(api, "DROP TABLE graph_chunks")
    store = SQLiteEvidenceAnnotations(api.path)
    assert store.list_annotations("saved").model_dump_json().encode() == before
    saved, created = store.create("saved", AnnotationSubmission.model_validate(_body()))
    assert created and saved.sequence > first.json()["sequence"]
    assert _events(api) == events


def test_concurrent_identical_retries_append_once(annotation_api: ComparisonAPI) -> None:
    store = annotation_api.container.evidence_annotations
    payload = AnnotationSubmission.model_validate(_body())
    with ThreadPoolExecutor(max_workers=6) as workers:
        results = list(workers.map(lambda _: store.create("saved", payload), range(12)))
    assert sum(created for _, created in results) == 1
    assert all(record == results[0][0] for record, _ in results)
    assert store.list_annotations("saved").annotations == [results[0][0]]


def test_concurrent_changed_payload_has_one_winner(annotation_api: ComparisonAPI) -> None:
    store = annotation_api.container.evidence_annotations
    body = _body()
    payloads = [
        AnnotationSubmission.model_validate({**body, "note": f"Human opinion {index}"})
        for index in range(2)
    ]

    def write(payload: AnnotationSubmission) -> SavedEvidenceAnnotation | str:
        try:
            record, created = store.create("saved", payload)
            assert created
            return record
        except AnnotationError as error:
            assert error.status_code == 409
            return error.code

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(write, payloads))
    assert results.count("annotation_id_conflict") == 1
    assert len(store.list_annotations("saved").annotations) == 1


@pytest.mark.parametrize(
    ("sql", "code", "status"),
    [
        ("DELETE FROM agent_events", "run_not_found", 404),
        ("DELETE FROM agent_events WHERE id = 8", "run_incomplete", 409),
        (
            "UPDATE agent_events SET payload = json_set(payload, '$.to_state', 'ERROR') "
            "WHERE id = 8",
            "run_failed",
            409,
        ),
        ("DELETE FROM agent_events WHERE id = 5", "snapshot_unavailable", 409),
        (
            "UPDATE agent_events SET payload = json_set(payload, '$.schema_version', '99') "
            "WHERE id = 5",
            "invalid_run_record",
            409,
        ),
        (
            "UPDATE agent_events SET payload = json_set(payload, "
            "'$.sources[0].chunk.text', 'private-note-or-storage-diagnostic') WHERE id = 5",
            "invalid_run_record",
            409,
        ),
        (
            "UPDATE agent_events SET payload = 'invalid JSON' WHERE id = 5",
            "invalid_run_record",
            409,
        ),
        ("UPDATE agent_events SET payload = X'80' WHERE id = 5", "invalid_run_record", 409),
        (
            "UPDATE agent_events SET payload = CAST(X'80' AS TEXT) WHERE id = 5",
            "invalid_run_record",
            409,
        ),
        (
            "UPDATE agent_events SET payload = json_set(payload, '$.sources[0].chunk.document_id', "
            "zeroblob(257)) WHERE id = 5",
            "invalid_run_record",
            409,
        ),
    ],
)
def test_unknown_failed_incomplete_legacy_or_corrupt_runs_fail_explicitly(
    annotation_api: ComparisonAPI, sql: str, code: str, status: int
) -> None:
    api = annotation_api
    body = _body()
    _created(api, body)
    if "zeroblob" in sql:
        sql = sql.replace("zeroblob(257)", "'" + "x" * 257 + "'")
    _mutate(api, sql)
    _error(api.client.get(URL), status, code)
    _error(api.client.post(URL, json=body), status, code)
    _error(api.client.post(URL, json=_body()), status, code)
    with pytest.raises(EvidenceExportError) as error:
        api.container.evidence_annotations.list_annotations("saved")
    assert (error.value.status_code, error.value.code) == (status, code)


def test_no_source_or_unknown_run_are_not_empty_success(annotation_api: ComparisonAPI) -> None:
    api = annotation_api
    _error(api.client.get("/runs/unknown/annotations"), 404, "run_not_found")
    _error(api.client.post("/runs/unknown/annotations", json=_body()), 404, "run_not_found")
    _save(api, "empty", [])
    assert api.client.get("/runs/empty/annotations").json() == {
        "annotations": [],
        "next_cursor": None,
    }
    _error(
        api.client.post("/runs/empty/annotations", json=_body()),
        404,
        "annotation_source_not_found",
    )


def test_internally_rehashed_source_does_not_relocate_an_old_annotation(
    annotation_api: ComparisonAPI,
) -> None:
    api = annotation_api
    body = _body()
    _created(api, body)
    bundle = api.container.evidence_exporter.export("saved")
    source = _source("chunk/one", document_id="paper/one", text="!" + TEXT)
    replaced = EvidenceSnapshot.capture(bundle.snapshot.request.prompt, [source])
    _mutate(
        api,
        "UPDATE agent_events SET payload = ? WHERE event_type = 'evidence_snapshot'",
        (replaced.model_dump_json(),),
    )
    _error(api.client.get(URL), 409, "invalid_annotation_record")
    _error(api.client.post(URL, json=body), 409, "invalid_annotation_record")
    _error(api.client.post(URL, json=_body()), 422, "invalid_annotation_selector")
    with closing(sqlite3.connect(api.path)) as connection:
        assert connection.execute(
            "SELECT start, end, source_text_sha256 FROM evidence_annotations"
        ).fetchall() == [(body["start"], body["end"], body["source_text_sha256"])]


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE evidence_annotations SET schema_version = '99'",
        "UPDATE evidence_annotations SET schema_version = X'80'",
        "UPDATE evidence_annotations SET sequence = 0",
        "UPDATE evidence_annotations SET annotation_id = 'not-a-uuid'",
        "UPDATE evidence_annotations SET annotation_id = upper(annotation_id)",
        "UPDATE evidence_annotations SET document_id = 'no-such-document'",
        "UPDATE evidence_annotations SET chunk_id = 'no-such-chunk'",
        "UPDATE evidence_annotations SET source_text_sha256 = 'invalid'",
        "UPDATE evidence_annotations SET start = -1",
        "UPDATE evidence_annotations SET start = end",
        "UPDATE evidence_annotations SET start = 1.5",
        "UPDATE evidence_annotations SET end = 'invalid'",
        "UPDATE evidence_annotations SET end = 10000",
        "UPDATE evidence_annotations SET quote = 'different quote'",
        "UPDATE evidence_annotations SET quote = X'80'",
        "UPDATE evidence_annotations SET note = ''",
        "UPDATE evidence_annotations SET note = '  '",
        "UPDATE evidence_annotations SET note = char(0)",
        "UPDATE evidence_annotations SET note = CAST(X'80' AS TEXT)",
        "UPDATE evidence_annotations SET note = X'80'",
        "UPDATE evidence_annotations SET created_at = '2026-01-01'",
        "UPDATE evidence_annotations SET created_at = '2026-01-01T00:00:00+00:00'",
        "UPDATE evidence_annotations SET created_at = 'not-a-dateZ'",
    ],
)
def test_malformed_annotation_rows_are_never_silently_skipped(
    annotation_api: ComparisonAPI, sql: str
) -> None:
    api = annotation_api
    _created(api)
    _mutate(api, sql)
    _error(api.client.get(URL), 409, "invalid_annotation_record")
    with pytest.raises(AnnotationError, match="invalid, unsupported, or inconsistent"):
        api.container.evidence_annotations.list_annotations("saved")


def test_corrupt_lookahead_is_validated_before_returning_a_cursor(
    annotation_api: ComparisonAPI,
) -> None:
    api = annotation_api
    oldest = _created(api)
    _created(api)
    _mutate(
        api,
        "UPDATE evidence_annotations SET note = ? WHERE sequence = ?",
        ("x" * 1001, oldest.json()["sequence"]),
    )
    _error(api.client.get(URL, params={"limit": 1}), 409, "invalid_annotation_record")
    body = _body(annotation_id=oldest.json()["annotation_id"])
    _error(api.client.post(URL, json=body), 409, "invalid_annotation_record")


def _trace_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[sqlite3.Connection]]:
    connect = sqlite3.connect
    queries: list[str] = []
    connections: list[sqlite3.Connection] = []

    def traced(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        connection = connect(database, uri=uri, timeout=timeout, check_same_thread=False)
        connection.set_trace_callback(queries.append)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced)
    return queries, connections


def test_oversized_annotations_fail_before_hydration(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = annotation_api
    body = _body()
    _created(api, body)
    _mutate(api, "UPDATE evidence_annotations SET note = ?", ("x" * (MAX_RECORD_TEXT_BYTES + 1),))
    queries, connections = _trace_connections(monkeypatch)
    _error(api.client.get(URL), 409, "invalid_annotation_record")
    _error(api.client.post(URL, json=body), 409, "invalid_annotation_record")
    assert not any("SELECT sequence, start, end," in query for query in queries)
    assert any("typeof(sequence)" in query for query in queries)
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_oversized_event_payload_fails_before_hydration(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = annotation_api
    _mutate(api, "UPDATE agent_events SET payload = ? WHERE id = 5", ("x" * (MAX_EVENT_BYTES + 1),))
    queries, _ = _trace_connections(monkeypatch)
    _error(api.client.get(URL), 409, "evidence_read_limit_exceeded")
    _error(api.client.post(URL, json=_body()), 409, "evidence_read_limit_exceeded")
    assert not any("SELECT id, CAST(timestamp AS BLOB)" in query for query in queries)
    assert not any("FROM evidence_annotations" in query for query in queries)


def test_event_count_limit_is_inclusive(annotation_api: ComparisonAPI) -> None:
    api = annotation_api
    for _ in range(MAX_RUN_EVENTS - 8):
        api.container.event_log.append_event("synthetic-agent", "saved", "diagnostic", {})
    assert api.client.get(URL).status_code == 200
    _created(api)
    api.container.event_log.append_event("synthetic-agent", "saved", "diagnostic", {})
    _error(api.client.get(URL), 409, "evidence_read_limit_exceeded")
    _error(api.client.post(URL, json=_body()), 409, "evidence_read_limit_exceeded")


def test_wire_response_byte_bound_is_exact_utf8_not_character_count(
    annotation_api: ComparisonAPI,
) -> None:
    api = annotation_api
    text = "\U0001f52c" * 1000
    _save(api, "wide", [_source("chunk/one", document_id="paper/one", text=text)])
    store = api.container.evidence_annotations
    records = [
        store.create(
            "wide",
            AnnotationSubmission.model_validate(
                _body(start=0, end=1000, quote=text, source_text_sha256=text_digest(text), note="x")
            ),
        )[0]
        for _ in range(33)
    ]
    url = "/runs/wide/annotations?limit=100"
    baseline = api.client.get(url)
    assert baseline.status_code == 200
    remaining = MAX_RESPONSE_BYTES - len(baseline.content)
    assert 0 < remaining < 3999 * len(records)
    partial: tuple[int, str] | None = None
    for record in records:
        added = min(remaining, 3999)
        units, remainder = divmod(added + 1, 4)
        note = "\U0001f52c" * units + ("", "x", "\u00e9", "\u7814")[remainder]
        assert len(note) <= 1000
        _mutate(
            api,
            "UPDATE evidence_annotations SET note = ? WHERE sequence = ?",
            (note, record.sequence),
        )
        remaining -= added
        if len(note) < 1000:
            partial = record.sequence, note
    assert remaining == 0 and partial is not None
    exact = api.client.get(url)
    assert exact.status_code == 200
    assert len(exact.content) == int(exact.headers["content-length"]) == MAX_RESPONSE_BYTES
    assert len(exact.text) < MAX_RESPONSE_BYTES
    assert exact.content == store.list_annotations("wide", limit=100).model_dump_json().encode()
    _mutate(
        api,
        "UPDATE evidence_annotations SET note = ? WHERE sequence = ?",
        (partial[1] + "x", partial[0]),
    )
    _error(api.client.get(url), 413, "annotation_response_too_large")
    assert api.client.get("/runs/wide/annotations?limit=1").status_code == 200
    with pytest.raises(AnnotationError) as error:
        store.list_annotations("wide", limit=100)
    assert error.value.status_code == 413


def test_read_is_bounded_readonly_and_connections_close(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = annotation_api
    _created(api)
    _forbid_work(api.container, monkeypatch)
    before = api.path.read_bytes()
    queries, connections = _trace_connections(monkeypatch)
    response = api.client.get(URL, params={"limit": 1})
    assert response.status_code == 200
    assert len(connections) == 1 and queries.count("BEGIN") == 1
    selects = [query for query in queries if query.lstrip().startswith("SELECT")]
    assert len(selects) == 4
    assert all("LIMIT" in query for query in selects)
    assert not any(
        word in query for word in ("INSERT", "UPDATE", "DELETE", "CREATE") for query in queries
    )
    assert api.path.read_bytes() == before
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_evidence_and_annotations_share_one_consistent_read_snapshot(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = annotation_api
    first = _created(api).json()
    prompt = api.container.evidence_exporter.export("saved").snapshot.request.prompt
    replacement = EvidenceSnapshot.capture(
        prompt, [_source("chunk/one", document_id="paper/one", text="!" + TEXT)]
    )
    _mutate(api, "PRAGMA journal_mode = WAL")
    connect = sqlite3.connect
    changed = False

    def concurrent_write(query: str) -> None:
        nonlocal changed
        if not changed and "SELECT sequence, start, end," in query:
            changed = True
            with closing(connect(api.path)) as writer, writer:
                writer.execute(
                    "UPDATE agent_events SET payload = ? WHERE event_type = 'evidence_snapshot'",
                    (replacement.model_dump_json(),),
                )
                writer.execute(
                    "UPDATE evidence_annotations SET start = start + 1, end = end + 1, "
                    "source_text_sha256 = ?, note = 'Concurrently edited'",
                    (text_digest("!" + TEXT),),
                )

    def traced(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        connection = connect(database, uri=uri, timeout=timeout)
        connection.set_trace_callback(concurrent_write)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced)
    assert api.client.get(URL).json() == {"annotations": [first], "next_cursor": None}
    assert changed
    next_read = api.client.get(URL)
    assert next_read.status_code == 200
    assert next_read.json()["annotations"][0]["note"] == "Concurrently edited"
    assert next_read.json()["annotations"][0]["start"] == first["start"] + 1


@pytest.mark.parametrize("encoding", ["UTF-16le", "UTF-16be"])
def test_utf16_sqlite_preserves_exact_utf8_api_records(tmp_path: Path, encoding: str) -> None:
    path = tmp_path / "utf16.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "PRAGMA encoding = 'UTF-16le'"
            if encoding == "UTF-16le"
            else "PRAGMA encoding = 'UTF-16be'"
        )
        connection.execute("CREATE TABLE encoding_anchor (id INTEGER)")
    app = create_app(offline_settings(path))
    with TestClient(app) as client:
        api = ComparisonAPI(app, client, app.state.container, path)
        _save(api, "saved", [_source("chunk/one", document_id="paper/one", text=TEXT)])
        body = _body(note="\u7814\u7a76 \U0001f52c")
        created = _created(api, body)
        assert client.post(URL, json=body).content == created.content
        assert client.get(URL).json()["annotations"] == [created.json()]
        _mutate(api, "UPDATE evidence_annotations SET note = ?", ("\u7814" * 6000,))
        _error(client.get(URL), 409, "invalid_annotation_record")


@pytest.mark.parametrize("table", ["agent_events", "evidence_annotations"])
def test_missing_tables_are_sanitized_storage_errors(
    annotation_api: ComparisonAPI, caplog: pytest.LogCaptureFixture, table: str
) -> None:
    api = annotation_api
    _created(api)
    _mutate(api, "DROP TABLE " + table)
    with caplog.at_level(logging.WARNING, logger="api.annotations"):
        _error(api.client.get(URL), 503, "annotation_storage_unavailable")
        _error(api.client.post(URL, json=_body()), 503, "annotation_storage_unavailable")
    assert "no such table" not in caplog.text and str(api.path) not in caplog.text
    assert "Evidence annotation failed: annotation_storage_unavailable" in caplog.text


def test_sqlite_diagnostics_never_escape_and_missing_db_is_not_created(
    annotation_api: ComparisonAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    absent = tmp_path / "absent.sqlite3"
    with pytest.raises(AnnotationError) as error:
        SQLiteEvidenceAnnotations(absent)
    assert error.value.status_code == 503 and not absent.exists()

    def broken(*args: object, **kwargs: object) -> NoReturn:
        raise sqlite3.OperationalError(PRIVATE + str(annotation_api.path))

    monkeypatch.setattr(sqlite3, "connect", broken)
    with caplog.at_level(logging.WARNING, logger="api.annotations"):
        _error(annotation_api.client.get(URL), 503, "annotation_storage_unavailable")
        _error(annotation_api.client.post(URL, json=_body()), 503, "annotation_storage_unavailable")
    assert PRIVATE not in caplog.text and str(annotation_api.path) not in caplog.text


def test_failed_append_rolls_back_without_consuming_a_record(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = annotation_api
    store = api.container.evidence_annotations
    original_read = store._read
    reads = 0

    def fail_after_insert(*args: object, **kwargs: object) -> list[SavedEvidenceAnnotation]:
        nonlocal reads
        reads += 1
        if reads == 2:
            raise sqlite3.OperationalError(PRIVATE)
        return original_read(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as guard:
        guard.setattr(store, "_read", fail_after_insert)
        _error(api.client.post(URL, json=_body()), 503, "annotation_storage_unavailable")
    assert store.list_annotations("saved").annotations == []
    assert _created(api).json()["sequence"] == 1


def test_locked_storage_fails_explicitly_and_recovers_after_unlock(
    annotation_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = annotation_api
    connect = sqlite3.connect

    def short_timeout(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        return connect(database, uri=uri, timeout=0.01)

    with closing(connect(api.path)) as blocker:
        blocker.execute("BEGIN EXCLUSIVE")
        monkeypatch.setattr(sqlite3, "connect", short_timeout)
        _error(api.client.get(URL), 503, "annotation_storage_unavailable")
        _error(api.client.post(URL, json=_body()), 503, "annotation_storage_unavailable")
        blocker.rollback()
    assert api.client.get(URL).json() == {"annotations": [], "next_cursor": None}
    _created(api)


def test_documented_openapi_contract(annotation_api: ComparisonAPI) -> None:
    schema = annotation_api.client.get("/openapi.json").json()
    path = schema["paths"]["/runs/{run_id}/annotations"]
    assert set(path) == {"get", "post"}
    assert set(path["post"]["responses"]) == {"200", "201", "404", "409", "413", "422", "503"}
    assert set(path["get"]["responses"]) == {"200", "404", "409", "413", "422", "503"}
    submission = schema["components"]["schemas"]["AnnotationSubmission"]
    assert set(submission["required"]) == set(AnnotationSubmission.model_fields)
    assert submission["additionalProperties"] is False
    assert submission["properties"]["quote"]["maxLength"] == 1000
    assert submission["properties"]["note"]["maxLength"] == 1000
    page_options = {field["name"]: field for field in path["get"]["parameters"]}
    assert page_options["limit"]["schema"]["maximum"] == 100
    assert page_options["limit"]["schema"]["default"] == 20
