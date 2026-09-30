"""Exact cited-only, bounded, read-only saved bibliography contracts."""

import json
import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import NoReturn
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.models import AgentAnswer, Citation, Claim
from api.application import create_app
from retrieval.bibtex_export import BibTeXExporter
from retrieval.models import Document, SearchResult
from storage.evidence_export import (
    MAX_EVENT_BYTES,
    MAX_RUN_BYTES,
    MAX_RUN_EVENTS,
    EvidenceExportError,
)
from storage.saved_bibliography import (
    BIBLIOGRAPHIC_FIELDS,
    MAX_RESPONSE_BYTES,
    BibliographyError,
    SavedBibliography,
    SQLiteSavedBibliography,
)
from tests.test_run_comparison import ComparisonAPI, _save, _source


def _no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Bibliography must not retrieve, generate, use HTTP, or write")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _no_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _no_work)


@pytest.fixture
def bibliography_api(tmp_path: Path) -> Iterator[ComparisonAPI]:
    path = tmp_path / "bibliography ?# \u7814\u7a76.sqlite3"
    application = create_app(offline_settings(path))
    with TestClient(application) as client:
        yield ComparisonAPI(application, client, application.state.container, path)


def _answer(sources: list[SearchResult]) -> AgentAnswer:
    return AgentAnswer(
        answer="Synthetic saved answer, not a scientific finding.",
        claims=[Claim(text="Synthetic evidence.", chunk_ids=[s.chunk.chunk_id for s in sources])],
        citations=[
            Citation(
                chunk_id=source.chunk.chunk_id,
                document_id=source.chunk.document_id,
                title=source.chunk.title,
                snippet=source.chunk.text[:240],
            )
            for source in sources
        ],
    )


def _get(api: ComparisonAPI, run_id: str = "saved", format: str = "json") -> httpx.Response:
    return api.client.get(f"/runs/{quote(run_id, safe='')}/bibliography", params={"format": format})


def _report(api: ComparisonAPI, run_id: str = "saved") -> SavedBibliography:
    response = _get(api, run_id)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-type"] == "application/json"
    assert response.headers["content-disposition"] == 'attachment; filename="bibliography.json"'
    result = SavedBibliography.model_validate_json(response.content)
    assert response.content == result.to_json().encode("utf-8")
    return result


def _error(
    api: ComparisonAPI, status: int, code: str, run_id: str = "saved", format: str = "json"
) -> httpx.Response:
    response = _get(api, run_id, format)
    assert response.status_code == status, response.text
    assert set(response.json()) == {"detail"}
    assert set(response.json()["detail"]) == {"code", "message"}
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-disposition" not in response.headers
    return response


def _mutate(api: ComparisonAPI, sql: str, parameters: tuple[object, ...] = ()) -> None:
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(sql, parameters)


def test_real_ingest_query_and_default_bibtex_keep_existing_contracts(
    bibliography_api: ComparisonAPI,
) -> None:
    api = bibliography_api
    ingested = api.client.post(
        "/ingest/text",
        json={
            "title": "Synthetic GraphRAG note",
            "text": "GraphRAG connects synthetic research evidence, not scientific findings.",
            "source": "synthetic:bibliography",
        },
    )
    assert ingested.status_code == 200
    assert set(ingested.json()) == {"document_id", "chunk_ids"}
    query = api.client.post("/query", json={"query": "What does GraphRAG connect?"})
    assert query.status_code == 200
    assert set(query.json()) == {"result"}
    result = query.json()["result"]
    assert result["state"] == "DONE"
    run_id = result["run_id"]
    report = _report(api, run_id)
    evidence = api.client.get(f"/runs/{run_id}/export").json()
    assert report.schema_version == report.evidence_schema_version == "1.0"
    assert report.run_id == run_id
    assert report.context_sha256 == evidence["snapshot"]["context_sha256"]
    assert len(report.sources) == 1
    source = report.sources[0]
    assert source.document_id == ingested.json()["document_id"]
    assert source.title == "Synthetic GraphRAG note"
    assert source.metadata == {}
    assert source.cited_chunks[0].citation_numbers == [1]
    assert report.bibtex.startswith("@misc{")
    assert "author =" not in report.bibtex and "year =" not in report.bibtex
    assert "doi =" not in report.bibtex and "url =" not in report.bibtex
    default = api.client.get(f"/runs/{run_id}/bibliography")
    assert default.status_code == 200
    assert default.content == report.bibtex.encode("utf-8")
    assert default.content == _get(api, run_id, "bibtex").content
    assert default.headers["content-type"] == "application/x-bibtex; charset=utf-8"
    assert default.headers["content-disposition"] == 'attachment; filename="bibliography.bib"'
    assert default.headers["cache-control"] == "no-store"
    assert default.headers["x-content-type-options"] == "nosniff"
    assert report == api.container.saved_bibliography.export(run_id)
    assert evidence["answer"] == result["answer"]
    assert set(evidence["configuration"]) == {
        "max_source_docs",
        "max_hops",
        "retrieval_timeout_seconds",
        "reasoning_timeout_seconds",
    }
    assert any("not truth or entailment" in warning for warning in report.warnings)


def test_only_final_citations_in_frozen_rank_order_with_exact_chunk_mapping(
    bibliography_api: ComparisonAPI,
) -> None:
    sources = [
        _source("uncited", document_id="same-paper", title="Must not supply metadata"),
        _source(
            "z-first", document_id="same-paper", title="Chosen title", metadata={"year": "2024"}
        ),
        _source("middle", document_id="other-paper"),
        _source(
            "a-last", document_id="same-paper", title="Chosen title", metadata={"year": "2024"}
        ),
        _source("unused-paper"),
    ]
    answer = _answer([sources[3], sources[2], sources[1], sources[3]])
    _save(bibliography_api, "saved", sources, answer=answer, proposed=[["unused-paper", "missing"]])
    report = _report(bibliography_api)
    assert [source.document_id for source in report.sources] == ["same-paper", "other-paper"]
    assert [source.first_evidence_rank for source in report.sources] == [2, 3]
    assert report.sources[0].cited_chunks[0].model_dump() == {
        "chunk_id": "z-first",
        "evidence_rank": 2,
        "citation_numbers": [3],
    }
    assert report.sources[0].cited_chunks[1].model_dump() == {
        "chunk_id": "a-last",
        "evidence_rank": 4,
        "citation_numbers": [1, 4],
    }
    assert report.sources[1].cited_chunks[0].citation_numbers == [2]
    assert report.bibtex == BibTeXExporter().export_chunks([sources[1].chunk, sources[2].chunk])
    assert "Must not supply metadata" not in report.to_json()
    assert "unused-paper" not in report.to_json()
    assert not any("Conflicting" in warning for warning in report.warnings)


@pytest.mark.parametrize("shared_doi", [False, True])
def test_preserves_distinct_case_punctuation_and_whitespace_document_ids(
    bibliography_api: ComparisonAPI,
    shared_doi: bool,
) -> None:
    identities = ["Paper", "paper", " Paper ", "paper!", ".", "-", "ref_2", "paper_2"]
    sources = [
        _source(
            f"chunk-{index}",
            document_id=identity,
            metadata={"doi": "10.0000/same-synthetic-doi"} if shared_doi else {},
        )
        for index, identity in enumerate(identities)
    ]
    _save(bibliography_api, "saved", sources, answer=_answer(list(reversed(sources))))
    report = _report(bibliography_api)
    assert [source.document_id for source in report.sources] == identities
    keys = re.findall(r"^@\w+\{([^,]+),", report.bibtex, re.MULTILINE)
    assert len(keys) == len(set(keys)) == len(identities)
    assert all(re.fullmatch(r"[a-z0-9_]+", key) for key in keys)
    assert report.bibtex == BibTeXExporter().export_chunks([source.chunk for source in sources])
    assert _get(bibliography_api).content == _get(bibliography_api).content


@pytest.mark.parametrize(
    "change",
    [
        {"title": "Conflicting later title"},
        {"metadata": {"year": "2025", "author": "Never merge me"}},
        {"metadata": {"year": "2024", "author": "Never backfill me"}},
        {"metadata": {}},
    ],
)
def test_conflicting_metadata_warns_and_never_merges_other_chunks(
    bibliography_api: ComparisonAPI,
    change: dict[str, object],
) -> None:
    first = _source("first", document_id="paper", title="First title", metadata={"year": "2024"})
    later = _source("second", document_id="paper", title="First title", metadata={"year": "2024"})
    later = later.model_copy(update={"chunk": later.chunk.model_copy(update=change)})
    _save(bibliography_api, "saved", [first, later], answer=_answer([later, first]))
    report = _report(bibliography_api)
    assert report.sources[0].title == first.chunk.title
    assert report.sources[0].metadata == first.chunk.metadata
    assert report.bibtex == BibTeXExporter().export_chunk(first.chunk)
    conflicts = [warning for warning in report.warnings if warning.startswith("Conflicting")]
    assert len(conflicts) == 1
    assert "evidence rank 1" in conflicts[0]
    assert "without merging fields" in conflicts[0]
    assert "Never" not in report.to_json()


def test_nonbibliographic_metadata_is_not_exported_or_compared(
    bibliography_api: ComparisonAPI,
) -> None:
    first = _source(
        "one", document_id="paper", title="Same", metadata={"year": "2024", "secret": "one"}
    )
    later = _source(
        "two", document_id="paper", title="Same", metadata={"year": "2024", "secret": "two"}
    )
    _save(bibliography_api, "saved", [first, later], diagnostics=True)
    report = _report(bibliography_api)
    assert report.sources[0].metadata == {"year": "2024"}
    assert "secret" not in report.to_json() and "private-provider" not in report.to_json()
    assert not any("Conflicting" in warning for warning in report.warnings)


def test_untrusted_unicode_metadata_uses_public_formatter_without_enrichment(
    bibliography_api: ComparisonAPI,
) -> None:
    metadata = dict.fromkeys(BIBLIOGRAPHIC_FIELDS, "captured fallback")
    metadata.update(
        {
            "authors": "Ada Example; Ren\u00e9 Fixture",
            "year": "published 2024-01",
            "doi": "10.0000/synthetic",
            "journal": "Synthetic {venue}",
            "entry_type": "article",
            "url": 'https://invalid.example/<img src=x>&unsafe="yes"',
            "abstract": "must not export this abstract",
            "api_key": "synthetic-sensitive-value",
        }
    )
    source = _source(
        "chunk-\u7814\u7a76",
        document_id='doc"\r\nheader:fixture',
        title="Synthetic caf\u00e9 \u7814\u7a76 <script>not executable</script> {x} \\ % &_",
        metadata=metadata,
        source="https://invalid.example/not-a-metadata-url",
    )
    answer = _answer([source])
    answer.citations[0].title = "Citation label is not the captured title"
    answer.citations[0].snippet = "This label must not supply metadata"
    _save(bibliography_api, 'run"\r\nX-Header:fixture', [source], answer=answer)
    report = _report(bibliography_api, 'run"\r\nX-Header:fixture')
    assert report.sources[0].title == source.chunk.title
    assert report.sources[0].document_id == source.chunk.document_id
    assert set(report.sources[0].metadata) == set(BIBLIOGRAPHIC_FIELDS)
    assert "synthetic-sensitive-value" not in report.to_json()
    assert "not-a-metadata-url" not in report.to_json()
    assert "Citation label" not in report.to_json()
    assert report.bibtex == BibTeXExporter().export_chunk(source.chunk)
    assert "Ada Example and Ren\u00e9 Fixture" in report.bibtex
    assert "year = {2024}" in report.bibtex
    assert "\\{x\\}" in report.bibtex
    response = _get(bibliography_api, 'run"\r\nX-Header:fixture', "bibtex")
    assert "x-header" not in response.headers
    assert response.headers["content-disposition"] == 'attachment; filename="bibliography.bib"'
    assert "\u7814\u7a76".encode() in response.content


def test_blank_title_placeholder_is_explicit(bibliography_api: ComparisonAPI) -> None:
    _save(bibliography_api, "saved", [_source("one", title=" \n", metadata={})])
    report = _report(bibliography_api)
    assert report.sources[0].title == " \n"
    assert "title = {Untitled}" in report.bibtex
    assert any("display placeholder" in warning for warning in report.warnings)


@pytest.mark.parametrize("with_evidence", [False, True])
def test_no_final_citations_is_empty_even_with_claim_or_context_references(
    bibliography_api: ComparisonAPI,
    with_evidence: bool,
) -> None:
    sources = [_source("uncited")] if with_evidence else []
    answer = _answer(sources)
    answer.citations = []
    _save(bibliography_api, "saved", sources, answer=answer)
    report = _report(bibliography_api)
    assert report.sources == [] and report.bibtex == ""
    assert any("No final-answer citations" in warning for warning in report.warnings)
    bibtex = _get(bibliography_api, format="bibtex")
    assert bibtex.status_code == 200 and bibtex.content == b""
    assert bibtex.headers["content-length"] == "0"


@pytest.mark.parametrize("damage", ["missing_chunk", "different_owner"])
def test_bad_final_citation_rejects_whole_export(
    bibliography_api: ComparisonAPI,
    damage: str,
) -> None:
    source = _source("one")
    answer = _answer([source, source])
    if damage == "missing_chunk":
        answer.citations[1].chunk_id = "not-in-snapshot"
    else:
        answer.citations[1].document_id = "not-the-owner"
    _save(bibliography_api, "saved", [source], answer=answer)
    for format in ("json", "bibtex"):
        _error(bibliography_api, 409, "invalid_bibliography_citation", format=format)


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("DELETE FROM agent_events", "run_not_found"),
        ("DELETE FROM agent_events WHERE id = 8", "run_incomplete"),
        (
            "UPDATE agent_events SET payload = json_set(payload, '$.to_state', 'ERROR') "
            "WHERE id = 8",
            "run_failed",
        ),
        ("DELETE FROM agent_events WHERE event_type = 'evidence_snapshot'", "snapshot_unavailable"),
        ("UPDATE agent_events SET payload = 'invalid' WHERE id = 5", "invalid_run_record"),
        ("UPDATE agent_events SET payload = X'80' WHERE id = 5", "invalid_run_record"),
        ("UPDATE agent_events SET timestamp = CAST(X'80' AS TEXT)", "invalid_run_record"),
        (
            "UPDATE agent_events SET payload = json_set(payload, '$.schema_version', '2.0') "
            "WHERE id = 5",
            "invalid_run_record",
        ),
        (
            "UPDATE agent_events SET payload = "
            "json_set(payload, '$.sources[0].text_sha256', 'bad') "
            "WHERE id = 5",
            "invalid_run_record",
        ),
        (
            "UPDATE agent_events SET payload = json_set(payload, '$.sources[0].chunk.metadata', "
            "json('{\"author\":false}')) WHERE id = 5",
            "invalid_run_record",
        ),
        (
            "UPDATE agent_events SET payload = json_set(payload, '$.payload.citations[0]', "
            "json('{}')) WHERE id = 7",
            "invalid_run_record",
        ),
        (
            "UPDATE agent_events SET payload = json_set(payload, '$.sources[0].chunk.metadata', "
            "'private-nonobject') WHERE id = 5",
            "invalid_run_record",
        ),
    ],
)
def test_malformed_failed_legacy_and_missing_records_keep_evidence_errors(
    bibliography_api: ComparisonAPI,
    sql: str,
    code: str,
) -> None:
    _save(bibliography_api, "saved", [_source("one")])
    _mutate(bibliography_api, sql)
    response = _error(bibliography_api, 404 if code == "run_not_found" else 409, code)
    assert "private-nonobject" not in response.text


def test_surrogate_bibliographic_metadata_is_rejected(bibliography_api: ComparisonAPI) -> None:
    _save(bibliography_api, "saved", [_source("one", metadata={"author": "\ud800"})])
    _error(bibliography_api, 409, "invalid_run_record")


@pytest.mark.parametrize("value", ["", None, 123, True, "x" * 257, "\ud800"])
def test_python_request_validation_precedes_storage(
    tmp_path: Path,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SQLiteSavedBibliography(tmp_path / "missing.sqlite3")
    monkeypatch.setattr(sqlite3, "connect", _no_work)
    with pytest.raises(BibliographyError) as exc:
        service.export(value)  # type: ignore[arg-type]
    assert exc.value.status_code == 422
    assert exc.value.code == "invalid_bibliography_request"


@pytest.mark.parametrize("format", ["", "JSON", "markdown", "bib", "x" * 500])
def test_api_format_validation_precedes_storage(
    bibliography_api: ComparisonAPI,
    format: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bibliography_api.container.saved_bibliography, "export", _no_work)
    _error(bibliography_api, 422, "invalid_bibliography_request", format=format)


@pytest.mark.parametrize("query", ["format=json&format=bibtex", "unexpected=private-input"])
def test_repeated_or_unknown_parameters_are_explicit_errors(
    bibliography_api: ComparisonAPI,
    query: str,
) -> None:
    response = bibliography_api.client.get(f"/runs/saved/bibliography?{query}")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_bibliography_request"
    assert "private-input" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_exact_run_and_frozen_identity_character_bounds(bibliography_api: ComparisonAPI) -> None:
    identity = "\u7814" * 256
    _save(bibliography_api, identity, [_source(identity, document_id=identity)])
    report = _report(bibliography_api, identity)
    assert report.sources[0].document_id == report.sources[0].cited_chunks[0].chunk_id == identity
    assert report.run_id == identity
    _error(bibliography_api, 422, "invalid_bibliography_request", identity + "x")
    _save(bibliography_api, "saved", [_source(identity + "x")])
    _error(bibliography_api, 409, "invalid_run_record")
    _save(bibliography_api, "long-doc", [_source("one", document_id=identity + "x")])
    _error(bibliography_api, 409, "invalid_run_record", "long-doc")


def test_fifty_sources_and_citations_inclusive_then_fail_without_truncation(
    bibliography_api: ComparisonAPI,
) -> None:
    sources = [_source(f"c-{index}") for index in range(50)]
    _save(bibliography_api, "saved", sources)
    report = _report(bibliography_api)
    assert len(report.sources) == report.bibtex.count("@") == 50
    assert [s.first_evidence_rank for s in report.sources] == list(range(1, 51))
    _save(bibliography_api, "too-many-citations", sources, answer=_answer([*sources, sources[0]]))
    _error(bibliography_api, 409, "bibliography_limit_exceeded", "too-many-citations")
    _mutate(
        bibliography_api,
        "UPDATE agent_events SET payload = json_insert(payload, '$.sources[#]', "
        "json_extract(payload, '$.sources[0]')) WHERE run_id = 'saved' AND id = 5",
    )
    _error(bibliography_api, 409, "invalid_run_record")


@pytest.mark.parametrize(("unit", "growth"), [("a", 2), ("\u00e9", 4), ("\\", 6)])
def test_exact_json_wire_byte_limit_includes_metadata_bibtex_warnings_and_newline(
    bibliography_api: ComparisonAPI, unit: str, growth: int
) -> None:
    api = bibliography_api
    source = _source("one", metadata={"author": unit, "doi": "10.0000/synthetic"})
    for padding in range(growth):
        run_id = "saved" + "x" * padding
        _save(api, run_id, [source])
        remaining = MAX_RESPONSE_BYTES - len(_get(api, run_id).content)
        if remaining % growth == 0:
            break
    assert remaining % growth == 0
    source.chunk.metadata["author"] = unit * (1 + remaining // growth)
    _mutate(
        api,
        "UPDATE agent_events SET payload = "
        "json_set(payload, '$.sources[0].chunk.metadata.author', ?) "
        "WHERE run_id = ? AND event_type = 'evidence_snapshot'",
        (source.chunk.metadata["author"], run_id),
    )
    exact = _get(api, run_id)
    assert exact.status_code == 200
    assert len(exact.content) == int(exact.headers["content-length"]) == MAX_RESPONSE_BYTES
    report = SavedBibliography.model_validate_json(exact.content)
    assert exact.content == report.to_json().encode("utf-8")
    assert len(report.to_bibtex().encode("utf-8")) < MAX_RESPONSE_BYTES
    assert _get(api, run_id, "bibtex").status_code == 200
    _save(api, run_id + "x", [source])
    for format in ("json", "bibtex"):
        _error(api, 413, "bibliography_too_large", run_id + "x", format)
    raw = (
        json.dumps(
            report.model_copy(update={"run_id": run_id + "x"}).model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    assert len(raw.encode("utf-8")) == MAX_RESPONSE_BYTES + 1


def test_bibtex_byte_limit_is_exact_utf8_not_character_count(
    bibliography_api: ComparisonAPI,
) -> None:
    _save(bibliography_api, "saved", [])
    report = _report(bibliography_api).model_copy(
        update={"bibtex": "\u00e9" * (MAX_RESPONSE_BYTES // 2)}
    )
    assert len(report.to_bibtex().encode("utf-8")) == MAX_RESPONSE_BYTES
    with pytest.raises(BibliographyError) as exc:
        report.model_copy(update={"bibtex": report.bibtex + "x"}).to_bibtex()
    assert exc.value.status_code == 413


def test_readonly_restart_and_removed_corpus_leave_exports_and_events_unchanged(
    bibliography_api: ComparisonAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = bibliography_api
    sources = [_source("one"), _source("two")]
    api.container.document_store.add_documents(
        [Document(**source.chunk.model_dump(exclude={"chunk_id"})) for source in sources],
        [source.chunk for source in sources],
    )
    _save(api, "saved", sources)
    before = {format: _get(api, format=format).content for format in ("json", "bibtex")}
    evidence = {
        format: api.client.get("/runs/saved/export", params={"format": format}).content
        for format in ("json", "markdown")
    }
    events = api.container.event_log.list_events()
    _mutate(api, "UPDATE chunks SET title = 'Changed corpus', metadata = '{}'")
    assert _get(api).content == before["json"]
    _mutate(api, "DELETE FROM chunks")
    _mutate(api, "DELETE FROM documents")
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
    before_bytes = api.path.read_bytes()
    with TestClient(restarted) as client:
        for format, content in before.items():
            response = client.get("/runs/saved/bibliography", params={"format": format})
            assert response.status_code == 200 and response.content == content
        for format, content in evidence.items():
            assert client.get("/runs/saved/export", params={"format": format}).content == content
    assert api.path.read_bytes() == before_bytes
    assert api.container.event_log.list_events() == events
    _mutate(api, "DROP TABLE documents")
    _mutate(api, "DROP TABLE chunks")
    _mutate(api, "DROP TABLE graph_chunks")
    assert SQLiteSavedBibliography(api.path).export("saved").to_json().encode() == before["json"]


def test_only_one_read_transaction_reads_bounded_event_rows(
    bibliography_api: ComparisonAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(bibliography_api, "saved")
    connect = sqlite3.connect
    queries: list[str] = []
    connections: list[sqlite3.Connection] = []

    def traced(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        assert uri is True and database.endswith("?mode=ro") and timeout == 5
        connection = connect(database, uri=uri, timeout=timeout, check_same_thread=False)
        connection.set_trace_callback(queries.append)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced)
    _report(bibliography_api)
    assert len(connections) == 1 and queries.count("BEGIN") == 1
    assert len([query for query in queries if query.lstrip().startswith("SELECT")]) == 2
    assert all(
        "agent_events" in query and "LIMIT" in query
        for query in queries
        if query.lstrip().startswith("SELECT")
    )
    assert not any("INSERT" in query or "UPDATE" in query or "CREATE" in query for query in queries)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_exact_event_count_and_payload_limits(bibliography_api: ComparisonAPI) -> None:
    api = bibliography_api
    _save(api, "saved")
    for _ in range(MAX_RUN_EVENTS - 8):
        api.container.event_log.append_event("synthetic-agent", "saved", "diagnostic", {})
    _report(api)
    _mutate(
        api,
        "UPDATE agent_events SET payload = ? WHERE id = ?",
        ("{}" + " " * (MAX_EVENT_BYTES - 2), MAX_RUN_EVENTS),
    )
    _report(api)
    _mutate(api, "UPDATE agent_events SET payload = payload || ' ' WHERE id = ?", (MAX_RUN_EVENTS,))
    _error(api, 409, "evidence_read_limit_exceeded")
    _mutate(api, "UPDATE agent_events SET payload = '{}' WHERE id = ?", (MAX_RUN_EVENTS,))
    api.container.event_log.append_event("synthetic-agent", "saved", "diagnostic", {})
    _error(api, 409, "evidence_read_limit_exceeded")


def test_exact_aggregate_read_limit_and_preflight(bibliography_api: ComparisonAPI) -> None:
    api = bibliography_api
    _save(api, "saved")
    ids = [
        api.container.event_log.append_event("synthetic-agent", "saved", "diagnostic", {})
        for _ in range(8)
    ]
    with closing(sqlite3.connect(api.path)) as connection, connection:
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
    _report(api)
    _mutate(api, "UPDATE agent_events SET payload = payload || ' ' WHERE id = ?", (ids[-1],))
    _error(api, 409, "evidence_read_limit_exceeded")


def test_oversized_events_fail_before_payload_hydration(
    bibliography_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = bibliography_api
    _save(api, "saved")
    _mutate(api, "UPDATE agent_events SET payload = ? WHERE id = 5", ("x" * (MAX_EVENT_BYTES + 1),))
    queries: list[str] = []
    connect = sqlite3.connect

    def traced(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        connection = connect(database, uri=uri, timeout=timeout)
        connection.set_trace_callback(queries.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced)
    _error(api, 409, "evidence_read_limit_exceeded")
    assert not any("SELECT id, CAST(timestamp AS BLOB)" in query for query in queries)


def test_preflight_and_payloads_share_a_consistent_read_snapshot(
    bibliography_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = bibliography_api
    _save(api, "saved", [_source("one", metadata={"author": "Original captured author"})])
    _mutate(api, "PRAGMA journal_mode = WAL")
    connect = sqlite3.connect
    changed = False

    def concurrent_write(query: str) -> None:
        nonlocal changed
        if not changed and "SELECT id, CAST(timestamp AS BLOB)" in query:
            changed = True
            with closing(connect(api.path)) as writer, writer:
                writer.execute(
                    "UPDATE agent_events SET payload = json_set(payload, "
                    "'$.sources[0].chunk.metadata.author', 'Concurrently modified') WHERE id = 5"
                )

    def traced(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        connection = connect(database, uri=uri, timeout=timeout)
        connection.set_trace_callback(concurrent_write)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced)
    assert _report(api).sources[0].metadata["author"] == "Original captured author"
    assert changed
    assert _report(api).sources[0].metadata["author"] == "Concurrently modified"


@pytest.mark.parametrize("encoding", ["UTF-16le", "UTF-16be"])
def test_utf16_event_storage_retains_utf8_output(tmp_path: Path, encoding: str) -> None:
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
        _save(api, "saved", [_source("one", metadata={"author": "Ren\u00e9 \u7814\u7a76"})])
        report = _report(api)
        assert "Ren\u00e9 \u7814\u7a76" in report.bibtex


def test_operational_failure_is_private_and_missing_storage_is_not_created(
    bibliography_api: ComparisonAPI,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _save(bibliography_api, "saved")
    _mutate(bibliography_api, "DROP TABLE agent_events")
    with caplog.at_level(logging.WARNING, logger="api.bibliography"):
        response = _error(bibliography_api, 503, "bibliography_storage_unavailable")
    assert "no such table" not in response.text + caplog.text
    assert str(bibliography_api.path) not in response.text + caplog.text
    assert "Saved bibliography failed: bibliography_storage_unavailable" in caplog.text
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(EvidenceExportError) as exc:
        SQLiteSavedBibliography(path).export("saved")
    assert exc.value.status_code == 503
    assert not path.exists()


def test_locked_storage_is_not_an_empty_bibliography(
    bibliography_api: ComparisonAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(bibliography_api, "saved")
    connect = sqlite3.connect

    def immediate(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        return connect(database, uri=uri, timeout=0)

    with closing(connect(bibliography_api.path)) as writer:
        writer.execute("BEGIN EXCLUSIVE")
        monkeypatch.setattr(sqlite3, "connect", immediate)
        _error(bibliography_api, 503, "bibliography_storage_unavailable")
        writer.rollback()


def test_openapi_documents_both_downloads_and_exact_selection_bounds(
    bibliography_api: ComparisonAPI,
) -> None:
    schema = bibliography_api.client.get("/openapi.json").json()
    route = schema["paths"]["/runs/{run_id}/bibliography"]
    assert set(route) == {"get"}
    operation = route["get"]
    assert {"200", "404", "409", "413", "422", "503"} <= set(operation["responses"])
    assert set(operation["responses"]["200"]["content"]) == {
        "application/json",
        "application/x-bibtex",
    }
    parameters = {parameter["name"]: parameter["schema"] for parameter in operation["parameters"]}
    assert parameters["run_id"]["minLength"] == 1 and parameters["run_id"]["maxLength"] == 256
    assert parameters["format"]["default"] == "bibtex"
    assert parameters["format"]["enum"] == ["bibtex", "json"]
    models = schema["components"]["schemas"]
    assert models["SavedBibliography"]["properties"]["sources"]["maxItems"] == 50
    assert models["BibliographySource"]["properties"]["cited_chunks"]["maxItems"] == 50
