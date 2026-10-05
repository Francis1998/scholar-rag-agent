"""Complete human-screening downloads, independent of models and retrieval."""

import csv
import io
import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from storage import paper_screening
from storage.document_store import SQLiteDocumentStore
from storage.paper_collections import MAX_REVISION, CollectionError, PaperCollection
from storage.paper_screening import ScreeningReview, ScreeningSubmission, SQLitePaperScreening


@dataclass
class ExportAPI:
    client: TestClient
    container: AppContainer
    database: Path
    collection: PaperCollection

    @property
    def path(self) -> str:
        return f"/collections/{self.collection.collection_id}/screening/export"


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("An export must not retrieve, generate, journal, or use the network.")


@pytest.fixture
def export_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ExportAPI]:
    database = tmp_path / "screening.sqlite3"
    app = create_app(offline_settings(database))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    with TestClient(app) as client:
        container = app.state.container
        ids = ["z-methods", "a-background", "n-unsure", "b-stale", "m-unread"]
        documents = [
            Document(
                document_id=identifier,
                title=f"Synthetic {identifier}",
                source="synthetic:screening-export",
                text="GraphRAG private-body-sentinel",
                metadata={"private": "private-metadata-sentinel"},
            )
            for identifier in ids
        ]
        chunks = [
            Chunk(chunk_id=f"chunk-{doc.document_id}", **doc.model_dump()) for doc in documents
        ]
        container.document_store.add_documents(documents, chunks)
        container.hybrid_retriever.add_chunks(chunks)
        collection = container.paper_collections.create(name="Human review", document_ids=ids)
        yield ExportAPI(client, container, database, collection)


def save(
    api: ExportAPI,
    index: int,
    decision: str = "include",
    *,
    reason: str = "Synthetic human opinion, not scientific validation.",
) -> dict[str, Any]:
    response = api.client.put(
        api.path.removesuffix("/export") + "/" + quote(api.collection.document_ids[index], safe=""),
        json={
            "collection_revision": api.collection.revision,
            "expected_decision_revision": 0,
            "decision": decision,
            "reason": reason,
        },
    )
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def download(api: ExportAPI, format: str = "json") -> httpx.Response:
    response = api.client.get(
        api.path, params={"collection_revision": api.collection.revision, "format": format}
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert int(response.headers["content-length"]) == len(response.content)
    assert response.headers["content-type"] == (
        "application/json" if format == "json" else "text/csv; charset=utf-8"
    )
    assert re.fullmatch(
        rf'attachment; filename="screening-[A-Za-z0-9_-]+-r{api.collection.revision}\.{format}"',
        response.headers["content-disposition"],
    )
    assert "private-body-sentinel" not in response.text
    assert "private-metadata-sentinel" not in response.text
    return response


def csv_rows(response: httpx.Response) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(response.text, newline="")))


def test_json_exports_every_status_in_collection_order_with_exact_review_provenance(
    export_api: ExportAPI,
) -> None:
    api = export_api
    old = save(api, 3)
    api.collection = api.container.paper_collections.replace(
        api.collection.collection_id,
        name="Revised human review",
        document_ids=api.collection.document_ids,
        expected_revision=1,
    )
    saved = [save(api, i, label) for i, label in enumerate(("include", "exclude", "unsure"))]
    result = download(api).json()
    assert result["schema_version"] == "1.0"
    assert result["collection_id"] == api.collection.collection_id
    assert result["collection_name"] == api.collection.name
    assert result["collection_revision"] == 2
    assert result["total_documents"] == 5
    assert result["counts"] == dict.fromkeys(
        ("include", "exclude", "unsure", "stale", "unscreened"), 1
    )
    assert result["included_document_ids"] == [api.collection.document_ids[0]]
    assert [item["document_id"] for item in result["items"]] == list(api.collection.document_ids)
    assert [item["status"] for item in result["items"]] == [
        "include",
        "exclude",
        "unsure",
        "stale",
        "unscreened",
    ]
    assert [item["review"] for item in result["items"]] == [*saved, old, None]
    assert "next_cursor" not in result
    assert all(item["title"] == f"Synthetic {item['document_id']}" for item in result["items"])
    assert all(item["source"] == "synthetic:screening-export" for item in result["items"])
    assert all(
        not item["title_truncated"] and not item["source_truncated"] for item in result["items"]
    )
    queue = api.client.get(
        api.path.removesuffix("/export"),
        params={"collection_revision": 2, "limit": 1, "status": "unscreened"},
    ).json()
    assert queue["counts"] == result["counts"]
    assert queue["included_document_ids"] == result["included_document_ids"]
    rows = csv_rows(download(api, "csv"))
    assert [row["status"][1:] for row in rows] == [item["status"] for item in result["items"]]
    assert [row["included_document_id"][1:] for row in rows if row["included_document_id"]] == (
        result["included_document_ids"]
    )
    assert rows[3]["review_decision"] == "'include"
    assert rows[3]["review_collection_revision"] == "1"
    assert rows[3]["review_revision"] == "1"
    assert rows[3]["included_document_id"] == ""
    assert rows[4]["review_decision"] == rows[4]["review_reason"] == ""
    for row in rows:
        assert row["collection_revision"] == "2"
        assert row["total_documents"] == "5"
        for status, count in result["counts"].items():
            assert row[f"count_{status}"] == str(count)


def test_direct_service_matches_api_and_restart_without_any_export_side_effects(
    export_api: ExportAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = export_api
    save(api, 0)
    expected: dict[str, bytes] = {}
    before = api.database.read_bytes()
    with monkeypatch.context() as patch:
        for target, method in (
            (api.container.llm, "generate"),
            (api.container.runner, "run"),
            (api.container.runner, "preview"),
            (api.container.hybrid_retriever, "retrieve"),
            (api.container.document_store, "list_chunks"),
            (api.container.document_catalog, "list_documents"),
            (api.container.event_log, "append_event"),
            (api.container.event_log, "append_transition"),
        ):
            patch.setattr(target, method, denied)
        for format in ("json", "csv"):
            direct = api.container.paper_screening.export_results(
                api.collection.collection_id, collection_revision=1, format=format
            )
            response = download(api, format)
            assert direct.content == response.content
            assert direct.media_type in response.headers["content-type"]
            assert direct.filename in response.headers["content-disposition"]
            expected[format] = direct.content
    assert api.database.read_bytes() == before
    restarted = create_app(offline_settings(api.database))
    with TestClient(restarted) as client:
        for format, content in expected.items():
            response = client.get(api.path, params={"collection_revision": 1, "format": format})
            assert response.content == content
        selected = json.loads(expected["json"])["included_document_ids"]
        preview = client.post("/retrieve", json={"query": "GraphRAG", "document_ids": selected})
        assert preview.status_code == 200, preview.text
        assert {source["chunk"]["document_id"] for source in preview.json()["sources"]} == set(
            selected
        )
        assert restarted.state.container.event_log.list_events() == []


def test_empty_inclusions_are_explicit_and_not_an_unscoped_retrieval(export_api: ExportAPI) -> None:
    result = download(export_api).json()
    assert result["included_document_ids"] == []
    assert result["counts"]["unscreened"] == result["total_documents"] == 5
    assert all(item["review"] is None for item in result["items"])
    response = export_api.client.post(
        "/retrieve", json={"query": "GraphRAG", "document_ids": result["included_document_ids"]}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("format", ["json", "csv"])
def test_exact_full_download_byte_limit_includes_encoding_and_csv_escaping(
    export_api: ExportAPI, monkeypatch: pytest.MonkeyPatch, format: str
) -> None:
    save(export_api, 0, reason=' \t= "quoted"\r\n' + "\U0001f4da" * 900)
    original = download(export_api, format)
    assert len(original.content) > len(original.text)
    monkeypatch.setattr(paper_screening, "MAX_EXPORT_BYTES", len(original.content))
    assert download(export_api, format).content == original.content
    monkeypatch.setattr(paper_screening, "MAX_EXPORT_BYTES", len(original.content) - 1)
    response = export_api.client.get(
        export_api.path, params={"collection_revision": 1, "format": format}
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "screening_export_too_large"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-disposition" not in response.headers
    assert "items" not in response.json()
    with pytest.raises(CollectionError) as error:
        export_api.container.paper_screening.export_results(
            export_api.collection.collection_id, collection_revision=1, format=format
        )
    assert error.value.code == "screening_export_too_large"


@pytest.mark.parametrize(
    "text",
    [
        "=1+1",
        "+1+1",
        "-1+1",
        "@SUM(1,1)",
        "\t=1+1",
        "\r=1+1",
        " \t\r\n=1+1",
        "\u00a0\u2003+1",
        "\ufeff=1",
        "'=literal",
        'Comma, quote " and \u7814\u7a76\r\nnext line',
        "plain text",
    ],
)
def test_csv_text_is_uniformly_formula_safe_and_reversibly_escaped(
    export_api: ExportAPI, text: str
) -> None:
    api = export_api
    save(api, 0, reason=text)
    with sqlite3.connect(api.database) as connection:
        connection.execute("UPDATE documents SET title = ?, source = ?", (text, text))
    row = csv_rows(download(api, "csv"))[0]
    assert row["csv_text_encoding"] == "'apostrophe-prefix-v1"
    for field in ("title", "source", "review_reason"):
        assert row[field] == "'" + text
        assert row[field][1:] == text
    assert row["document_id"] == "'" + api.collection.document_ids[0]
    assert row["included_document_id"] == "'" + api.collection.document_ids[0]
    assert row["review_updated_at"].startswith("'")
    assert row["collection_name"] == "'Human review"
    assert download(api).json()["items"][0]["review"]["reason"] == text


def test_csv_preserves_formula_like_and_nul_document_ids_and_safe_download_names(
    export_api: ExportAPI,
) -> None:
    api = export_api
    ids = ("=1+1", "+1", "-1", "@SUM(1,1)", "nul\x00id", "export")
    api.container.document_store.add_documents(
        [Document(document_id=i, title="", source="", text="Synthetic") for i in ids], []
    )
    api.collection = api.container.paper_collections.create(
        name='=Review / "quoted"; \u7814\u7a76', document_ids=ids
    )
    for i in range(len(ids)):
        save(api, i)
    rows = csv_rows(download(api, "csv"))
    assert [r["included_document_id"][1:] for r in rows] == list(ids)
    assert all(r["title"] == r["source"] == "'" for r in rows)
    assert all(r["collection_name"] == "'" + api.collection.name for r in rows)
    assert download(api).json()["included_document_ids"] == list(ids)


@pytest.mark.parametrize(
    "query",
    [
        "",
        "collection_revision=0",
        "collection_revision=-1",
        "collection_revision=1.0",
        "collection_revision=true",
        "collection_revision=%2B1",
        "collection_revision=%201",
        "collection_revision=%EF%BC%91",
        f"collection_revision={MAX_REVISION + 1}",
        "collection_revision=" + "1" * 5000,
        "collection_revision=1&format=xml",
        "collection_revision=1&format=JSON",
        "collection_revision=1&format=",
        "collection_revision=1&format=json&format=csv",
        "collection_revision=1&collection_revision=2",
        "collection_revision=1&limit=1",
        "collection_revision=1&cursor=z-methods",
        "collection_revision=1&status=include",
    ],
)
def test_export_rejects_invalid_duplicate_or_partial_selection_parameters(
    export_api: ExportAPI, query: str, caplog: pytest.LogCaptureFixture
) -> None:
    response = export_api.client.get(f"{export_api.path}?{query}")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_screening_request"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "invalid_screening_request" in caplog.text


@pytest.mark.parametrize("revision", [True, False, 1.0, "1", 0, -1, MAX_REVISION + 1])
def test_export_python_revision_boundary_is_strict(export_api: ExportAPI, revision: object) -> None:
    with pytest.raises(ValidationError):
        export_api.container.paper_screening.export_results(
            export_api.collection.collection_id,
            collection_revision=revision,  # type: ignore[arg-type]
        )


def test_export_python_format_boundary_is_strict(export_api: ExportAPI) -> None:
    with pytest.raises(ValidationError):
        export_api.container.paper_screening.export_results(
            export_api.collection.collection_id, collection_revision=1, format="JSON"
        )


@pytest.mark.parametrize("format", ["json", "csv"])
def test_errors_for_stale_revision_missing_documents_and_collection(
    export_api: ExportAPI, format: str
) -> None:
    api = export_api
    save(api, 0)
    api.collection = api.container.paper_collections.replace(
        api.collection.collection_id,
        name="Revision 2",
        document_ids=api.collection.document_ids,
        expected_revision=1,
    )
    stale = api.client.get(api.path, params={"collection_revision": 1, "format": format})
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "collection_revision_conflict"
    assert download(api).json()["included_document_ids"] == []
    with sqlite3.connect(api.database) as connection:
        connection.execute(
            "DELETE FROM documents WHERE document_id = ?", (api.collection.document_ids[0],)
        )
    missing = api.client.get(api.path, params={"collection_revision": 2, "format": format})
    assert missing.status_code == 409
    assert missing.json()["detail"]["code"] == "collection_documents_missing"
    api.container.paper_collections.delete(api.collection.collection_id, expected_revision=2)
    deleted = api.client.get(api.path, params={"collection_revision": 2, "format": format})
    assert deleted.status_code == 404
    assert deleted.json()["detail"]["code"] == "collection_not_found"


@pytest.mark.parametrize(
    "record",
    ["private-invalid-record", "\x00", b"private-blob", "x" * 16385],
)
def test_invalid_reviews_fail_with_sanitized_errors(
    export_api: ExportAPI, record: object, caplog: pytest.LogCaptureFixture
) -> None:
    save(export_api, 4)
    with sqlite3.connect(export_api.database) as connection:
        connection.execute("UPDATE paper_screening_decisions SET record = ?", (record,))
    response = export_api.client.get(export_api.path, params={"collection_revision": 1})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_screening_record"
    assert "private-" not in response.text + caplog.text
    assert "invalid_screening_record" in caplog.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_id", "wrong"),
        ("collection_id", "col_" + "0" * 32),
        ("collection_revision", 2),
        ("revision", True),
        ("updated_at", "not-a-date"),
        ("reason", "\ud800"),
        ("schema_version", "2.0"),
    ],
)
def test_review_consistency_validation_cannot_be_bypassed_by_export(
    export_api: ExportAPI, field: str, value: object
) -> None:
    record = save(export_api, 4)
    record[field] = value
    with sqlite3.connect(export_api.database) as connection:
        connection.execute("UPDATE paper_screening_decisions SET record = ?", (json.dumps(record),))
    response = export_api.client.get(export_api.path, params={"collection_revision": 1})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_screening_record"


@pytest.mark.parametrize("field", ["title", "source"])
@pytest.mark.parametrize("value", [b"private-blob", "private\x00label", "x" * 2500 + "\x00"])
def test_invalid_or_nul_labels_fail_instead_of_truncating_away_corruption(
    export_api: ExportAPI, field: str, value: object, caplog: pytest.LogCaptureFixture
) -> None:
    with sqlite3.connect(export_api.database) as connection:
        if field == "title":
            connection.execute("UPDATE documents SET title = ?", (value,))
        else:
            connection.execute("UPDATE documents SET source = ?", (value,))
    response = export_api.client.get(export_api.path, params={"collection_revision": 1})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_document_record"
    assert "private-" not in response.text + caplog.text


def test_invalid_unicode_label_and_unavailable_storage_are_explicit(
    export_api: ExportAPI, caplog: pytest.LogCaptureFixture
) -> None:
    with sqlite3.connect(export_api.database) as connection:
        connection.execute("UPDATE documents SET title = CAST(X'FF' AS TEXT)")
    response = export_api.client.get(export_api.path, params={"collection_revision": 1})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_document_record"
    with sqlite3.connect(export_api.database) as connection:
        connection.execute("DROP TABLE paper_screening_decisions")
    unavailable = export_api.client.get(export_api.path, params={"collection_revision": 1})
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"]["code"] == "collection_storage_error"
    assert "paper_screening_decisions" not in unavailable.text + caplog.text


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
@pytest.mark.parametrize("extra", [0, 1])
def test_label_bounds_use_catalog_unicode_decoding(
    tmp_path: Path, encoding: str, extra: int
) -> None:
    path = tmp_path / "unicode.sqlite3"
    with sqlite3.connect(path) as connection:
        if encoding == "UTF-16le":
            connection.execute("PRAGMA encoding = 'UTF-16le'")
        elif encoding == "UTF-16be":
            connection.execute("PRAGMA encoding = 'UTF-16be'")
        connection.execute("CREATE TABLE marker (value TEXT)")
    title = "\U0001f4da" * (300 + extra)
    source = "\u7814" * (512 + extra)
    SQLiteDocumentStore(path).add_documents(
        [Document(document_id="synthetic", title=title, source=source, text="private")], []
    )
    store = SQLitePaperScreening(path)
    collection = store.collections.create(name="Unicode", document_ids=["synthetic"])
    before = path.read_bytes()
    result = json.loads(
        store.export_results(collection.collection_id, collection_revision=1).content
    )
    item = result["items"][0]
    assert item["title"] == title[:300]
    assert item["source"] == source[:512]
    assert item["title_truncated"] is bool(extra)
    assert item["source_truncated"] is bool(extra)
    assert path.read_bytes() == before


def test_wal_writer_cannot_mix_membership_reviews_and_labels_across_revisions(
    export_api: ExportAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = export_api
    old_review = save(api, 0)
    with sqlite3.connect(api.database) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    original = api.container.paper_screening._reviews

    def concurrent_write(
        connection: sqlite3.Connection, collection: PaperCollection
    ) -> dict[str, ScreeningReview]:
        assert connection.in_transaction
        reviews = original(connection, collection)
        changed = {**old_review, "revision": 2, "collection_revision": 2, "reason": "New opinion"}
        with sqlite3.connect(api.database, timeout=1) as writer:
            writer.execute("UPDATE paper_collections SET name = 'New name', revision = 2")
            writer.execute("UPDATE documents SET title = 'New title', source = 'New source'")
            writer.execute(
                "UPDATE paper_screening_decisions SET record = ?", (json.dumps(changed),)
            )
        return reviews

    with monkeypatch.context() as patch:
        patch.setattr(api.container.paper_screening, "_reviews", concurrent_write)
        snapshot = download(api).json()
    assert snapshot["collection_revision"] == 1
    assert snapshot["collection_name"] == "Human review"
    assert snapshot["items"][0]["review"] == old_review
    assert snapshot["items"][0]["title"] == f"Synthetic {api.collection.document_ids[0]}"
    assert snapshot["items"][0]["source"] == "synthetic:screening-export"
    current = api.client.get(api.path, params={"collection_revision": 2}).json()
    assert current["collection_name"] == "New name"
    assert current["items"][0]["review"]["reason"] == "New opinion"
    assert current["items"][0]["title"] == "New title"


def test_complete_hundred_member_export_has_no_page_dependency_and_checks_last_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.sqlite3"
    ids = [f"paper-{n:03}" for n in reversed(range(100))]
    SQLiteDocumentStore(path).add_documents(
        [Document(document_id=i, title=i, source="synthetic", text="private") for i in ids], []
    )
    store = SQLitePaperScreening(path)
    collection = store.collections.create(name="100 members", document_ids=ids)
    result = json.loads(
        store.export_results(collection.collection_id, collection_revision=1).content
    )
    assert [item["document_id"] for item in result["items"]] == ids
    assert result["counts"]["unscreened"] == result["total_documents"] == 100
    for identifier in ids:
        store.submit(
            collection.collection_id,
            identifier,
            ScreeningSubmission(
                collection_revision=1,
                expected_decision_revision=0,
                decision="include",
                reason="\x01" * 1000,
            ),
        )
    with pytest.raises(CollectionError) as large:
        store.export_results(collection.collection_id, collection_revision=1)
    assert large.value.code == "screening_export_too_large"
    csv_download = store.export_results(
        collection.collection_id, collection_revision=1, format="csv"
    )
    assert len(list(csv.DictReader(io.StringIO(csv_download.content.decode(), newline="")))) == 100
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE paper_screening_decisions SET record = 'invalid-last-record' "
            "WHERE document_id = ?",
            (ids[-1],),
        )
    with pytest.raises(CollectionError) as invalid:
        store.export_results(collection.collection_id, collection_revision=1, format="csv")
    assert invalid.value.code == "invalid_screening_record"


@pytest.mark.parametrize("format", ["json", "csv"])
def test_real_export_cap_rejects_complete_large_unicode_results(
    tmp_path: Path, format: str
) -> None:
    path = tmp_path / "oversized.sqlite3"
    ids = [f"paper-{n}" for n in range(100)]
    SQLiteDocumentStore(path).add_documents(
        [
            Document(
                document_id=i, title="\U0001f4da" * 300, source="\U0001f4da" * 512, text="private"
            )
            for i in ids
        ],
        [],
    )
    store = SQLitePaperScreening(path)
    collection = store.collections.create(name="Maximum Unicode labels", document_ids=ids)
    for identifier in ids:
        store.submit(
            collection.collection_id,
            identifier,
            ScreeningSubmission(
                collection_revision=1,
                expected_decision_revision=0,
                decision="include",
                reason="\U0001f4da" * 1000,
            ),
        )
    before = path.read_bytes()
    with pytest.raises(CollectionError) as error:
        store.export_results(collection.collection_id, collection_revision=1, format=format)
    assert error.value.code == "screening_export_too_large"
    assert error.value.status_code == 413
    assert path.read_bytes() == before


def test_removed_and_readded_members_never_reactivate_old_inclusions(export_api: ExportAPI) -> None:
    api = export_api
    old = save(api, 0)
    ids = api.collection.document_ids
    api.collection = api.container.paper_collections.replace(
        api.collection.collection_id,
        name=api.collection.name,
        document_ids=ids[1:],
        expected_revision=1,
    )
    removed = download(api).json()
    assert removed["total_documents"] == 4
    assert removed["included_document_ids"] == []
    assert old["document_id"] not in [item["document_id"] for item in removed["items"]]
    api.collection = api.container.paper_collections.replace(
        api.collection.collection_id,
        name=api.collection.name,
        document_ids=ids,
        expected_revision=2,
    )
    restored = download(api).json()
    assert restored["items"][0]["status"] == "stale"
    assert restored["items"][0]["review"] == old
    assert restored["included_document_ids"] == []


def test_only_current_members_are_read_and_corrupt_collection_is_not_repaired(
    export_api: ExportAPI,
) -> None:
    api = export_api
    api.container.document_store.add_documents(
        [Document(document_id="unselected", title="outside", source="synthetic", text="private")],
        [],
    )
    with sqlite3.connect(api.database) as connection:
        connection.execute(
            "UPDATE documents SET title = CAST(X'FF' AS TEXT) WHERE document_id = 'unselected'"
        )
        connection.execute(
            "INSERT INTO paper_screening_decisions VALUES (?, 'unselected', 'invalid-unselected')",
            (api.collection.collection_id,),
        )
        connection.execute("UPDATE documents SET metadata = 'invalid-unused-metadata'")
    assert download(api).json()["total_documents"] == 5
    with sqlite3.connect(api.database) as connection:
        connection.execute(
            "UPDATE paper_collections SET document_ids = 'invalid-private-membership'"
        )
    invalid = api.client.get(api.path, params={"collection_revision": 1})
    assert invalid.status_code == 409
    assert invalid.json()["detail"]["code"] == "invalid_collection_record"
    assert "invalid-private-membership" not in invalid.text


def test_export_openapi_describes_both_downloads_and_errors(export_api: ExportAPI) -> None:
    schema = export_api.client.get("/openapi.json").json()
    operation = schema["paths"]["/collections/{collection_id}/screening/export"]["get"]
    responses = operation["responses"]
    assert set(responses) == {"200", "404", "409", "413", "422", "503"}
    assert set(responses["200"]["content"]) == {"application/json", "text/csv"}
    assert set(schema["components"]["schemas"]["ScreeningExport"]["properties"]) == {
        "schema_version",
        "collection_id",
        "collection_name",
        "collection_revision",
        "total_documents",
        "counts",
        "items",
        "included_document_ids",
    }


def test_latest_human_review_revision_and_recorded_timestamp_are_retained(
    export_api: ExportAPI,
) -> None:
    api = export_api
    save(api, 0, reason="An older opinion that must not be exported.")
    response = api.client.put(
        api.path.removesuffix("/export") + "/" + api.collection.document_ids[0],
        json={
            "collection_revision": 1,
            "expected_decision_revision": 1,
            "decision": "exclude",
            "reason": "The latest human reason.",
        },
    )
    assert response.status_code == 200
    latest = response.json()
    result = download(api).json()
    assert result["items"][0]["status"] == "exclude"
    assert result["items"][0]["review"] == latest
    assert result["included_document_ids"] == []
    row = csv_rows(download(api, "csv"))[0]
    assert row["review_revision"] == "2"
    assert row["review_reason"] == "'The latest human reason."
    assert datetime.fromisoformat(row["review_updated_at"][1:]) == datetime.fromisoformat(
        latest["updated_at"]
    )


def test_maximum_revision_is_valid_and_not_rounded(export_api: ExportAPI) -> None:
    api = export_api
    with sqlite3.connect(api.database) as connection:
        connection.execute("UPDATE paper_collections SET revision = ?", (MAX_REVISION,))
    api.collection = api.container.paper_collections.get(api.collection.collection_id)
    save(api, 0)
    result = download(api).json()
    assert result["collection_revision"] == MAX_REVISION
    assert result["items"][0]["review"]["collection_revision"] == MAX_REVISION
    assert csv_rows(download(api, "csv"))[0]["collection_revision"] == str(MAX_REVISION)


@pytest.mark.parametrize(("name", "slug"), [("\u7814" * 120, "collection"), ("x" * 120, "x" * 40)])
def test_filename_slug_is_bounded_ascii_without_changing_the_saved_name(
    export_api: ExportAPI, name: str, slug: str
) -> None:
    api = export_api
    api.collection = api.container.paper_collections.replace(
        api.collection.collection_id,
        name=name,
        document_ids=api.collection.document_ids,
        expected_revision=1,
    )
    response = download(api)
    assert response.json()["collection_name"] == name
    assert response.headers["content-disposition"] == (
        f'attachment; filename="screening-{slug}-{api.collection.collection_id}-r2.json"'
    )
