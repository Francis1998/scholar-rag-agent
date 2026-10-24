"""A complete human progress artifact, never a screening recommendation or flow estimate."""

import hashlib
import json
import re
import socket
import sqlite3
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from agent.markdown import literal_block
from api.application import create_app
from retrieval.models import Document
from storage import paper_screening, screening_report
from storage.paper_collections import MAX_REVISION, CollectionError, PaperCollection
from storage.paper_screening import ScreeningExport, ScreeningReview, ScreeningSubmission
from storage.screening_report import (
    ScreeningReport,
    build_screening_report,
    export_screening_report,
)
from tests.test_screening_export import ExportAPI, denied, save
from tests.test_screening_export import export_api as export_api

REASON = "Wrong population\n``````\n<script>literal</script> & [link](javascript:x)\r\n研究"
IDS = (
    "z-included",
    "../paper/?draft#v1",
    "a-exclusion",
    "case-distinct",
    "space-distinct",
    "unsure",
    "old-exclude",
    "old-include",
    "unread",
)


@pytest.fixture
def report_api(export_api: ExportAPI) -> ExportAPI:
    api = export_api
    api.container.document_store.add_documents(
        [
            Document(
                document_id=identifier,
                title="Same synthetic title",
                source="synthetic:progress",
                text="PRIVATE_BODY",
                metadata={"private": "PRIVATE_METADATA"},
            )
            for identifier in IDS
        ],
        [],
    )
    api.collection = api.container.paper_collections.create(
        name="Synthetic review <literal>", document_ids=IDS
    )
    save(api, 6, "exclude", reason=REASON)
    save(api, 7, "include")
    api.collection = api.container.paper_collections.replace(
        api.collection.collection_id,
        name=api.collection.name,
        document_ids=IDS,
        expected_revision=1,
    )
    for index, decision, reason in (
        (0, "include", "Human include"),
        (1, "exclude", REASON),
        (2, "exclude", REASON),
        (3, "exclude", REASON.lower()),
        (4, "exclude", " " + REASON),
        (5, "unsure", "Needs a human discussion"),
    ):
        save(api, index, decision, reason=reason)
    return api


def endpoint(api: ExportAPI) -> str:
    return f"/collections/{api.collection.collection_id}/screening-report"


def download(api: ExportAPI, format: str = "json") -> httpx.Response:
    response = api.client.get(
        endpoint(api),
        params={"collection_revision": api.collection.revision, "format": format},
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-type"] == (
        "application/json" if format == "json" else "text/markdown; charset=utf-8"
    )
    extension = "json" if format == "json" else "md"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="screening-report-{api.collection.collection_id}'
        f'-r{api.collection.revision}.{extension}"'
    )
    assert int(response.headers["content-length"]) == len(response.content)
    assert "PRIVATE_" not in response.text
    return response


def snapshot(api: ExportAPI) -> ScreeningExport:
    return ScreeningExport.model_validate_json(
        api.container.paper_screening.export_results(
            api.collection.collection_id, collection_revision=api.collection.revision
        ).content
    )


def failure(api: ExportAPI, status: int, code: str, **params: object) -> None:
    response = api.client.get(
        endpoint(api), params={"collection_revision": api.collection.revision, **params}
    )
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-disposition" not in response.headers
    assert "snapshot" not in response.json()
    assert "PRIVATE_" not in response.text


def test_complete_report_partitions_same_snapshot_and_groups_only_current_excludes(
    report_api: ExportAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = report_api
    expected = snapshot(api)
    exported = api.container.paper_screening.export_results(
        api.collection.collection_id, collection_revision=2
    ).content
    export = Mock(wraps=api.container.paper_screening.export_results)
    monkeypatch.setattr(api.container.paper_screening, "export_results", export)
    report = ScreeningReport.model_validate_json(download(api).content)
    export.assert_called_once_with(
        api.collection.collection_id, collection_revision=2, format="json"
    )
    assert report.snapshot == expected
    assert report.snapshot_sha256 == hashlib.sha256(exported).hexdigest()
    assert report.snapshot.collection_name == api.collection.name
    assert report.snapshot.collection_revision == 2
    assert report.snapshot.total_documents == 9
    assert report.snapshot.counts.model_dump() == {
        "include": 1,
        "exclude": 4,
        "unsure": 1,
        "stale": 2,
        "unscreened": 1,
    }
    assert report.resolved_documents == 5
    assert report.needs_review_documents == 4
    assert report.resolved_document_ids == IDS[:5]
    assert report.needs_review_document_ids == IDS[5:]
    assert report.resolved_documents + report.needs_review_documents == len(IDS)
    assert [
        (group.reason, group.count, group.document_ids) for group in report.exclusion_reasons
    ] == [
        (REASON, 2, IDS[1:3]),
        (REASON.lower(), 1, (IDS[3],)),
        (" " + REASON, 1, (IDS[4],)),
    ]
    assert sum(group.count for group in report.exclusion_reasons) == report.snapshot.counts.exclude
    assert report.snapshot.items[6].status == report.snapshot.items[7].status == "stale"
    assert [item.document_id for item in report.snapshot.items] == list(IDS)
    assert report.snapshot.included_document_ids == (IDS[0],)


def test_json_and_literal_markdown_reproduce_offline_without_requery(report_api: ExportAPI) -> None:
    report = build_screening_report(snapshot(report_api))
    assert report.to_json().encode("utf-8") == download(report_api).content
    markdown = download(report_api, "markdown").text
    assert markdown == report.to_markdown()
    assert markdown.startswith("# Human screening progress report\n")
    assert literal_block(REASON) in markdown
    assert "```````text\n" in markdown
    assert literal_block(json.dumps(IDS[1:3], ensure_ascii=True, indent=2), "json") in markdown
    assert "| exclude | 4 |" in markdown
    assert "Resolved" in markdown and "Still needing human review" in markdown
    assert "not PRISMA" in markdown
    assert "not scientific validation" in markdown
    assert "case" in markdown and "whitespace" in markdown
    assert "not frozen" in markdown
    assert ScreeningReport.model_validate_json(report.to_json()).to_markdown() == markdown
    blocks = re.findall(r"(`{3,})json\n(.*?)\n\1", markdown, re.DOTALL)
    assert report.snapshot.model_dump(mode="json") in [json.loads(block) for _, block in blocks]


def test_report_reads_do_not_generate_retrieve_write_or_scan_history_and_restart_is_identical(
    report_api: ExportAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = report_api
    before = api.database.read_bytes()
    with browsing_guards(api.container) as guards:
        for target, method in (
            (api.container.paper_screening, "submit"),
            (api.container.paper_screening, "list_queue"),
            (api.container.paper_screening, "list_history"),
            (api.container.event_log, "append_transition"),
            (socket, "create_connection"),
            (socket.socket, "connect"),
        ):
            monkeypatch.setattr(target, method, denied)
        expected = {}
        for format in ("json", "markdown"):
            direct = export_screening_report(
                api.container.paper_screening,
                api.collection.collection_id,
                collection_revision=2,
                format=format,
            )
            expected[format] = download(api, format).content
            assert direct.content == expected[format]
        assert not any(guard.call_count for guard in guards.values())
    assert api.database.read_bytes() == before
    assert api.container.event_log.list_events() == []
    restarted = create_app(offline_settings(api.database))
    with TestClient(restarted) as client, browsing_guards(restarted.state.container) as guards:
        for format, content in expected.items():
            response = client.get(
                endpoint(api), params={"collection_revision": 2, "format": format}
            )
            assert response.content == content
        assert not any(guard.call_count for guard in guards.values())
    assert api.database.read_bytes() == before


def test_unscreened_unsure_and_stale_are_unresolved_and_rename_invalidates_old_decisions(
    report_api: ExportAPI,
) -> None:
    api = report_api
    old = download(api).content
    api.collection = api.container.paper_collections.replace(
        api.collection.collection_id, name="Renamed review", document_ids=IDS, expected_revision=2
    )
    failure(api, 409, "collection_revision_conflict", collection_revision=2)
    report = ScreeningReport.model_validate_json(download(api).content)
    assert report.resolved_documents == 0 and report.needs_review_documents == 9
    assert report.exclusion_reasons == ()
    assert report.snapshot.counts.stale == 8 and report.snapshot.counts.unscreened == 1
    assert report.snapshot.included_document_ids == ()
    assert report.needs_review_document_ids == IDS
    assert ScreeningReport.model_validate_json(old).resolved_documents == 5
    api.container.paper_screening.submit(
        api.collection.collection_id,
        IDS[6],
        ScreeningSubmission(
            collection_revision=3,
            expected_decision_revision=1,
            decision="exclude",
            reason=REASON,
        ),
    )
    changed = ScreeningReport.model_validate_json(download(api).content)
    assert changed.snapshot.collection_revision == report.snapshot.collection_revision
    assert changed.snapshot_sha256 != report.snapshot_sha256
    assert changed.resolved_document_ids == (IDS[6],)
    assert changed.exclusion_reasons[0].document_ids == (IDS[6],)


def test_unscreened_collection_has_no_invented_exclusions(export_api: ExportAPI) -> None:
    report = ScreeningReport.model_validate_json(download(export_api).content)
    assert report.resolved_documents == 0 and report.needs_review_documents == 5
    assert report.exclusion_reasons == ()
    assert report.snapshot.counts.unscreened == report.snapshot.total_documents
    assert "No current exclusions" in report.to_markdown()


@pytest.mark.parametrize("format", ["json", "markdown"])
def test_serialized_utf8_cap_includes_reason_repetition_escaping_and_newlines(
    report_api: ExportAPI, monkeypatch: pytest.MonkeyPatch, format: str
) -> None:
    api = report_api
    original = download(api, format)
    assert len(original.content) > len(original.text)
    monkeypatch.setattr(screening_report, "MAX_REPORT_BYTES", len(original.content))
    assert download(api, format).content == original.content
    monkeypatch.setattr(screening_report, "MAX_REPORT_BYTES", len(original.content) - 1)
    failure(api, 413, "screening_report_too_large", format=format)
    with pytest.raises(CollectionError, match="complete"):
        export_screening_report(
            api.container.paper_screening,
            api.collection.collection_id,
            collection_revision=2,
            format=format,
        )


def test_report_cannot_bypass_existing_complete_export_cap(
    report_api: ExportAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = snapshot(report_api)
    monkeypatch.setattr(paper_screening, "MAX_EXPORT_BYTES", 1)
    for format in ("json", "markdown"):
        failure(report_api, 413, "screening_export_too_large", format=format)
    with pytest.raises(CollectionError) as error:
        build_screening_report(current)
    assert error.value.code == "screening_export_too_large"


@pytest.mark.parametrize(
    "query",
    [
        "",
        "collection_revision=0",
        "collection_revision=-1",
        "collection_revision=2.0",
        "collection_revision=true",
        "collection_revision=%2B2",
        "collection_revision=%202",
        "collection_revision=%EF%BC%92",
        f"collection_revision={MAX_REVISION + 1}",
        "collection_revision=" + "2" * 20,
        "collection_revision=2&format=csv",
        "collection_revision=2&format=MARKDOWN",
        "collection_revision=2&format=",
        "collection_revision=2&format=json&format=markdown",
        "collection_revision=2&collection_revision=2",
        "collection_revision=2&limit=1",
        "collection_revision=2&status=exclude",
        "collection_revision=2&format=%FF",
        "collection_revision=2&format=json%00",
        "collection_revision=2&%FF=x",
        "collection_revision=2&cursor=" + "x" * 16384,
    ],
)
def test_request_validation_is_bounded_strict_and_precedes_any_read(
    report_api: ExportAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    query: str,
) -> None:
    monkeypatch.setattr(report_api.container.paper_screening, "export_results", denied)
    response = report_api.client.get(endpoint(report_api) + "?" + query)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_screening_report_request"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-disposition" not in response.headers
    assert "invalid_screening_report_request" in caplog.text


@pytest.mark.parametrize("revision", [True, False, 2.0, "2", 0, -1, MAX_REVISION + 1])
def test_python_input_validation_precedes_the_snapshot_read(
    report_api: ExportAPI, monkeypatch: pytest.MonkeyPatch, revision: object
) -> None:
    monkeypatch.setattr(report_api.container.paper_screening, "export_results", denied)
    with pytest.raises(ValidationError):
        export_screening_report(
            report_api.container.paper_screening,
            report_api.collection.collection_id,
            collection_revision=revision,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("format", ["CSV", "", "json\0", "\ud800", 5])
def test_invalid_python_formats_do_not_read(
    report_api: ExportAPI, monkeypatch: pytest.MonkeyPatch, format: object
) -> None:
    monkeypatch.setattr(report_api.container.paper_screening, "export_results", denied)
    with pytest.raises(ValidationError):
        export_screening_report(
            report_api.container.paper_screening,
            report_api.collection.collection_id,
            collection_revision=2,
            format=format,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("method", ["PUT", "POST", "PATCH", "DELETE", "HEAD"])
def test_download_route_cannot_be_a_document_screening_write(
    report_api: ExportAPI, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    monkeypatch.setattr(report_api.container.paper_screening, "submit", denied)
    monkeypatch.setattr(report_api.container.paper_screening, "export_results", denied)
    response = report_api.client.request(method, endpoint(report_api))
    assert response.status_code == 405
    assert response.headers["allow"] == "GET"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    ("sql", "status", "code"),
    [
        ("DELETE FROM paper_collections", 404, "collection_not_found"),
        ("DELETE FROM documents", 409, "collection_documents_missing"),
        ("UPDATE paper_collections SET document_ids = '[]'", 409, "invalid_collection_record"),
        (
            "UPDATE paper_screening_decisions SET record = 'PRIVATE_CORRUPT'",
            409,
            "invalid_screening_record",
        ),
        ("UPDATE documents SET title = CAST(X'FF' AS TEXT)", 409, "invalid_document_record"),
        ("UPDATE documents SET source = 'prefix' || char(0)", 409, "invalid_document_record"),
        ("DROP TABLE paper_screening_decisions", 503, "collection_storage_error"),
    ],
)
def test_missing_corrupt_and_unavailable_inputs_fail_without_partial_artifacts(
    report_api: ExportAPI,
    caplog: pytest.LogCaptureFixture,
    sql: str,
    status: int,
    code: str,
) -> None:
    with sqlite3.connect(report_api.database) as connection:
        connection.execute(sql)
    for format in ("json", "markdown"):
        failure(report_api, status, code, format=format)
    assert code in caplog.text and "PRIVATE_CORRUPT" not in caplog.text


@pytest.mark.parametrize(
    "kind",
    [
        "counts",
        "total",
        "duplicate",
        "include_ids",
        "status",
        "foreign_review",
        "unicode",
        "nul",
        "name",
    ],
)
def test_pure_builder_rejects_forged_or_partial_snapshots(report_api: ExportAPI, kind: str) -> None:
    original = snapshot(report_api)
    if kind == "counts":
        forged = original.model_copy(
            update={"counts": original.counts.model_copy(update={"exclude": 0})}
        )
    elif kind == "total":
        forged = original.model_copy(update={"total_documents": 1})
    elif kind == "duplicate":
        forged = original.model_copy(update={"items": [original.items[0]] * 9})
    elif kind == "include_ids":
        forged = original.model_copy(update={"included_document_ids": (IDS[7],)})
    elif kind == "name":
        forged = original.model_copy(update={"collection_name": " padded "})
    else:
        item = original.items[1]
        assert item.review is not None
        if kind == "status":
            item = item.model_copy(update={"status": "include"})
        elif kind == "foreign_review":
            item = item.model_copy(
                update={"review": item.review.model_copy(update={"document_id": "wrong"})}
            )
        else:
            item = item.model_copy(
                update={"title": "\ud800" if kind == "unicode" else "bad\0label"}
            )
        forged = original.model_copy(
            update={"items": [original.items[0], item, *original.items[2:]]}
        )
    with pytest.raises(CollectionError) as error:
        build_screening_report(forged)
    assert error.value.code == "invalid_screening_snapshot"


def test_tampered_portable_report_cannot_serialize_success(report_api: ExportAPI) -> None:
    report = build_screening_report(snapshot(report_api))
    forged = report.model_copy(update={"resolved_documents": 0})
    for renderer in (forged.to_json, forged.to_markdown):
        with pytest.raises(CollectionError) as error:
            renderer()
        assert error.value.code == "invalid_screening_report"
    data = json.loads(report.to_json())
    data["exclusion_reasons"][0]["document_ids"] = [IDS[7]]
    with pytest.raises(ValidationError):
        ScreeningReport.model_validate(data)


def test_wal_writer_cannot_mix_report_counts_reasons_name_and_labels(
    report_api: ExportAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = report_api
    expected = download(api).content
    with sqlite3.connect(api.database) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    original = api.container.paper_screening._reviews

    def write_during_snapshot(
        connection: sqlite3.Connection, collection: PaperCollection
    ) -> dict[str, ScreeningReview]:
        assert connection.in_transaction
        reviews = original(connection, collection)
        with sqlite3.connect(api.database, timeout=1) as writer:
            writer.execute(
                "UPDATE paper_collections SET name = 'Later name', revision = 3 "
                "WHERE collection_id = ?",
                (collection.collection_id,),
            )
            writer.execute("UPDATE documents SET title = 'Later label'")
            writer.execute("DELETE FROM paper_screening_decisions")
        return reviews

    with monkeypatch.context() as patch:
        patch.setattr(api.container.paper_screening, "_reviews", write_during_snapshot)
        assert download(api).content == expected
    failure(api, 409, "collection_revision_conflict")


def test_complete_membership_labels_flags_and_opaque_ids_are_not_deduplicated(
    export_api: ExportAPI,
) -> None:
    api = export_api
    ids = (
        "nul\0id",
        "//external.invalid/x",
        "quotes\"'?%#&+\\",
        "cafe\u0301/研究",
        *(f"paper-{index}" for index in reversed(range(96))),
    )
    api.container.document_store.add_documents(
        [
            Document(document_id=i, title="T" * 301, source="S" * 513, text="PRIVATE_BODY")
            for i in ids
        ],
        [],
    )
    api.collection = api.container.paper_collections.create(name="100 members", document_ids=ids)
    save(api, 99, "exclude", reason="Last member")
    report = ScreeningReport.model_validate_json(download(api).content)
    assert report.snapshot.total_documents == 100
    assert [item.document_id for item in report.snapshot.items] == list(ids)
    assert report.exclusion_reasons[0].document_ids == (ids[-1],)
    assert report.needs_review_document_ids == ids[:-1]
    assert all(
        item.title == "T" * 300
        and item.title_truncated
        and item.source == "S" * 512
        and item.source_truncated
        for item in report.snapshot.items
    )
    assert "\\u0000" in report.to_markdown() and "\0" not in report.to_markdown()


def test_real_large_unicode_snapshot_fails_instead_of_becoming_an_empty_report(
    export_api: ExportAPI,
) -> None:
    api = export_api
    ids = [f"large-{index}" for index in range(100)]
    api.container.document_store.add_documents(
        [Document(document_id=i, title="📚" * 300, source="📚" * 512, text="private") for i in ids],
        [],
    )
    api.collection = api.container.paper_collections.create(name="Large", document_ids=ids)
    for index in range(100):
        save(api, index, "exclude", reason="📚" * 1000)
    failure(api, 413, "screening_export_too_large")
    failure(api, 413, "screening_export_too_large", format="markdown")
