"""Whole-run portable human opinions, anchored to a single frozen evidence snapshot."""

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings
from scripts.demo_review_notebook import notebook_guards

from agent.models import AgentState
from api.application import create_app
from storage import review_notebook
from storage.evidence_export import MAX_EVENT_BYTES, MAX_RUN_BYTES, MAX_RUN_EVENTS, BoundedRunEvents
from storage.review_notebook import NotebookExportError, SQLiteReviewNotebook
from tests.test_corpus_explorer import ExplorerAPI, ExplorerDOM, href, values
from tests.test_corpus_explorer import api as api
from tests.test_review_notebook import (
    COMMENT,
    NOTEBOOK,
    PRIVATE,
    TEXT,
    annotate,
    fixture_run,
    mutate,
    opinion,
)
from tests.test_run_explorer import complete, record

EXPORT = "/runs/saved/notebook"
DOWNLOAD = "/explore/run/notebook/export"


def test_export_keeps_complete_histories_and_original_provenance(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    count = 103
    for number in range(count):
        opinion(api, bundle, f"Review {number}: {COMMENT}")
        annotate(api, bundle, note=f"Note {number}")
    other = fixture_run(api, run_id="other")
    opinion(api, other, "EXCLUDED_OTHER_RUN")
    annotate(api, other, note="EXCLUDED_OTHER_RUN")
    visible = api.page(NOTEBOOK, run_id=bundle.run_id, review_limit=1, annotation_limit=1)
    assert len(values(visible, "review-comment")) == 1

    response = api.client.get(EXPORT)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["schema_version"] == "1.0"
    assert result["evidence"] == bundle.model_dump(mode="json")
    assert result["history_order"] == "sequence_desc"
    assert result["consistency"] == "single_sqlite_read_transaction"
    assert len(result["reviews"]) == len(result["annotations"]) == count
    assert [item["sequence"] for item in result["reviews"]] == list(range(count, 0, -1))
    assert [item["sequence"] for item in result["annotations"]] == list(range(count, 0, -1))
    review_page = api.container.answer_reviews.list_reviews(bundle.run_id, limit=100)
    annotation_page = api.container.evidence_annotations.list_annotations(bundle.run_id, limit=100)
    assert result["reviews"][:100] == review_page.model_dump(mode="json")["reviews"]
    assert result["annotations"][:100] == annotation_page.model_dump(mode="json")["annotations"]
    assert result["reviews"][-1]["comment"] == f"Review 0: {COMMENT}"
    assert result["annotations"][-1]["note"] == "Note 0"
    assert "EXCLUDED_OTHER_RUN" not in response.text
    assert "next_cursor" not in result and "latest_verdict" not in result
    assert response.headers["content-type"] == "application/json"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert re.fullmatch(
        r'attachment; filename="review-notebook-[0-9a-f]{16}\.json"',
        response.headers["content-disposition"],
    )


def test_markdown_keeps_literal_quotes_offsets_and_full_frozen_evidence(api: ExplorerAPI) -> None:
    hostile = "\n``````\n# Not a heading\n[link](javascript:alert(1))\n<script>x</script>\n"
    bundle = fixture_run(api, text=TEXT + hostile)
    opinion(api, bundle, hostile)
    saved = annotate(api, bundle, note=hostile)
    response = api.client.get(EXPORT, params={"format": "markdown"})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "text/markdown; charset=utf-8"
    assert response.text.startswith("# Portable human-review notebook\n")
    assert TEXT + hostile in response.text
    assert "```````text\n" in response.text
    assert str(saved.annotation_id) in response.text
    assert saved.source_text_sha256 in response.text
    assert f"[{saved.start}, {saved.end})" in response.text
    assert saved.start == (TEXT + hostile).rindex("Repeated quote")
    assert "Human opinions" in response.text
    assert "not scientific validation" in response.text
    assert "not authenticated" in response.text
    assert "single read-only SQLite snapshot" in response.text


@pytest.mark.parametrize("run_id", [".", "..", "paper/../\u7814?x#v1", "'\"?%#&+\\", "run"])
@pytest.mark.parametrize("mounted", [False, True])
def test_native_downloads_preserve_exact_ids_and_mounts(
    api: ExplorerAPI, run_id: str, mounted: bool
) -> None:
    fixture_run(api, run_id=run_id)
    application = api.app
    prefix = "/prefix"
    if mounted:
        application = FastAPI()
        application.mount("/library", api.app)
        prefix += "/library"
    with TestClient(application, root_path="/prefix") as client:
        page = client.get(prefix + NOTEBOOK, params={"run_id": run_id, "review_limit": 1})
        assert page.status_code == 200
        root = ExplorerDOM(page.text).root
        for format in ("json", "markdown"):
            link = href(root, "download-notebook-" + format)
            assert urlsplit(link).path == prefix + DOWNLOAD
            assert parse_qs(urlsplit(link).query) == {"run_id": [run_id], "format": [format]}
            assert "download" in root.by_id("download-notebook-" + format).attrs
            response = client.get(link)
            assert response.status_code == 200, response.text
            assert response.headers["content-disposition"].startswith("attachment;")
            if format == "json":
                assert response.json()["evidence"]["run_id"] == run_id


def test_export_is_offline_readonly_and_stable_after_corpus_deletion_restart(
    api: ExplorerAPI,
) -> None:
    bundle = complete(api)
    opinion(api, bundle)
    annotate(api, bundle, "Synthetic graph evidence")
    url = DOWNLOAD + "?" + urlencode({"run_id": bundle.run_id})
    events = api.container.event_log.list_events()
    expected = {}
    with notebook_guards(api.container) as guards:
        before = api.path.read_bytes()
        for format in ("json", "markdown"):
            response = api.client.get(url, params={"run_id": bundle.run_id, "format": format})
            assert response.status_code == 200, response.text
            expected[format] = response.content
        assert not any(guard.call_count for guard in guards.values())
        assert api.path.read_bytes() == before
    for stage in ("changed", "deleted"):
        with closing(sqlite3.connect(api.path)) as connection, connection:
            if stage == "changed":
                connection.execute("UPDATE chunks SET text = 'REPLACED CURRENT CORPUS'")
            else:
                for table in (
                    "entity_edges",
                    "entity_mentions",
                    "graph_chunks",
                    "chunks",
                    "documents",
                ):
                    connection.execute(f"DELETE FROM {table}")  # noqa: S608
        restarted = create_app(offline_settings(api.path))
        before = api.path.read_bytes()
        with TestClient(restarted) as client, notebook_guards(restarted.state.container) as guards:
            for format, content in expected.items():
                response = client.get(DOWNLOAD, params={"run_id": bundle.run_id, "format": format})
                assert response.status_code == 200
                assert response.content == content
            assert not any(guard.call_count for guard in guards.values())
        assert api.path.read_bytes() == before
        assert restarted.state.container.event_log.list_events() == events


@pytest.mark.parametrize(
    "state,code,status",
    [
        (None, "run_not_found", 404),
        (AgentState.REASONING, "run_incomplete", 409),
        (AgentState.ERROR, "run_failed", 409),
        (AgentState.DONE, "snapshot_unavailable", 409),
    ],
)
def test_unexportable_is_not_an_empty_success(
    api: ExplorerAPI, state: AgentState | None, code: str, status: int
) -> None:
    if state is not None:
        record(api, "saved", state)
    response = api.client.get(EXPORT)
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"
    assert "content-disposition" not in response.headers


@pytest.mark.parametrize(
    "sql,code",
    [
        ("UPDATE answer_reviews SET created_at = 'PRIVATE_DIAGNOSTIC'", "invalid_review_record"),
        ("UPDATE answer_reviews SET cited_chunk_ids = '[\"missing\"]'", "invalid_review_record"),
        ("UPDATE evidence_annotations SET start = end", "invalid_annotation_record"),
        (
            "UPDATE evidence_annotations SET quote = 'PRIVATE_DIAGNOSTIC'",
            "invalid_annotation_record",
        ),
        (
            "UPDATE evidence_annotations SET source_text_sha256 = 'wrong'",
            "invalid_annotation_record",
        ),
        (
            "UPDATE agent_events SET payload = '{}' WHERE event_type = 'evidence_snapshot'",
            "invalid_run_record",
        ),
    ],
)
def test_corrupt_records_fail_completely_without_diagnostics(
    api: ExplorerAPI, sql: str, code: str, caplog: pytest.LogCaptureFixture
) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    annotate(api, bundle)
    mutate(api, sql)
    for format in ("json", "markdown"):
        response = api.client.get(EXPORT, params={"format": format})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == code
        assert "content-disposition" not in response.headers
        assert PRIVATE not in response.text + caplog.text


@pytest.mark.parametrize(
    "query",
    ["format=html", "format=json&format=markdown", "limit=1", "run_id=other", "format=%FF"],
)
def test_export_rejects_page_filters_duplicate_or_invalid_format(
    api: ExplorerAPI, query: str, caplog: pytest.LogCaptureFixture
) -> None:
    response = api.client.get(EXPORT + "?" + query)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_notebook_export_request"
    assert response.headers["cache-control"] == "no-store"
    assert PRIVATE not in response.text + caplog.text


def test_existing_json_opinion_apis_keep_their_shapes_and_cursor_behavior(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    for _ in range(2):
        opinion(api, bundle)
        annotate(api, bundle)
    before = {
        kind: api.client.get(f"/runs/saved/{kind}", params={"limit": 1}).json()
        for kind in ("reviews", "annotations")
    }
    response = api.client.get(EXPORT)
    assert response.status_code == 200
    for kind, page in before.items():
        assert set(page) == {kind, "next_cursor"}
        assert page["next_cursor"] is not None
        assert api.client.get(f"/runs/saved/{kind}", params={"limit": 1}).json() == page
    assert json.loads(response.content)["evidence"]["answer"] == bundle.answer.model_dump(
        mode="json"
    )


def _append_copies(connection: sqlite3.Connection, count: int, kind: str) -> None:
    if kind == "review":
        statement = (
            "INSERT INTO answer_reviews "
            "(schema_version, run_id, review_id, created_at, decision, comment, cited_chunk_ids) "
            "SELECT schema_version, run_id, ?, created_at, decision, comment, cited_chunk_ids "
            "FROM answer_reviews WHERE run_id = 'saved' ORDER BY id LIMIT 1"
        )
    else:
        statement = (
            "INSERT INTO evidence_annotations "
            "(schema_version, run_id, annotation_id, document_id, chunk_id, source_text_sha256, "
            "start, end, quote, note, created_at) "
            "SELECT schema_version, run_id, ?, document_id, chunk_id, source_text_sha256, "
            "start, end, quote, note, created_at FROM evidence_annotations "
            "WHERE run_id = 'saved' ORDER BY sequence LIMIT 1"
        )
    connection.executemany(statement, [(str(uuid4()),) for _ in range(count)])


def _seed_histories(api: ExplorerAPI, count: int) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    annotate(api, bundle)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        for kind in ("review", "annotation"):
            _append_copies(connection, count - 1, kind)


@pytest.mark.parametrize("kind", ["review", "annotation"])
def test_real_record_maximum_is_complete_and_one_extra_fails_413(
    api: ExplorerAPI, kind: str
) -> None:
    assert review_notebook.MAX_NOTEBOOK_REVIEWS == review_notebook.MAX_NOTEBOOK_ANNOTATIONS == 1000
    _seed_histories(api, 1000)
    response = api.client.get(EXPORT)
    assert response.status_code == 200
    assert len(response.json()["reviews"]) == len(response.json()["annotations"]) == 1000
    assert response.json()["reviews"][-1]["sequence"] == 1
    with closing(sqlite3.connect(api.path)) as connection, connection:
        _append_copies(connection, 1, kind)
    for format in ("json", "markdown"):
        response = api.client.get(EXPORT, params={"format": format})
        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "notebook_export_too_large"
        assert "content-disposition" not in response.headers


def test_all_three_histories_share_one_snapshot_during_atomic_append(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_histories(api, 1)
    with closing(sqlite3.connect(api.path)) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    original = BoundedRunEvents.list_events
    appended = False

    def after_events(self: BoundedRunEvents, run_id: str | None = None) -> list[dict[str, object]]:
        nonlocal appended
        events = original(self, run_id)
        if not appended:
            with closing(sqlite3.connect(api.path)) as writer, writer:
                _append_copies(writer, 1, "review")
                _append_copies(writer, 1, "annotation")
            appended = True
        return events

    monkeypatch.setattr(BoundedRunEvents, "list_events", after_events)
    first = api.client.get(EXPORT)
    assert first.status_code == 200 and appended
    assert len(first.json()["reviews"]) == len(first.json()["annotations"]) == 1
    second = api.client.get(EXPORT)
    assert second.status_code == 200
    assert len(second.json()["reviews"]) == len(second.json()["annotations"]) == 2
    assert first.json()["evidence"] == second.json()["evidence"]


def test_read_only_connection_is_single_closed_and_cannot_write(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_histories(api, 1)
    connect = sqlite3.connect
    connections = []
    statements: list[str] = []

    def observed(database: str, **kwargs: object) -> sqlite3.Connection:
        assert database.endswith("?mode=ro")
        connection = connect(database, **kwargs)
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE forbidden (id INTEGER)")
        connection.set_trace_callback(statements.append)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", observed)
    result = api.container.review_notebook.export("saved")
    assert len(result.reviews) == len(result.annotations) == 1
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
    assert statements.count("BEGIN") == 1
    assert all(
        statement.lstrip().split()[0] in {"BEGIN", "PRAGMA", "SELECT"} for statement in statements
    )
    assert any("agent_events" in statement for statement in statements)
    assert any("answer_reviews" in statement for statement in statements)
    assert any("evidence_annotations" in statement for statement in statements)


def _review_sizes(connection: sqlite3.Connection) -> list[tuple[int, int]]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute("SELECT * FROM answer_reviews ORDER BY id").fetchall()
    return [
        (
            row["id"],
            sum(
                len(value.encode("utf-8")) for value in dict(row).values() if isinstance(value, str)
            ),
        )
        for row in rows
    ]


def test_exact_default_history_byte_budget_includes_all_stored_utf8_text(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    assert review_notebook.MAX_HISTORY_BYTES == 4 * 1024 * 1024
    with closing(sqlite3.connect(api.path)) as connection, connection:
        _append_copies(connection, 16, "review")
        connection.execute("UPDATE answer_reviews SET cited_chunk_ids = '[]'")
        sizes = _review_sizes(connection)
        share, remainder = divmod(review_notebook.MAX_HISTORY_BYTES, len(sizes))
        for number, (sequence, size) in enumerate(sizes):
            padding = share + (number < remainder) - size
            connection.execute(
                "UPDATE answer_reviews SET cited_chunk_ids = ? WHERE id = ?",
                ("[" + " " * padding + "]", sequence),
            )
        assert (
            sum(size for _, size in _review_sizes(connection)) == review_notebook.MAX_HISTORY_BYTES
        )
    assert api.client.get(EXPORT).status_code == 200
    mutate(api, "UPDATE answer_reviews SET cited_chunk_ids = cited_chunk_ids || ' ' WHERE id = 1")
    assert api.client.get(EXPORT).status_code == 413


def test_exact_review_record_byte_limit_is_preflighted(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE answer_reviews SET cited_chunk_ids = '[]'")
        _, size = _review_sizes(connection)[0]
        padding = review_notebook.MAX_REVIEW_RECORD_BYTES - size
        connection.execute(
            "UPDATE answer_reviews SET cited_chunk_ids = ?", ("[" + " " * padding + "]",)
        )
    assert api.client.get(EXPORT).status_code == 200
    mutate(api, "UPDATE answer_reviews SET cited_chunk_ids = cited_chunk_ids || ' '")
    assert api.client.get(EXPORT).status_code == 413


def test_both_serialized_formats_must_fit_exact_utf8_limit(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_histories(api, 1)
    result = api.container.review_notebook.export("saved")
    content = {format: getattr(result, "to_" + format)() for format in ("json", "markdown")}
    limit = max(len(text.encode("utf-8")) for text in content.values())
    assert limit > max(len(text) for text in content.values())
    monkeypatch.setattr(review_notebook, "MAX_EXPORT_BYTES", limit)
    for format in content:
        response = api.client.get(EXPORT, params={"format": format})
        assert response.status_code == 200
        assert response.text == content[format]
    monkeypatch.setattr(review_notebook, "MAX_EXPORT_BYTES", limit - 1)
    for format in content:
        response = api.client.get(EXPORT, params={"format": format})
        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "notebook_export_too_large"


def test_actual_markdown_expansion_cannot_be_bypassed_by_requesting_json(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle, '"' * 4000)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        _append_copies(connection, 699, "review")
        assert (
            sum(size for _, size in _review_sizes(connection)) < review_notebook.MAX_HISTORY_BYTES
        )
    assert review_notebook.MAX_EXPORT_BYTES == 8 * 1024 * 1024
    for format in ("json", "markdown"):
        response = api.client.get(EXPORT, params={"format": format})
        assert response.status_code == 413
        assert "content-disposition" not in response.headers


@pytest.mark.parametrize("overflow", ["count", "payload", "total"])
def test_event_read_limits_map_to_notebook_413_even_after_done(
    api: ExplorerAPI, overflow: str
) -> None:
    bundle = fixture_run(api)
    if overflow == "count":
        for _ in range(MAX_RUN_EVENTS):
            api.container.event_log.append_event("synthetic-agent", "saved", "later", {})
    elif overflow == "payload":
        api.container.event_log.append_event(
            "synthetic-agent", "saved", "later", {"x": "a" * MAX_EVENT_BYTES}
        )
    else:
        for _ in range(MAX_RUN_BYTES // MAX_EVENT_BYTES + 1):
            api.container.event_log.append_event(
                "synthetic-agent", "saved", "later", {"x": "a" * (MAX_EVENT_BYTES - 100)}
            )
    response = api.client.get(EXPORT)
    assert response.status_code == 413
    # The old API intentionally keeps the original first-terminal-event behavior.
    assert api.container.evidence_exporter.export("saved") == bundle
    assert api.client.get("/runs/saved/reviews").status_code == 200


@pytest.mark.parametrize("kind", ["review", "annotation"])
def test_corruption_beyond_maximum_visible_page_is_not_hidden(api: ExplorerAPI, kind: str) -> None:
    _seed_histories(api, 102)
    table, sequence = (
        ("answer_reviews", "id") if kind == "review" else ("evidence_annotations", "sequence")
    )
    mutate(api, f"UPDATE {table} SET created_at = 'PRIVATE_DIAGNOSTIC' WHERE {sequence} = 1")  # noqa: S608
    assert api.client.get(EXPORT).status_code == 409
    assert api.client.get(f"/runs/saved/{kind}s", params={"limit": 100}).status_code == 200


@pytest.mark.parametrize(
    "sql,code",
    [
        ("UPDATE answer_reviews SET comment = CAST(X'FF' AS TEXT)", "invalid_review_record"),
        ("UPDATE evidence_annotations SET note = CAST(X'FF' AS TEXT)", "invalid_annotation_record"),
        ("UPDATE answer_reviews SET review_id = upper(review_id)", "invalid_review_record"),
        (
            "UPDATE evidence_annotations SET annotation_id = upper(annotation_id)",
            "invalid_annotation_record",
        ),
        ("UPDATE evidence_annotations SET schema_version = '2.0'", "invalid_annotation_record"),
        ("UPDATE answer_reviews SET cited_chunk_ids = '[1]'", "invalid_review_record"),
    ],
)
def test_invalid_utf8_and_unsupported_opinion_records_fail_safely(
    api: ExplorerAPI, sql: str, code: str
) -> None:
    _seed_histories(api, 1)
    mutate(api, sql)
    response = api.client.get(EXPORT)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == code


@pytest.mark.parametrize("table", ["agent_events", "answer_reviews", "evidence_annotations"])
def test_missing_table_is_unavailable_not_empty(api: ExplorerAPI, table: str) -> None:
    fixture_run(api)
    mutate(api, f"DROP TABLE {table}")
    response = api.client.get(EXPORT)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "notebook_storage_unavailable"


def test_python_reader_does_not_initialize_a_missing_database(tmp_path: Path) -> None:
    path = tmp_path / "absent.sqlite3"
    reader = SQLiteReviewNotebook(path)
    assert not path.exists()
    with pytest.raises(NotebookExportError) as failure:
        reader.export("saved")
    assert failure.value.status_code == 503
    assert not path.exists()


@pytest.mark.parametrize("run_id", ["", "x" * 257, "\ud800", 1, None])
def test_python_reader_rejects_invalid_ids_without_io(tmp_path: Path, run_id: object) -> None:
    path = tmp_path / "absent.sqlite3"
    with pytest.raises(NotebookExportError) as failure:
        SQLiteReviewNotebook(path).export(run_id)  # type: ignore[arg-type]
    assert failure.value.status_code == 422
    assert not path.exists()


@pytest.mark.parametrize("encoding", ["UTF-16le", "UTF-16be"])
def test_utf16_storage_uses_exact_utf8_limits_and_shared_validators(
    tmp_path: Path, encoding: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "utf16.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(f"PRAGMA encoding='{encoding}'")
        connection.execute("CREATE TABLE fixture_encoding (id INTEGER)")
    app = create_app(offline_settings(path))
    with TestClient(app) as client:
        api = ExplorerAPI(app, client, app.state.container, path)
        _seed_histories(api, 1)
        response = client.get(EXPORT)
        assert response.status_code == 200
        assert response.json()["annotations"][0]["quote"] == "Repeated quote"
        with closing(sqlite3.connect(path)) as connection:
            review_bytes = _review_sizes(connection)[0][1]
            rows = connection.execute("SELECT * FROM evidence_annotations").fetchall()
            annotation_bytes = sum(
                len(value.encode("utf-8"))
                for value in dict(rows[0]).values()
                if isinstance(value, str)
            )
        monkeypatch.setattr(review_notebook, "MAX_HISTORY_BYTES", review_bytes + annotation_bytes)
        assert client.get(EXPORT).content == response.content
        monkeypatch.setattr(
            review_notebook, "MAX_HISTORY_BYTES", review_bytes + annotation_bytes - 1
        )
        assert client.get(EXPORT).status_code == 413


@pytest.mark.parametrize("path", ["/runs/%FF/notebook", "/runs/%ED%A0%80/notebook"])
def test_invalid_path_utf8_must_not_be_lossily_replaced(api: ExplorerAPI, path: str) -> None:
    response = api.client.get(path)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_notebook_export_request"


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
def test_export_method_errors_keep_private_download_headers(api: ExplorerAPI, method: str) -> None:
    response = api.client.request(method, EXPORT, json={"private": PRIVATE})
    assert response.status_code == 405
    assert response.headers["allow"] == "GET"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert PRIVATE not in response.text


def test_documented_python_example_exports_the_same_snapshot(
    api: ExplorerAPI, tmp_path: Path
) -> None:
    _seed_histories(api, 1)
    guide = Path(__file__).resolve().parents[1] / "docs/guides/PORTABLE_REVIEW_NOTEBOOK_GUIDE.md"
    blocks = re.findall(r"```python\n(.*?)```", guide.read_text(), re.DOTALL)
    assert len(blocks) == 1
    output = tmp_path / "documented-download"
    source = blocks[0].replace(
        'Path("/absolute/path/to/existing.sqlite3")', f"Path({str(api.path)!r})"
    )
    source = source.replace(
        'Path("/absolute/path/to/new-notebook-download")', f"Path({str(output)!r})"
    ).replace('"copy-the-exact-completed-run-id"', '"saved"')
    exec(compile(source, str(guide), "exec"), {})  # noqa: S102
    assert (output / "notebook.json").read_bytes() == api.client.get(EXPORT).content
    assert (output / "notebook.md").read_bytes() == api.client.get(
        EXPORT, params={"format": "markdown"}
    ).content
