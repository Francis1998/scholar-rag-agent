"""Integrated native discovery -> frozen evidence -> downloads, without executing a run."""

import base64
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from agent.evidence import EvidenceBundle
from agent.evidence_html import BrowserNavigation, render_html
from agent.models import AgentState, StateTransition
from api.application import create_app
from api.schemas import QueryResponse
from storage.evidence_export import (
    MAX_EVENT_BYTES,
    MAX_RUN_EVENTS,
    EvidenceExportError,
    SQLiteEvidenceReader,
)
from storage.run_history import MAX_EVENT_ID, MAX_ID_CHARACTERS
from tests.test_corpus_explorer import (
    HOSTILE,
    ExplorerAPI,
    ExplorerDOM,
    assert_headers,
    href,
    identities,
    no_work,
    values,
)
from tests.test_corpus_explorer import api as api
from tests.test_evidence_html import Node

QUERY = "  Synthetic graph evidence \u7814\u7a76  "
PASSAGE = "\nSynthetic graph evidence.\r\nLiteral <script>not executable</script> & cafe\u0301."


def complete(api: ExplorerAPI, query: str = QUERY) -> EvidenceBundle:
    ingested = api.client.post(
        "/ingest/text",
        json={"title": "Synthetic note", "source": "synthetic:saved-runs", "text": PASSAGE},
    )
    assert ingested.status_code == 200
    queried = api.client.post("/query", json={"query": query})
    assert queried.status_code == 200
    result = QueryResponse.model_validate_json(queried.content).result
    assert result.state == AgentState.DONE
    return api.container.evidence_exporter.export(result.run_id)


def record(
    api: ExplorerAPI,
    run_id: str,
    state: AgentState = AgentState.PLANNING,
    *,
    query: str | None = QUERY,
    agent_id: str = "synthetic-agent",
) -> int:
    return api.container.event_log.append_transition(
        StateTransition(
            agent_id=agent_id,
            run_id=run_id,
            from_state=AgentState.IDLE,
            to_state=state,
            payload={"query": query} if state == AgentState.PLANNING else {},
        )
    )


def form_parameters(root: Node) -> dict[str, str]:
    form = root.by_id("run-filters")
    params = {node.attrs["name"]: node.attrs["value"] for node in form.by_tag("input")}
    params["state"] = next(
        node.attrs["value"] for node in form.by_tag("option") if "selected" in node.attrs
    )
    return params


def reader_headers(response: httpx.Response) -> Node:
    assert response.status_code == 200, response.text[:1000]
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "content-disposition" not in response.headers
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    root = ExplorerDOM(response.text).root
    style = root.by_tag("style")
    assert len(style) == 1
    digest = base64.b64encode(hashlib.sha256(style[0].text().encode()).digest()).decode()
    csp = response.headers["content-security-policy"]
    assert f"style-src 'sha256-{digest}'" in csp
    for directive in ("default-src 'none'", "frame-ancestors 'none'", "form-action 'none'"):
        assert directive in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert len(response.content) <= 4_194_304
    return root


def test_native_catalog_history_filter_paging_evidence_downloads_and_exact_return(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    older = complete(api)
    newer = complete(api, "Newer synthetic graph evidence")
    record(api, "partial", AgentState.REASONING)
    catalog_params = {"limit": 1, "title": "note", "source": "synthetic:saved-runs"}
    events_before = api.container.event_log.list_events()
    database_before = api.path.read_bytes()
    original_export = api.container.evidence_exporter.export
    with browsing_guards(api.container) as guards:
        monkeypatch.setattr(api.container.evidence_exporter, "export", no_work)
        catalog = api.page(**catalog_params)
        history_url = href(catalog, "saved-runs")
        first = api.page(history_url)
        assert identities(first, "run-id") == ["partial", newer.run_id, older.run_id]
        assert not any("download" in node.attrs.get("id", "") for node in first.by_tag("a"))
        assert "recorded DONE alone does not make it exportable" in first.text()
        form = first.by_id("run-filters")
        labels = {node.attrs["for"] for node in form.by_tag("label")}
        for node in (*form.by_tag("input"), *form.by_tag("select")):
            if node.attrs.get("type") != "hidden":
                assert node.attrs["id"] in labels
        assert form.attrs["method"] == "get"
        params = {**form_parameters(first), "state": "DONE", "limit": "1"}
        assert "cursor" not in params
        done = api.page(form.attrs["action"], **params)
        assert identities(done, "run-id") == [newer.run_id]
        next_url = href(done, "next-page")
        next_params = parse_qs(urlsplit(next_url).query)
        assert next_params["state"] == ["DONE"] and next_params["limit"] == ["1"]
        assert next_params["source"] == ["synthetic:saved-runs"]
        second = api.page(next_url)
        assert identities(second, "run-id") == [older.run_id]
        assert not second.by_class("next-page")
        reader = reader_headers(api.client.get(href(second, "inspect-run-1")))
        assert href(reader, "back-to-runs") == next_url
        assert parse_qs(urlsplit(href(reader, "back-to-catalog")).query) == {
            key: [str(value)] for key, value in catalog_params.items()
        }
        assert values(reader, "passage") == [source.chunk.text for source in older.snapshot.sources]
        assert "FROZEN SAVED EVIDENCE, not today's corpus" in reader.text()
        assert reader.by_tag("a")[0].attrs["href"] == "#main"
        assert reader.by_id("main").attrs["tabindex"] == "-1"
        for anchor in reader.by_tag("a"):
            url = anchor.attrs["href"]
            if url.startswith("#"):
                reader.by_id(url[1:])
            else:
                assert urlsplit(url).path.startswith(("/explore", "/runs/"))
        assert identities(api.page(href(reader, "back-to-runs")), "run-id") == [older.run_id]
        monkeypatch.setattr(api.container.evidence_exporter, "export", original_export)
        for fmt in ("json", "markdown", "html"):
            response = api.client.get(href(reader, f"download-{fmt}"))
            assert response.status_code == 200
            assert response.headers["content-disposition"].startswith("attachment;")
            if fmt == "json":
                assert EvidenceBundle.model_validate_json(response.content) == older
            if fmt == "html":
                assert response.text == render_html(older)
                assert "back-to-runs" not in response.text
        assert not any(guard.call_count for guard in guards.values())
    assert events_before == api.container.event_log.list_events()
    assert database_before == api.path.read_bytes()


def test_history_is_summary_only_and_exposes_exact_previews_and_times(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    previews = (None, "", " " * 3, HOSTILE + "\U0001f52c" * 301, "\n\r\tliteral")
    for number, query in enumerate(previews):
        record(api, f"run-{number}", query=query)
    api.container.event_log.append_event(
        "synthetic-agent", "no-state", "audit", {"private": "SECRET"}
    )
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE agent_events SET timestamp = '2026-10-11T16:00:00.123456Z'")
    for target in (api.container.saved_run_evidence, api.container.evidence_exporter):
        monkeypatch.setattr(target, "export", no_work)
    monkeypatch.setattr(api.container.event_log, "list_events", no_work)
    root = api.page("/explore/runs")
    assert identities(root, "run-id") == ["no-state", *[f"run-{n}" for n in reversed(range(5))]]
    assert values(root, "recorded-state")[0] == "Not recorded"
    assert values(root, "query-preview") == [previews[4], str(previews[3])[:300], previews[2]]
    assert "No planning query was recorded" in root.text()
    assert "The recorded query is empty" in root.text()
    assert root.text().count("Query preview truncated: only the first 300 characters") == 1
    assert {node.attrs["datetime"] for node in root.by_tag("time")} == {
        "2026-10-11T16:00:00.123456+00:00"
    }
    assert all(node.text() == node.attrs["datetime"] for node in root.by_tag("time"))
    assert "SECRET" not in root.text()
    assert not any(root.by_tag(tag) for tag in ("script", "img", "base", "iframe", "object", "svg"))
    for node in root.walk():
        assert not any(name.startswith("on") or name == "style" for name in node.attrs)


def test_exclusive_creation_cursor_filters_before_lookahead_and_survives_appends(
    api: ExplorerAPI,
) -> None:
    keys = {f"run-{n}": record(api, f"run-{n}") for n in range(6)}
    for number in (0, 2, 4):
        record(api, f"run-{number}", AgentState.ERROR)
    first = api.page("/explore/runs", state="ERROR", limit=1)
    assert identities(first, "run-id") == ["run-4"]
    cursor = parse_qs(urlsplit(href(first, "next-page")).query)["cursor"]
    assert cursor == [str(keys["run-4"])]
    record(api, "new", AgentState.ERROR)
    api.container.event_log.append_event("synthetic-agent", "run-0", "audit", {})
    second = api.page(href(first, "next-page"))
    third = api.page(href(second, "next-page"))
    assert identities(second, "run-id") == ["run-2"]
    assert identities(third, "run-id") == ["run-0"]
    assert not third.by_class("next-page")
    assert identities(api.page(href(third, "first-page")), "run-id") == ["new"]
    assert "cursor" not in form_parameters(second)
    reset = api.page(href(second, "all-states"))
    assert len(reset.by_tag("article")) == 1
    assert "state" not in parse_qs(urlsplit(href(reset, "next-page")).query)


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_root_path_and_mounts_preserve_every_navigation_target(
    api: ExplorerAPI, prefix: str, mounted: bool
) -> None:
    complete(api)
    application = api.app
    root_path = prefix
    if mounted:
        application = FastAPI()
        application.mount("/library", api.app)
        root_path = "/prefix"
    with TestClient(application, root_path=root_path) as client:
        catalog = ExplorerDOM(client.get(prefix + "/explore").text).root
        history = ExplorerDOM(client.get(href(catalog, "saved-runs")).text).root
        reader = reader_headers(client.get(href(history, "inspect-run-1")))
        error = client.get(prefix + "/explore/run", params={"run_id": "missing"})
        invalid = client.get(prefix + "/explore/runs?state=done")
        for response in (error, invalid):
            assert response.status_code in (404, 422)
            assert_headers(response)
        for root in (
            catalog,
            history,
            reader,
            ExplorerDOM(error.text).root,
            ExplorerDOM(invalid.text).root,
        ):
            for anchor in root.by_tag("a"):
                path = urlsplit(anchor.attrs["href"]).path
                assert not path or path.startswith(prefix + "/")
            for form in root.by_tag("form"):
                assert form.attrs["action"].startswith(prefix + "/")
        for fmt in ("json", "markdown", "html"):
            assert client.get(href(reader, f"download-{fmt}")).status_code == 200


def rename_fixture_run(api: ExplorerAPI, original: str, renamed: str) -> None:
    with closing(sqlite3.connect(api.path)) as connection, connection:
        rows = connection.execute(
            "SELECT id, payload FROM agent_events WHERE run_id = ?", (original,)
        ).fetchall()
        for event_id, serialized in rows:
            payload = json.loads(serialized)
            if "run_id" in payload:
                payload["run_id"] = renamed
            if isinstance(payload.get("payload"), dict) and "run_id" in payload["payload"]:
                payload["payload"]["run_id"] = renamed
            connection.execute(
                "UPDATE agent_events SET run_id = ?, payload = ? WHERE id = ?",
                (renamed, json.dumps(payload), event_id),
            )


@pytest.mark.parametrize(
    "run_id",
    [
        ".",
        "..",
        "paper/../\u7814?draft#v1",
        "//example.invalid/a",
        " quotes'\"?%#&+\\ ",
        "line\r\nid\t",
        "cafe\u0301/\u202eRTL\u202c",
        "\U0001f52c" * MAX_ID_CHARACTERS,
        "nul\0id",
    ],
)
def test_exact_ids_are_never_trimmed_or_interpreted_as_paths(api: ExplorerAPI, run_id: str) -> None:
    bundle = complete(api)
    rename_fixture_run(api, bundle.run_id, run_id)
    catalog = api.page("/explore/runs")
    assert identities(catalog, "run-id") == [run_id]
    url = href(catalog, "inspect-run-1")
    assert urlsplit(url).path == "/explore/run"
    assert parse_qs(urlsplit(url).query)["run_id"] == [run_id]
    response = api.client.get(url)
    if "\0" in run_id:
        assert response.status_code == 409
        assert_headers(response)
        assert "embedded NUL" in response.text
    else:
        reader = reader_headers(response)
        assert run_id in values(reader, "literal")
        assert values(reader, "passage") == [
            source.chunk.text for source in bundle.snapshot.sources
        ]
        if "/" in run_id or run_id in {".", ".."}:
            assert "cannot use the existing path-based download routes" in reader.text()
            assert not any("download-" in node.attrs.get("id", "") for node in reader.by_tag("a"))
        else:
            download = api.client.get(href(reader, "download-json"))
            assert download.status_code == 200
            assert download.json()["run_id"] == run_id


def test_control_bearing_catalog_return_keeps_exact_links_without_lossy_forms(
    api: ExplorerAPI,
) -> None:
    record(api, "older")
    record(api, "newer")
    params = {
        "limit": 1,
        "source": "synthetic:\0\r\nsource",
        "title": "\n\rtitle",
        "catalog_cursor": "control\r\n\0id",
        "catalog_limit": 3,
    }
    first = api.page("/explore/runs", **params)
    assert not first.by_tag("form")
    assert "control characters that native inputs cannot preserve" in first.text()
    second = api.page(href(first, "next-page"))
    error = api.client.get(href(second, "inspect-run-1"))
    assert error.status_code == 409
    root = ExplorerDOM(error.text).root
    assert href(root, "back-to-runs") == href(first, "next-page")
    returned = parse_qs(urlsplit(href(root, "back-to-catalog")).query)
    assert returned == {
        "limit": ["3"],
        "source": [params["source"]],
        "title": [params["title"]],
        "cursor": [params["catalog_cursor"]],
    }


@pytest.mark.parametrize(
    "suffix",
    [
        "limit=0",
        "limit=101",
        "limit=",
        "limit=1.0",
        "limit=true",
        "limit=%EF%BC%91",
        "cursor=",
        "cursor=0",
        "cursor=-1",
        f"cursor={MAX_EVENT_ID + 1}",
        "cursor=1e1",
        "state=done",
        "state=null",
        "state=%20DONE",
        "state=DONE&state=ERROR",
        "limit=1&limit=2",
        "cursor=1&cursor=2",
        "source=x&source=y",
        "unknown=PRIVATE_INPUT",
        "catalog_limit=0",
        "catalog_cursor=",
        "query=PRIVATE_INPUT",
        "source=%FF",
        "title=%C0%AF",
        "catalog_cursor=%ED%A0%80",
        "state=" + "9" * 16_384,
    ],
)
@pytest.mark.parametrize("path", ["/explore/runs?", "/explore/run?run_id=PRIVATE_ID&"])
def test_invalid_parameters_are_sanitized_before_any_store_read(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, path: str, suffix: str
) -> None:
    monkeypatch.setattr(api.container.run_history, "list_runs", no_work)
    monkeypatch.setattr(api.container.saved_run_evidence, "export", no_work)
    response = api.client.get(path + suffix)
    assert response.status_code == 422
    assert_headers(response)
    assert "No invalid filter was ignored or selection widened" in response.text
    assert "PRIVATE_" not in response.text
    assert not ExplorerDOM(response.text).root.by_tag("article")


@pytest.mark.parametrize(
    "suffix", ["", "?run_id=", "?run_id=" + "a" * 257, "?run_id=%FF", "?run_id=a&run_id=b"]
)
def test_invalid_run_identity_is_never_looked_up(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    monkeypatch.setattr(api.container.saved_run_evidence, "export", no_work)
    response = api.client.get("/explore/run" + suffix)
    assert response.status_code == 422
    assert_headers(response)


def test_default_maximum_empty_state_and_exhausted_history(api: ExplorerAPI) -> None:
    assert "No persisted runs" in api.page("/explore/runs").text()
    for number in range(101):
        record(api, f"run-{number}")
    assert len(api.page("/explore/runs").by_tag("article")) == 20
    maximum = api.page("/explore/runs", limit=100, state="")
    assert len(maximum.by_tag("article")) == 100
    assert len(api.page(href(maximum, "next-page")).by_tag("article")) == 1
    assert "No matching runs remain" in api.page("/explore/runs", cursor=1).text()
    assert "No runs match this exact" in api.page("/explore/runs", state="DONE").text()
    for state in AgentState:
        assert api.client.get("/explore/runs", params={"state": state.value}).status_code == 200


@pytest.mark.parametrize(
    "state,expected",
    [
        (AgentState.REASONING, "No terminal DONE event"),
        (AgentState.ERROR, "first terminal event records failure"),
        (AgentState.DONE, "no saved evidence snapshot"),
    ],
)
def test_incomplete_failed_and_legacy_done_are_not_advertised_as_exports(
    api: ExplorerAPI, state: AgentState, expected: str
) -> None:
    record(api, "partial", state)
    history = api.page("/explore/runs", state=state.value, limit=1)
    response = api.client.get(href(history, "inspect-run-1"))
    assert response.status_code == 409
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    assert expected in root.text()
    assert not root.by_tag("article")
    assert not any("download-" in node.attrs.get("id", "") for node in root.by_tag("a"))
    back = api.page(href(root, "back-to-runs"))
    assert identities(back, "run-id") == ["partial"]
    assert form_parameters(back)["state"] == state.value


def test_completed_capture_is_frozen_even_if_latest_history_state_changes(api: ExplorerAPI) -> None:
    bundle = complete(api)
    record(api, bundle.run_id, AgentState.ERROR, agent_id=bundle.agent_id)
    history = api.page("/explore/runs", state="ERROR")
    assert values(history, "recorded-state") == ["ERROR"]
    reader = reader_headers(api.client.get(href(history, "inspect-run-1")))
    assert values(reader, "answer-text") == [bundle.answer.answer]
    assert "may differ from this first-terminal-event snapshot" in reader.text()


def test_restart_and_changed_deleted_corpus_do_not_change_frozen_reader_or_exports(
    api: ExplorerAPI,
) -> None:
    bundle = complete(api)
    history = api.page("/explore/runs", state="DONE", limit=1)
    url = href(history, "inspect-run-1")
    initial = api.client.get(url)
    reader = reader_headers(initial)
    downloads = {
        fmt: api.client.get(href(reader, f"download-{fmt}")).content
        for fmt in ("json", "markdown", "html")
    }
    events = api.container.event_log.list_events()
    for delete in (False, True):
        with closing(sqlite3.connect(api.path)) as connection, connection:
            if delete:
                for table in (
                    "entity_edges",
                    "entity_mentions",
                    "graph_chunks",
                    "chunks",
                    "documents",
                ):
                    connection.execute(f"DELETE FROM {table}")  # noqa: S608
            else:
                connection.execute("UPDATE chunks SET text = 'CHANGED_CURRENT_CORPUS'")
        restarted = create_app(offline_settings(api.path))
        container = restarted.state.container
        before = api.path.read_bytes()
        with TestClient(restarted) as client, browsing_guards(container) as guards:
            container.event_log.list_events = Mock(side_effect=no_work)
            assert client.get(url).content == initial.content
            for fmt, content in downloads.items():
                # Existing downloads have their own event reader; unblock only that reader.
                original = api.container.event_log.list_events
                container.event_log.list_events = original
                assert client.get(href(reader, f"download-{fmt}")).content == content
                container.event_log.list_events = Mock(side_effect=no_work)
            assert not any(guard.call_count for guard in guards.values())
        assert api.path.read_bytes() == before
        assert api.container.event_log.list_events() == events
    assert api.container.saved_run_evidence.export(bundle.run_id) == bundle


def test_invalid_summary_and_invalid_evidence_fail_without_saved_payloads(api: ExplorerAPI) -> None:
    bundle = complete(api)
    record(api, "invalid", query="PRIVATE_PREVIEW")
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(
            "UPDATE agent_events SET timestamp = 'PRIVATE_TIMESTAMP' WHERE run_id = 'invalid'"
        )
    response = api.client.get("/explore/runs", params={"limit": 100})
    assert response.status_code == 409
    assert_headers(response)
    assert "PRIVATE_" not in response.text
    assert identities(api.page("/explore/runs", state="DONE"), "run-id") == [bundle.run_id]
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE agent_events SET payload = '{}' WHERE run_id = 'invalid'")
    for params in ({"limit": 100}, {"limit": 1, "state": "DONE"}):
        response = api.client.get("/explore/runs", params=params)
        assert response.status_code == 409
        assert_headers(response)
        assert "PRIVATE_" not in response.text
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(
            "UPDATE agent_events SET payload = '{PRIVATE_PAYLOAD' "
            "WHERE run_id = ? AND event_type = 'evidence_snapshot'",
            (bundle.run_id,),
        )
    response = api.client.get("/explore/run", params={"run_id": bundle.run_id})
    assert response.status_code == 409
    assert_headers(response)
    assert "Saved evidence is inconsistent" in response.text
    assert "PRIVATE_PAYLOAD" not in response.text


def test_invalid_lookahead_never_returns_partial_history(api: ExplorerAPI) -> None:
    record(api, "older")
    record(api, "newer")
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE agent_events SET timestamp = 'invalid' WHERE run_id = 'older'")
    response = api.client.get("/explore/runs", params={"limit": 1})
    assert response.status_code == 409
    assert not ExplorerDOM(response.text).root.by_tag("article")


def test_saved_json_integer_decode_failure_is_sanitized(api: ExplorerAPI) -> None:
    bundle = complete(api)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(
            "INSERT INTO agent_events (timestamp, agent_id, run_id, event_type, payload) "
            "VALUES ('2026-10-11T00:00:00Z', 'local-agent', ?, 'later', ?)",
            (
                bundle.run_id,
                '{"private":"PRIVATE_DIAGNOSTIC","number":' + "9" * 10_000 + "}",
            ),
        )
    response = api.client.get("/explore/run", params={"run_id": bundle.run_id})
    assert response.status_code == 409
    assert_headers(response)
    assert "PRIVATE_DIAGNOSTIC" not in response.text
    assert "Saved evidence is inconsistent" in response.text


@pytest.mark.parametrize("failure", ["missing", "events", "payload", "total", "bad-utf8"])
def test_bounded_evidence_failures_have_protected_sanitized_browser_errors(
    api: ExplorerAPI, failure: str
) -> None:
    bundle = complete(api)
    expected = 409
    if failure == "missing":
        run_id = "PRIVATE_MISSING_ID"
        expected = 404
    else:
        run_id = bundle.run_id
        with closing(sqlite3.connect(api.path)) as connection, connection:
            if failure in {"events", "total"}:
                count = MAX_RUN_EVENTS if failure == "events" else 9
                padding = 0 if failure == "events" else MAX_EVENT_BYTES - 128
                for _ in range(count):
                    connection.execute(
                        "INSERT INTO agent_events "
                        "(timestamp, agent_id, run_id, event_type, payload) "
                        "VALUES ('2026-10-11T00:00:00Z', 'local-agent', ?, 'later', ?)",
                        (run_id, json.dumps({"padding": "a" * padding})),
                    )
            else:
                payload = json.dumps({"PRIVATE_PAYLOAD": "a" * MAX_EVENT_BYTES})
                if failure == "bad-utf8":
                    connection.execute(
                        "UPDATE agent_events SET payload = CAST(x'FF' AS TEXT) "
                        "WHERE run_id = ? AND event_type = 'evidence_snapshot'",
                        (run_id,),
                    )
                else:
                    connection.execute(
                        "UPDATE agent_events SET payload = ? "
                        "WHERE run_id = ? AND event_type = 'evidence_snapshot'",
                        (payload, run_id),
                    )
    response = api.client.get("/explore/run", params={"run_id": run_id})
    assert response.status_code == expected
    assert_headers(response)
    assert "PRIVATE_" not in response.text
    assert not ExplorerDOM(response.text).root.by_tag("article")


def test_history_and_inline_bounds_count_full_escaped_utf8_including_navigation(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent import evidence_html
    from api import explorer

    bundle = complete(api, "&" * 300)
    for number in range(10):
        record(api, f"partial-{number}", query="\U0001f52c" * 300)
    for path, params, module in (
        ("/explore/runs", {"limit": 20}, explorer),
        ("/explore/run", {"run_id": bundle.run_id}, evidence_html),
    ):
        response = api.client.get(path, params=params)
        assert response.status_code == 200
        with monkeypatch.context() as patch:
            patch.setattr(module, "MAX_HTML_BYTES", len(response.content))
            assert api.client.get(path, params=params).content == response.content
            patch.setattr(module, "MAX_HTML_BYTES", len(response.content) - 1)
            oversized = api.client.get(path, params=params)
            assert oversized.status_code == 413
            assert_headers(oversized)
            assert not ExplorerDOM(oversized.text).root.by_tag("article")


def test_real_html_cap_does_not_truncate_a_bounded_complete_saved_run(api: ExplorerAPI) -> None:
    bundle = complete(api)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        event_id, raw = connection.execute(
            "SELECT id, payload FROM agent_events WHERE run_id = ? "
            "AND json_extract(payload, '$.to_state') = 'ANSWERING'",
            (bundle.run_id,),
        ).fetchone()
        payload = json.loads(raw)
        payload["payload"]["answer"] = "&" * 450_000
        connection.execute(
            "UPDATE agent_events SET payload = ? WHERE id = ?", (json.dumps(payload), event_id)
        )
    response = api.client.get("/explore/run", params={"run_id": bundle.run_id})
    assert response.status_code == 413
    assert_headers(response)
    assert "including navigation, exceeds 4 MiB" in response.text


def test_unavailable_storage_is_not_empty_success(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    for component, path in (
        (api.container.run_history, "/explore/runs"),
        (api.container.saved_run_evidence, "/explore/run?run_id=PRIVATE_ID"),
    ):
        monkeypatch.setattr(
            component, "_database_uri", (api.path.parent / "missing").as_uri() + "?mode=ro"
        )
        response = api.client.get(path)
        assert response.status_code == 503
        assert_headers(response)
        assert "PRIVATE_ID" not in response.text
        assert not (api.path.parent / "missing").exists()


def test_standalone_reader_is_readonly_and_rejects_invalid_ids(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    reader = SQLiteEvidenceReader(missing)
    for run_id in ("", "x" * 257, "\ud800"):
        with pytest.raises(EvidenceExportError) as raised:
            reader.export(run_id)
        assert raised.value.status_code == 422
    with pytest.raises(EvidenceExportError) as raised:
        reader.export("valid-id")
    assert raised.value.code == "evidence_storage_unavailable"
    assert not missing.exists()


@pytest.mark.parametrize(
    "url", ["https://example.invalid/", "//example.invalid/", "/\\example.invalid/", "/\n/evil"]
)
def test_renderer_opt_in_navigation_cannot_activate_external_or_obfuscated_urls(
    api: ExplorerAPI, url: str
) -> None:
    bundle = complete(api)
    with pytest.raises(ValueError, match="application-local"):
        render_html(bundle, navigation=BrowserNavigation(url, "/explore"))


def test_browser_routes_are_get_only_and_in_openapi(api: ExplorerAPI) -> None:
    schema = api.client.get("/openapi.json").json()
    for path in ("/explore/runs", "/explore/run"):
        assert set(schema["paths"][path]) == {"get"}
        operation = schema["paths"][path]["get"]
        assert "text/html" in operation["responses"]["200"]["content"]
        assert {"404", "409", "413", "422", "503"} <= set(operation["responses"])
        unsupported = api.client.post(path)
        assert unsupported.status_code == 405
        assert_headers(unsupported)
        assert unsupported.headers["allow"] == "GET"
