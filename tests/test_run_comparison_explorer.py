"""Offline acceptance checks for the native saved-run comparison workbench."""

import json
import socket
import sqlite3
from contextlib import closing
from unittest.mock import Mock
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from agent.models import AgentAnswer, AgentState, StateTransition
from agent.run_comparison import compare_bundles
from api.application import create_app
from storage.evidence_export import MAX_EVENT_BYTES, MAX_RUN_EVENTS, EvidenceExportError
from tests.test_corpus_explorer import (
    HOSTILE,
    ExplorerDOM,
    assert_headers,
    href,
    identities,
    no_work,
    values,
)
from tests.test_evidence_html import Node
from tests.test_run_comparison import ComparisonAPI, _save, _source
from tests.test_run_comparison import comparison_api as comparison_api

PATH = "/explore/runs/compare"


def selection(baseline: str, candidate: str, **extra: str | int) -> dict[str, str | int]:
    return {"baseline": json.dumps(baseline), "candidate": json.dumps(candidate), **extra}


def page(api: ComparisonAPI, url: str = PATH, **params: str | int) -> Node:
    response = api.client.get(url, params=params) if params else api.client.get(url)
    assert response.status_code == 200, response.text[:1000]
    assert_headers(response)
    return ExplorerDOM(response.text).root


def fields(root: Node) -> dict[str, str]:
    form = root.by_id("comparison-selection")
    result = {node.attrs["name"]: node.attrs["value"] for node in form.by_tag("input")}
    for select in form.by_tag("select"):
        result[select.attrs["name"]] = next(
            option.attrs["value"]
            for option in select.by_tag("option")
            if "selected" in option.attrs
        )
    return result


def test_native_comparison_route_inspects_two_saved_runs(comparison_api: ComparisonAPI) -> None:
    api = comparison_api
    _save(api, "baseline")
    _save(api, "candidate", query="Another synthetic question")
    response = api.client.get(
        "/explore/runs/compare",
        params={"baseline": json.dumps("baseline"), "candidate": json.dumps("candidate")},
    )
    assert response.status_code == 200
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    assert identities(root, "compared-run-id") == ["baseline", "candidate"]
    assert root.by_id("query_changed").text() == "Changed"
    assert href(root, "back-to-runs") == "/explore/runs?limit=20&catalog_limit=20"


def test_discoverable_native_form_pins_across_pages_and_preserves_history(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = comparison_api
    for run_id in ("older", "middle", "newer"):
        _save(api, run_id)
    history_params = {
        "limit": 1,
        "state": "DONE",
        "source": "synthetic:\0\r\nsource",
        "title": "\r\nliteral",
        "catalog_cursor": "paper/\0",
        "catalog_limit": 3,
    }
    history = page(api, "/explore/runs", **history_params)
    history_url = href(history, "next-page")
    history = page(api, history_url)
    reader = Mock(side_effect=no_work)
    with monkeypatch.context() as patch:
        patch.setattr(api.container.saved_run_evidence, "export", reader)
        chooser = page(
            api,
            href(history, "compare-saved-runs"),
            limit=1,
            history=json.dumps(parse_history(history_url)),
        )
        form = chooser.by_id("comparison-selection")
        assert form.attrs["method"] == "get"
        assert {n.attrs["for"] for n in form.by_tag("label")} == {"baseline", "candidate", "limit"}
        assert fields(chooser)["baseline"] == fields(chooser)["candidate"] == ""
        pinned = page(api, href(chooser, "pin-baseline-1"))
        next_page = page(api, href(pinned, "next-choices"))
        last_page = page(api, href(next_page, "next-choices"))
        assert json.loads(fields(last_page)["baseline"]) == "newer"
        assert "cursor" not in fields(last_page)
        assert reader.call_count == 0
    result = page(
        api,
        form.attrs["action"],
        **{
            **fields(last_page),
            "candidate": json.dumps("older"),
        },
    )
    assert identities(result, "compared-run-id") == ["newer", "older"]
    assert parse_qs(urlsplit(href(result, "back-to-runs")).query) == parse_qs(
        urlsplit(history_url).query
    )
    assert identities(page(api, href(result, "back-to-runs")), "run-id") == ["middle"]
    changed = page(api, href(result, "change-selection"))
    assert fields(changed)["baseline"] == json.dumps("newer")
    assert fields(changed)["candidate"] == ""
    entry = page(api, href(history, "compare-run-1"))
    assert json.loads(fields(entry)["baseline"]) == "middle"


def parse_history(url: str) -> dict[str, str]:
    return {key: items[0] for key, items in parse_qs(urlsplit(url).query).items()}


def test_same_and_distinct_no_change_controls_read_same_id_once(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = comparison_api
    _save(api, "first")
    _save(api, "second")
    export = Mock(wraps=api.container.saved_run_evidence.export)
    monkeypatch.setattr(api.container.saved_run_evidence, "export", export)
    result = page(api, **selection("first", "second"))
    assert "No changes in compared saved fields" in result.by_id("comparison-outcome").text()
    assert "does not mean full trace equality" in result.text()
    export.reset_mock()
    same = page(api, href(result, "same-run-control"))
    assert identities(same, "compared-run-id") == ["first", "first"]
    export.assert_called_once_with("first")
    assert "No changes" in same.by_id("comparison-outcome").text()


def test_directional_evidence_matches_pure_comparison_and_exact_frozen_links(
    comparison_api: ComparisonAPI,
) -> None:
    api = comparison_api
    _save(api, "left", [_source("shared"), _source("removed")])
    _save(
        api,
        "right",
        [_source("added"), _source("shared", score=0.9, text="Revised passage")],
        query="Candidate question",
        provider="historical-fixture",
        model_name="recorded-model",
    )
    expected = compare_bundles(
        *(api.container.saved_run_evidence.export(x) for x in ("left", "right"))
    )
    result = page(api, **selection("left", "right"))
    for name, changed in expected.changes.model_dump().items():
        assert result.by_id(name).text() == ("Changed" if changed else "Unchanged")
    assert json.loads(values(result, "evidence-statistics")[0]) == (
        expected.evidence.statistics.model_dump()
    )
    assert values(result, "rank-delta") == ["Candidate rank delta: +1"]
    assert "recorded-model" in result.text()
    swapped = page(api, href(result, "swap-comparison"))
    assert identities(swapped, "compared-run-id") == ["right", "left"]
    assert values(swapped, "rank-delta") == ["Candidate rank delta: -1"]
    assert identities(swapped, "candidate-chunk-id") == ["shared", "removed"]
    for root in (result, swapped):
        for anchor in root.by_tag("a"):
            if not anchor.attrs.get("id", "").endswith("-source"):
                continue
            url = anchor.attrs["href"]
            parts = urlsplit(url)
            assert parts.path == "/explore/run"
            run_id = parse_qs(parts.query)["run_id"][0]
            source = api.container.saved_run_evidence.export(run_id).snapshot.sources[
                int(parts.fragment.removeprefix("source-")) - 1
            ]
            response = api.client.get(url)
            assert response.status_code == 200
            source_node = ExplorerDOM(response.text).root.by_id(parts.fragment)
            assert values(source_node, "passage") == [source.chunk.text]
            assert href(ExplorerDOM(response.text).root, "back-to-runs") == href(
                root, "back-to-runs"
            )
    original = api.client.get("/runs/left/compare/right")
    assert original.status_code == 200 and original.json() == expected.model_dump(mode="json")


@pytest.mark.parametrize(
    "run_id",
    [
        ".",
        "..",
        "x/../\u7814?draft#v1",
        "//example.invalid/",
        " quotes'\"?%#&+\\ ",
        "line\r\nid\t",
        "nul\0id",
        "cafe\u0301/\u202eRTL\u202c",
        "\U0001f52c" * 256,
    ],
)
def test_json_native_selection_preserves_opaque_exact_ids(
    comparison_api: ComparisonAPI,
    run_id: str,
) -> None:
    api = comparison_api
    _save(
        api,
        run_id,
        [
            _source(
                "chunk/\0\r\n", "doc/\t\u7814", title="Synthetic title", text="Synthetic passage"
            )
        ],
    )
    history = page(api, "/explore/runs")
    chooser = page(api, href(history, "compare-run-1"))
    form = chooser.by_id("comparison-selection")
    encoded = json.dumps(run_id, ensure_ascii=True)
    assert fields(chooser)["baseline"] == encoded
    assert all("\0" not in option.attrs["value"] for option in form.by_tag("option"))
    result = page(api, form.attrs["action"], **{**fields(chooser), "candidate": encoded})
    assert identities(result, "compared-run-id") == [run_id, run_id]
    assert identities(result, "baseline-chunk-id") == ["chunk/\0\r\n"]
    url = href(result, "shared-1-baseline-source")
    assert parse_qs(urlsplit(url).query)["run_id"] == [run_id]
    assert urlsplit(url).path == "/explore/run" and urlsplit(url).fragment == "source-1"
    assert identities(page(api, href(result, "swap-comparison")), "compared-run-id") == [run_id] * 2


@pytest.mark.parametrize("which", ["baseline", "candidate"])
@pytest.mark.parametrize(
    "raw",
    [
        '""',
        '"x' + "a" * 256 + '"',
        "null",
        "123",
        "true",
        "[]",
        "{}",
        '"\\ud800"',
        '"unfinished',
        '"x" trailing',
        "[" * 1000 + "]" * 1000,
    ],
)
def test_invalid_json_identity_fails_before_reads(
    comparison_api: ComparisonAPI,
    monkeypatch: pytest.MonkeyPatch,
    which: str,
    raw: str,
) -> None:
    api = comparison_api
    monkeypatch.setattr(api.container.saved_run_evidence, "export", no_work)
    monkeypatch.setattr(api.container.run_history, "list_runs", no_work)
    response = api.client.get(PATH, params={**selection("PRIVATE_ID", "PRIVATE_ID"), which: raw})
    assert response.status_code == 422
    assert_headers(response)
    assert "PRIVATE_ID" not in response.text


@pytest.mark.parametrize(
    "suffix",
    [
        "baseline=%22a%22&baseline=%22b%22",
        "candidate=%FF",
        "candidate=%C0%AF",
        "candidate=%ED%A0%80",
        "candidate=%GG",
        "candidate=%",
        "limit=0",
        "limit=101",
        "limit=",
        "limit=1.0",
        "cursor=",
        "cursor=0",
        "cursor=-1",
        "cursor=9223372036854775808",
        "unknown=PRIVATE",
        "history=null",
        "history=%5B%5D",
        "history=%7Bbad",
        "history=" + json.dumps({"url": "https://example.invalid/"}),
        "history=" + json.dumps({"limit": True}),
        'history={"limit":1,"limit":2}',
        'history={"source":"\\ud800"}',
        'history={"state":"done"}',
        'history={"catalog_cursor":""}',
        "history=" + "[" * 1500 + "]" * 1500,
    ],
)
def test_request_and_return_state_errors_never_read_or_echo_inputs(
    comparison_api: ComparisonAPI,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    api = comparison_api
    monkeypatch.setattr(api.container.saved_run_evidence, "export", no_work)
    monkeypatch.setattr(api.container.run_history, "list_runs", no_work)
    response = api.client.get(PATH + "?" + suffix)
    assert response.status_code == 422
    assert_headers(response)
    assert "PRIVATE" not in response.text and "example.invalid" not in response.text


@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize(
    "failure,status",
    [
        ("missing", 404),
        ("incomplete", 409),
        ("failed", 409),
        ("legacy", 409),
        ("malformed", 409),
        ("invalid", 409),
        ("bad-utf8", 409),
        ("events", 409),
        ("payload", 409),
        ("total", 409),
        ("unrepresentable", 409),
    ],
)
def test_either_side_failure_is_sanitized_and_never_a_partial_comparison(
    comparison_api: ComparisonAPI,
    side: str,
    failure: str,
    status: int,
) -> None:
    api = comparison_api
    for run_id in ("baseline", "candidate"):
        if run_id == side and failure in {"missing", "incomplete", "failed", "legacy"}:
            if failure == "missing":
                continue
            state = {
                "incomplete": AgentState.REASONING,
                "failed": AgentState.ERROR,
                "legacy": AgentState.DONE,
            }[failure]
            api.container.event_log.append_transition(
                StateTransition(
                    agent_id="fixture",
                    run_id=run_id,
                    from_state=AgentState.IDLE,
                    to_state=state,
                    payload={"private": "PRIVATE"},
                )
            )
        else:
            _save(
                api,
                run_id,
                query="PRIVATE_QUERY",
                answer=AgentAnswer(answer="PRIVATE_ANSWER", claims=[], citations=[]),
            )
    with closing(sqlite3.connect(api.path)) as connection, connection:
        if failure in {"malformed", "invalid", "payload", "bad-utf8"}:
            if failure == "bad-utf8":
                connection.execute(
                    "UPDATE agent_events SET payload=CAST(x'FF' AS TEXT) "
                    "WHERE run_id=? AND event_type='evidence_snapshot'",
                    (side,),
                )
            else:
                payload = {
                    "malformed": "{PRIVATE",
                    "invalid": "{}",
                    "payload": json.dumps({"PRIVATE": "x" * MAX_EVENT_BYTES}),
                }[failure]
                connection.execute(
                    "UPDATE agent_events SET payload=? "
                    "WHERE run_id=? AND event_type='evidence_snapshot'",
                    (payload, side),
                )
        elif failure in {"events", "total"}:
            for _ in range(MAX_RUN_EVENTS if failure == "events" else 9):
                connection.execute(
                    "INSERT INTO agent_events (timestamp,agent_id,run_id,event_type,payload) "
                    "VALUES ('2026-10-22T00:00:00Z','fixture',?,'later',?)",
                    (
                        side,
                        json.dumps({"padding": "x" * (MAX_EVENT_BYTES - 128)})
                        if failure == "total"
                        else "{}",
                    ),
                )
        elif failure == "unrepresentable":
            connection.execute(
                "UPDATE agent_events SET payload=json_set(payload,'$.payload.answer',?) "
                "WHERE run_id=? AND json_extract(payload,'$.to_state')='ANSWERING'",
                ("\0PRIVATE", side),
            )
    response = api.client.get(PATH, params=selection("baseline", "candidate"))
    assert response.status_code == status
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    assert f"{side.capitalize()} evidence unavailable" in root.text()
    assert not root.by_class("compared-run-id") and not root.by_tag("table")
    assert "PRIVATE" not in root.text()


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_unavailable_storage_side_and_baseline_first_precedence(
    comparison_api: ComparisonAPI,
    monkeypatch: pytest.MonkeyPatch,
    side: str,
) -> None:
    api = comparison_api
    _save(api, "baseline")
    export = api.container.saved_run_evidence.export

    def unavailable(run_id: str) -> object:
        if run_id == side:
            raise EvidenceExportError("evidence_storage_unavailable", "PRIVATE_PATH", 503)
        return export(run_id)

    monkeypatch.setattr(api.container.saved_run_evidence, "export", unavailable)
    response = api.client.get(PATH, params=selection("baseline", "candidate"))
    assert response.status_code == 503
    assert_headers(response)
    assert f"{side.capitalize()} evidence unavailable" in response.text
    assert "PRIVATE_PATH" not in response.text
    monkeypatch.setattr(api.container.saved_run_evidence, "export", export)
    missing = api.client.get(PATH, params=selection("missing", "also-missing"))
    assert missing.status_code == 404 and "Baseline evidence unavailable" in missing.text


def test_literal_output_previews_and_safe_query_addresses(comparison_api: ComparisonAPI) -> None:
    api = comparison_api
    query = "\n\r\t" + HOSTILE + "long " * 100
    _save(
        api,
        "base",
        [_source("shared", title=HOSTILE, source="javascript:alert(1)")],
        query=query,
        answer=AgentAnswer(answer=query, claims=[], citations=[]),
        provider=HOSTILE,
        model_name=HOSTILE,
    )
    root = page(api, **selection("base", "base"))
    assert values(root, "query-preview") == [query[:240]] * 2
    assert values(root, "answer-preview") == [query[:240]] * 2
    assert "Preview truncated at 240 characters" in root.text()
    assert not any(root.by_tag(tag) for tag in ("script", "img", "base", "iframe", "object", "svg"))
    for node in root.walk():
        assert not any(name.startswith("on") or name == "style" for name in node.attrs)
    for anchor in root.by_tag("a"):
        parts = urlsplit(anchor.attrs["href"])
        assert not parts.scheme and not parts.netloc
        assert not parts.path or parts.path == "/explore" or parts.path.startswith("/explore/")


def test_empty_evidence_and_reassigned_identity_are_not_support_scores(
    comparison_api: ComparisonAPI,
) -> None:
    api = comparison_api
    _save(api, "empty", sources=[])
    empty = page(api, **selection("empty", "empty"))
    assert json.loads(values(empty, "evidence-statistics")[0])["identity_jaccard"] is None
    _save(api, "left", [_source("shared", "old-owner")])
    _save(api, "right", [_source("shared", "new-owner")])
    root = page(api, **selection("left", "right"))
    assert json.loads(values(root, "reassigned-chunk-ids")[0]) == ["shared"]
    assert json.loads(values(root, "evidence-statistics")[0])["shared_count"] == 0


def test_exact_request_and_escaped_output_byte_caps(
    comparison_api: ComparisonAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api import explorer

    api = comparison_api
    run_id = "\U0001f52c" * 256
    _save(api, run_id, query="&" * 300)
    query = urlencode(selection(run_id, run_id, history="{}"))
    query += "+" * (16_384 - len(query))
    assert len(query) == 16_384
    response = api.client.get(PATH + "?" + query)
    assert response.status_code == 200
    assert api.client.get(PATH + "?" + query + "+").status_code == 422
    with monkeypatch.context() as patch:
        patch.setattr(explorer, "MAX_HTML_BYTES", len(response.content))
        assert api.client.get(PATH + "?" + query).content == response.content
        patch.setattr(explorer, "MAX_HTML_BYTES", len(response.content) - 1)
        rejected = api.client.get(PATH + "?" + query)
        assert rejected.status_code == 413
        assert_headers(rejected)
        assert not ExplorerDOM(rejected.text).root.by_tag("table")


def test_real_output_cap_rejects_escaped_large_record_without_truncation(
    comparison_api: ComparisonAPI,
) -> None:
    api = comparison_api
    _save(api, "oversized", model_name="&" * 150_000)
    response = api.client.get(PATH, params=selection("oversized", "oversized"))
    assert response.status_code == 413
    assert_headers(response)
    assert "Nothing was truncated" in response.text
    assert not ExplorerDOM(response.text).root.by_tag("table")


def test_restart_deleted_corpus_and_no_side_effects(
    comparison_api: ComparisonAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = comparison_api
    _save(api, "baseline")
    _save(api, "candidate")
    events = api.container.event_log.list_events()
    before = api.path.read_bytes()
    for target, method in ((socket, "create_connection"), (socket.socket, "connect")):
        monkeypatch.setattr(target, method, no_work)
    with browsing_guards(api.container) as guards:
        monkeypatch.setattr(api.container.evidence_exporter, "export", no_work)
        monkeypatch.setattr(api.container.event_log, "list_events", no_work)
        initial = api.client.get(PATH, params=selection("baseline", "candidate"))
        assert initial.status_code == 200
        page(api)
        assert not any(guard.call_count for guard in guards.values())
    assert api.path.read_bytes() == before
    with closing(sqlite3.connect(api.path)) as connection, connection:
        for table in ("entity_edges", "entity_mentions", "graph_chunks", "chunks", "documents"):
            connection.execute(f"DELETE FROM {table}")  # noqa: S608
    restarted = create_app(offline_settings(api.path))
    before = api.path.read_bytes()
    with TestClient(restarted) as client, browsing_guards(restarted.state.container) as guards:
        monkeypatch.setattr(restarted.state.container.evidence_exporter, "export", no_work)
        assert (
            client.get(PATH, params=selection("baseline", "candidate")).content == initial.content
        )
        assert not any(guard.call_count for guard in guards.values())
    assert api.path.read_bytes() == before
    assert restarted.state.container.event_log.list_events() == events


@pytest.mark.parametrize("mounted", [False, True])
def test_mount_prefixes_history_reader_navigation_and_get_only(
    comparison_api: ComparisonAPI,
    mounted: bool,
) -> None:
    api = comparison_api
    _save(api, "left/opaque")
    _save(api, "right")
    application = api.application
    prefix, root_path = "/library", "/library"
    if mounted:
        application = FastAPI()
        application.mount("/library", api.application)
        prefix, root_path = "/prefix/library", "/prefix"
    with TestClient(application, root_path=root_path) as client:
        response = client.get(
            prefix + PATH,
            params=selection(
                "left/opaque",
                "right",
                history=json.dumps({"state": "DONE", "limit": 1}),
            ),
        )
        assert response.status_code == 200
        root = ExplorerDOM(response.text).root
        for form in root.by_tag("form"):
            assert form.attrs["action"].startswith(prefix + "/")
        for anchor in root.by_tag("a"):
            parts = urlsplit(anchor.attrs["href"])
            assert not parts.path or parts.path.startswith(prefix + "/")
        reader = client.get(href(root, "shared-1-baseline-source"))
        assert reader.status_code == 200
        assert href(ExplorerDOM(reader.text).root, "back-to-runs") == href(root, "back-to-runs")
        assert client.get(href(root, "swap-comparison")).status_code == 200
        for method in ("post", "put", "delete", "head"):
            rejected = getattr(client, method)(prefix + PATH)
            assert rejected.status_code == 405 and rejected.headers["allow"] == "GET"
            assert rejected.headers["cache-control"] == "no-store"
        invalid = client.get(prefix + PATH, params={"baseline": "not-json"})
        assert invalid.status_code == 422
        assert prefix + "/explore/runs" in invalid.text
        schema = client.get(prefix + "/openapi.json").json()["paths"][PATH]
        assert set(schema) == {"get"}
        assert {"404", "409", "413", "422", "503"} <= set(schema["get"]["responses"])
