"""Full original-question literals, never preview-only, answer-body or semantic search."""

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.models import AgentState, StateTransition
from storage.run_history import MAX_QUESTION_CHARACTERS, RunHistoryError, SQLiteRunHistory
from tests.test_run_history import HistoryAPI, _no_work, _page, _transition
from tests.test_run_history import history_api as history_api


@pytest.mark.parametrize(
    "needle,decoy",
    [
        ("100%", "100 percent"),
        ("under_score", "underXscore"),
        ("[a-z]*?.+^$", "abcdef"),
        (r"\d+\\", "42"),
        ("' OR 1=1 --", "' OR 2=2 --"),
        ('"<script>&', "<script>"),
        ("\u7814\u7a76\U0001f52c", "research"),
        ("caf\u00e9", "cafe\u0301"),
        ("cafe\u0301", "caf\u00e9"),
        ("GraphRAG", "graphrag"),
        ("Stra\u00dfe", "STRASSE"),
        ("  exact ", " exact"),
        ("\0after", "after"),
        ("\r\nTab\t", "Tab "),
    ],
)
def test_literal_full_question_beyond_preview(
    history_api: HistoryAPI, needle: str, decoy: str
) -> None:
    original = "Synthetic prefix. " * 40 + needle + " end"
    _transition(history_api, "match", query=original)
    _transition(history_api, "newer-decoy", query=decoy)
    page = _page(history_api, question=needle, limit=1)
    assert [run["run_id"] for run in page["runs"]] == ["match"]
    assert page["next_cursor"] is None
    assert page["runs"][0]["query_summary"] == original[:300]
    assert needle not in page["runs"][0]["query_summary"]
    assert page["runs"][0]["query_truncated"] is True
    assert (
        SQLiteRunHistory(history_api.database_path)
        .list_runs(question=needle, limit=1)
        .model_dump(mode="json")
        == page
    )


def test_only_first_planning_question_is_searchable_and_legacy_rows_are_not_invented(
    history_api: HistoryAPI,
) -> None:
    api = history_api
    needle = "original-marker"
    _transition(api, "original", query=needle)
    _transition(api, "original", query="Later planning query")
    _transition(api, "later-only", query=None)
    _transition(api, "later-only", query=needle)
    _transition(api, "empty", query="")
    _transition(api, "null", query=None)
    api.container.event_log.append_transition(
        StateTransition(
            agent_id="synthetic-agent",
            run_id="missing",
            from_state=AgentState.IDLE,
            to_state=AgentState.PLANNING,
            payload={},
        )
    )
    _transition(api, "legacy-done", AgentState.DONE)
    api.container.event_log.append_event(
        "synthetic-agent", "event-only", "audit", {"query": needle}
    )
    _transition(api, "answer-only", query="Different question")
    api.container.event_log.append_event(
        "synthetic-agent", "answer-only", "generation_record", {"answer": needle, "query": needle}
    )
    page = _page(api, question=needle)
    assert [run["run_id"] for run in page["runs"]] == ["original"]
    assert page["runs"][0]["query_summary"] == needle
    assert _page(api, question="Later planning query") == {"runs": [], "next_cursor": None}
    unfiltered = _page(api)
    assert len(unfiltered["runs"]) == 8
    assert (
        SQLiteRunHistory(api.database_path).list_runs(question=None).model_dump(mode="json")
        == unfiltered
    )


def test_state_and_question_filter_before_keyset_paging_without_duplicates(
    history_api: HistoryAPI,
) -> None:
    api = history_api
    keys = {
        f"run-{number}": _transition(
            api, f"run-{number}", query="needle" if number % 2 == 0 else "different"
        )
        for number in range(8)
    }
    for number in (0, 1, 2, 3, 5, 6, 7):
        _transition(api, f"run-{number}", AgentState.ERROR)
    first = _page(api, question="needle", state="ERROR", limit=1)
    assert [run["run_id"] for run in first["runs"]] == ["run-6"]
    assert first["next_cursor"] == keys["run-6"]
    _transition(api, "new", query="needle")
    _transition(api, "new", AgentState.ERROR)
    _transition(api, "run-2", query="Do not replace the first question")
    _transition(api, "run-2", AgentState.ERROR)
    _transition(api, "run-4", AgentState.ERROR)
    api.container.event_log.append_event("synthetic-agent", "run-0", "audit", {})
    seen = first["runs"]
    cursor = first["next_cursor"]
    while cursor is not None:
        page = _page(api, question="needle", state="ERROR", limit=1, cursor=cursor)
        seen.extend(page["runs"])
        cursor = page["next_cursor"]
    assert [run["run_id"] for run in seen] == ["run-6", "run-4", "run-2", "run-0"]
    assert len({run["run_id"] for run in seen}) == 4
    assert all(run["first_event_id"] == keys[run["run_id"]] for run in seen)
    assert _page(api, question="needle", state="ERROR", limit=1)["runs"][0]["run_id"] == "new"
    assert _page(api, question="needle", cursor=1) == {"runs": [], "next_cursor": None}
    assert _page(api, question="needle", state="DONE") == {"runs": [], "next_cursor": None}


@pytest.mark.parametrize("question", ["x", "\U0001f52c" * MAX_QUESTION_CHARACTERS])
def test_filter_accepts_exact_code_point_bounds(history_api: HistoryAPI, question: str) -> None:
    _transition(history_api, "exact", query=question)
    assert _page(history_api, question=question)["runs"][0]["run_id"] == "exact"


@pytest.mark.parametrize(
    "question",
    ["", " ", "\r\n\t", "x" * 201, "\U0001f52c" * 201, "\ud800", 1, True, 1.5, b"x", [], {}],
)
def test_invalid_python_search_fails_before_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, question: object
) -> None:
    path = tmp_path / "must-not-create.sqlite3"
    history = SQLiteRunHistory(path)
    monkeypatch.setattr(sqlite3, "connect", _no_work)
    with pytest.raises(ValidationError):
        history.list_runs(question=question)
    assert not path.exists()


@pytest.mark.parametrize(
    "suffix",
    [
        "question=",
        "question",
        "question=%20%20",
        "question=%09%0D%0A",
        "question=" + "x" * 201,
        "question=" + "%F0%9F%94%AC" * 201,
        "question=%FF",
        "question=%ED%A0%80",
        "question=%",
        "question=%GG",
        "question=PRIVATE_FIRST&question=PRIVATE_SECOND",
        "question=x&limit=1&limit=2",
        "question=x&unknown=PRIVATE_INPUT",
        "question=x&state=done",
        "question=x&cursor=0",
        "question=x&limit=101",
        "question=x&limit=" + "0" * 16_384,
    ],
)
def test_invalid_api_search_is_sanitized_before_read(
    history_api: HistoryAPI, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    monkeypatch.setattr(history_api.container.run_history, "list_runs", _no_work)
    response = history_api.client.get("/runs?" + suffix)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_run_search_request"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "PRIVATE_" not in response.text


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b'{"from_state":"IDLE","to_state":"PLANNING","payload":{"query":42}}',
        b'{"from_state":"IDLE","to_state":"UNKNOWN","payload":{}}',
        b'{"from_state":"IDLE","to_state":"PLANNING","payload":{"query":"'
        + b"x" * 500
        + b'\xff"}}',
        b'{"from_state":"IDLE","to_state":"PLANNING","payload":{"query":"'
        + b"x" * 500
        + b'\\ud800"}}',
    ],
)
@pytest.mark.parametrize("state", [None, "DONE"])
def test_bad_transition_or_full_question_unicode_cannot_hide_as_a_nonmatch(
    history_api: HistoryAPI, payload: bytes, state: str | None
) -> None:
    _transition(history_api, "bad")
    with sqlite3.connect(history_api.database_path) as connection:
        connection.execute("UPDATE agent_events SET payload = CAST(? AS TEXT)", (payload,))
    params = {} if state is None else {"state": state}
    response = history_api.client.get("/runs", params={"question": "not-present", **params})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "invalid_run_record"
    with pytest.raises(RunHistoryError):
        SQLiteRunHistory(history_api.database_path).list_runs(question="not-present")


def test_matching_lookahead_still_validates_metadata(history_api: HistoryAPI) -> None:
    _transition(history_api, "bad-older", query="needle")
    _transition(history_api, "good-newer", query="needle")
    with sqlite3.connect(history_api.database_path) as connection:
        connection.execute(
            "UPDATE agent_events SET timestamp = 'PRIVATE_TIME' WHERE run_id = 'bad-older'"
        )
    response = history_api.client.get("/runs", params={"question": "needle", "limit": 1})
    assert response.status_code == 409 and "PRIVATE_TIME" not in response.text


@pytest.mark.parametrize("question", [" ", 123, True, ["x"], {"value": "x"}, "\ud800", "x" * 201])
@pytest.mark.parametrize(
    "route,field",
    [
        ("/explore/runs/compare", "history"),
        ("/explore/run/review", "selection"),
        ("/explore/run/annotate", "selection"),
    ],
)
def test_invalid_structured_return_question_never_reaches_a_reader(
    history_api: HistoryAPI,
    monkeypatch: pytest.MonkeyPatch,
    question: object,
    route: str,
    field: str,
) -> None:
    monkeypatch.setattr(history_api.container.run_history, "list_runs", _no_work)
    monkeypatch.setattr(history_api.container.saved_run_evidence, "export", _no_work)
    selection: dict[str, object] = {"question": question}
    if field == "selection":
        selection["run_id"] = "PRIVATE_ID"
    if route.endswith("/annotate"):
        selection.update(document_id="paper", chunk_id="chunk", source_text_sha256="0" * 64)
    response = history_api.client.get(route, params={field: json.dumps(selection)})
    assert response.status_code == 422
    assert "PRIVATE_ID" not in response.text


def test_search_does_not_create_storage_or_mask_missing_tables(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        SQLiteRunHistory(path).list_runs(question="needle")
    assert not path.exists()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
    before = path.read_bytes()
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        SQLiteRunHistory(path).list_runs(question="needle")
    assert path.read_bytes() == before


def test_search_parameter_is_documented_without_changing_response_shape(
    history_api: HistoryAPI,
) -> None:
    parameters = history_api.client.get("/openapi.json").json()["paths"]["/runs"]["get"][
        "parameters"
    ]
    question = next(parameter for parameter in parameters if parameter["name"] == "question")
    assert question["required"] is False
    schema = next(item for item in question["schema"]["anyOf"] if item["type"] == "string")
    assert schema["minLength"] == 1 and schema["maxLength"] == 200
    _transition(history_api, "same", query="null 123")
    original = _page(history_api)
    assert _page(history_api, question="null") == original
    assert _page(history_api, question="123") == original
    assert set(json.loads(json.dumps(original))) == {"runs", "next_cursor"}
