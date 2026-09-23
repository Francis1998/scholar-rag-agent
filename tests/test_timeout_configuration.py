"""Regression-first contracts for finite execution timeouts and saved configuration."""

import asyncio
import json
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from agent.evidence import EvidenceBundle, RunConfiguration
from agent.models import AgentState
from agent.retrieval_preview import RetrievalPreviewError
from agent.safety import SafetyLimits
from api.application import create_app
from api.dependencies import AppContainer
from config import Settings
from storage.event_log import SQLiteEventLog
from storage.evidence_export import EvidenceExporter

TIMEOUT_FIELDS = ("retrieval_timeout_seconds", "reasoning_timeout_seconds")
INVALID_TIMEOUTS = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param(0.0, id="zero"),
    pytest.param(-0.1, id="negative"),
    pytest.param(False, id="false"),
    pytest.param(None, id="none"),
    pytest.param("not-a-timeout", id="nonnumeric"),
]


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for field in TIMEOUT_FIELDS:
        monkeypatch.delenv(f"SCHOLAR_RAG_{field.upper()}", raising=False)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(key, "")
    send = AsyncMock(side_effect=AssertionError("Timeout tests must not make live HTTP calls."))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield
    send.assert_not_called()


@pytest.fixture(params=TIMEOUT_FIELDS)
def timeout_field(request: pytest.FixtureRequest) -> str:
    assert isinstance(request.param, str)
    return request.param


def configuration_values(**overrides: object) -> dict[str, object]:
    return {
        "max_source_docs": 50,
        "max_hops": 5,
        "retrieval_timeout_seconds": 30.0,
        "reasoning_timeout_seconds": 60.0,
        **overrides,
    }


@pytest.mark.parametrize("value", INVALID_TIMEOUTS)
def test_settings_reject_invalid_timeouts(timeout_field: str, value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, **{timeout_field: value})
    assert caught.value.errors()[0]["loc"] == (timeout_field,)


@pytest.mark.parametrize("value", ["NaN", "Inf", "Infinity", "-Inf", "-Infinity", "1e999"])
def test_settings_reject_nonfinite_environment(
    timeout_field: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(f"SCHOLAR_RAG_{timeout_field.upper()}", value)
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None)
    assert caught.value.errors()[0]["loc"] == (timeout_field,)


@pytest.mark.parametrize("source", ["environment", "dotenv"])
@pytest.mark.parametrize("value", ["Inf", "Infinity", "1e999"])
def test_invalid_timeouts_fail_at_startup_before_storage(
    timeout_field: str,
    value: str,
    source: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "startup.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(database_path))
    variable = f"SCHOLAR_RAG_{timeout_field.upper()}"
    if source == "environment":
        monkeypatch.setenv(variable, value)
    else:
        (tmp_path / ".env").write_text(f"{variable}={value}\n", encoding="utf-8")

    with pytest.raises(ValidationError) as caught:
        create_app()
    assert caught.value.errors()[0]["loc"] == (timeout_field,)
    assert not database_path.exists()


@pytest.mark.parametrize("value", INVALID_TIMEOUTS)
def test_safety_limits_reject_invalid_timeouts(timeout_field: str, value: object) -> None:
    with pytest.raises(ValueError, match=timeout_field):
        SafetyLimits(**{timeout_field: value})


@pytest.mark.parametrize("value", INVALID_TIMEOUTS)
def test_copied_safety_limits_revalidate_mutated_timeouts(
    timeout_field: str, value: object
) -> None:
    limits = SafetyLimits()
    setattr(limits, timeout_field, value)
    with pytest.raises(ValueError, match=timeout_field):
        replace(limits)


@pytest.mark.parametrize("value", INVALID_TIMEOUTS)
@pytest.mark.parametrize("encoding", ["python", "json"])
def test_run_configuration_rejects_invalid_timeouts(
    timeout_field: str, value: object, encoding: str
) -> None:
    values = configuration_values(**{timeout_field: value})
    with pytest.raises(ValidationError) as caught:
        if encoding == "python":
            RunConfiguration.model_validate(values)
        else:
            RunConfiguration.model_validate_json(json.dumps(values))
    assert caught.value.errors()[0]["loc"] == (timeout_field,)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, 1.0),
        (2, 2.0),
        ("1.25", 1.25),
        ("1e200", 1e200),
        (sys.float_info.max, sys.float_info.max),
    ],
)
def test_finite_timeouts_preserve_existing_float_coercion_and_round_trip(
    timeout_field: str, value: object, expected: float
) -> None:
    settings = Settings(_env_file=None, **{timeout_field: value})
    limits = SafetyLimits(**{timeout_field: value})
    configuration = RunConfiguration.model_validate(
        configuration_values(**{timeout_field: getattr(limits, timeout_field)})
    )
    for validated in (settings, limits, configuration):
        assert getattr(validated, timeout_field) == expected
        assert type(getattr(validated, timeout_field)) is float
    encoded = json.dumps(configuration.model_dump(mode="json"), allow_nan=False)
    assert RunConfiguration.model_validate_json(encoded) == configuration
    assert RunConfiguration.model_validate_json(configuration.model_dump_json()) == configuration


@pytest.mark.parametrize("value", [1.0, 1.25, 1e200, sys.float_info.max])
def test_settings_accept_finite_environment_values(
    timeout_field: str, value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(f"SCHOLAR_RAG_{timeout_field.upper()}", str(value))
    assert getattr(Settings(_env_file=None), timeout_field) == value


@pytest.mark.parametrize("value", [0.1, 0.999])
def test_settings_keep_one_second_minimum(timeout_field: str, value: float) -> None:
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, **{timeout_field: value})
    assert caught.value.errors()[0]["loc"] == (timeout_field,)


@pytest.mark.parametrize("value", [0.1, 1e-9, float.fromhex("0x0.0000000000001p-1022")])
def test_library_timeouts_preserve_positive_fractions(timeout_field: str, value: float) -> None:
    limits = SafetyLimits(**{timeout_field: value})
    copied = replace(limits)
    configuration = RunConfiguration.model_validate(
        configuration_values(**{timeout_field: getattr(copied, timeout_field)})
    )
    assert getattr(copied, timeout_field) == value
    assert getattr(configuration, timeout_field) == value


def test_timeout_defaults_are_unchanged() -> None:
    for limits in (Settings(_env_file=None), SafetyLimits()):
        assert limits.retrieval_timeout_seconds == 30.0
        assert limits.reasoning_timeout_seconds == 60.0


@pytest.fixture
def application(tmp_path: Path) -> FastAPI:
    return create_app(offline_settings(tmp_path / "timeouts.sqlite3"))


@pytest.fixture
def work_spies(
    application: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> tuple[Mock, AsyncMock, AsyncMock]:
    container: AppContainer = application.state.container
    observe = Mock(wraps=container.runner._analyzer.analyze)
    retrieve = AsyncMock(wraps=container.runner._executor.retrieve)
    generate = AsyncMock(wraps=container.llm.generate)
    monkeypatch.setattr(container.runner._analyzer, "analyze", observe)
    monkeypatch.setattr(container.runner._executor, "retrieve", retrieve)
    monkeypatch.setattr(container.llm, "generate", generate)
    return observe, retrieve, generate


@pytest.mark.parametrize("value", INVALID_TIMEOUTS)
async def test_mutated_timeouts_return_run_error_before_work(
    application: FastAPI,
    work_spies: tuple[Mock, AsyncMock, AsyncMock],
    timeout_field: str,
    value: object,
) -> None:
    container: AppContainer = application.state.container
    setattr(container.runner._safety_limits, timeout_field, value)
    result = await asyncio.wait_for(container.runner.run("GraphRAG evidence"), timeout=2)
    assert result.state == AgentState.ERROR
    assert result.error is not None and timeout_field in result.error
    assert result.observation is None and result.plan is None and result.answer is None
    for spy in work_spies:
        spy.assert_not_called()
    events = container.event_log.list_events(result.run_id)
    assert len(events) == 1
    assert events[0]["payload"]["from_state"] == "IDLE"
    assert events[0]["payload"]["to_state"] == "ERROR"
    assert events[0]["payload"]["payload"] == {"error": result.error}


@pytest.mark.parametrize("value", INVALID_TIMEOUTS)
async def test_mutated_timeouts_preserve_python_preview_error_contract(
    application: FastAPI,
    work_spies: tuple[Mock, AsyncMock, AsyncMock],
    timeout_field: str,
    value: object,
) -> None:
    container: AppContainer = application.state.container
    setattr(container.runner._safety_limits, timeout_field, value)
    with pytest.raises(RetrievalPreviewError) as caught:
        await asyncio.wait_for(container.runner.preview("GraphRAG evidence"), timeout=2)
    assert caught.value.code == "planning_failed"
    assert caught.value.status_code == 500
    assert isinstance(caught.value.__cause__, ValueError)
    assert timeout_field in str(caught.value.__cause__)
    for spy in work_spies:
        spy.assert_not_called()
    assert container.event_log.list_events() == []


@pytest.mark.parametrize("value", INVALID_TIMEOUTS)
def test_mutated_timeouts_preserve_http_preview_error_contract(
    application: FastAPI,
    work_spies: tuple[Mock, AsyncMock, AsyncMock],
    timeout_field: str,
    value: object,
) -> None:
    container: AppContainer = application.state.container
    setattr(container.runner._safety_limits, timeout_field, value)
    with TestClient(application) as client:
        response = client.post("/retrieve", json={"query": "GraphRAG evidence"})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "planning_failed"
    assert response.headers["cache-control"] == "no-store"
    assert "sources" not in response.json()
    assert timeout_field not in response.text
    for spy in work_spies:
        spy.assert_not_called()
    assert container.event_log.list_events() == []


@pytest.mark.parametrize("budget_seconds", [0.1, 1.25, 1e200, sys.float_info.max])
async def test_finite_library_timeouts_survive_run_snapshot_and_export_replay(
    application: FastAPI, tmp_path: Path, budget_seconds: float
) -> None:
    container: AppContainer = application.state.container
    container.runner._safety_limits = SafetyLimits(
        retrieval_timeout_seconds=budget_seconds, reasoning_timeout_seconds=budget_seconds
    )
    result = await asyncio.wait_for(container.runner.run("GraphRAG evidence"), timeout=2)
    assert result.state == AgentState.DONE, result.error
    original = container.evidence_exporter.export(result.run_id)
    reopened = EvidenceExporter(SQLiteEventLog(tmp_path / "timeouts.sqlite3")).export(result.run_id)
    encoded = json.dumps(reopened.model_dump(mode="json"), allow_nan=False)
    replayed = EvidenceBundle.model_validate_json(encoded)
    assert replayed == original
    assert replayed.configuration.retrieval_timeout_seconds == budget_seconds
    assert replayed.configuration.reasoning_timeout_seconds == budget_seconds
    assert replayed.events[0].payload is not None
    assert replayed.events[0].payload["payload"][
        "configuration"
    ] == replayed.configuration.model_dump(mode="json")
    assert replayed.snapshot == original.snapshot
