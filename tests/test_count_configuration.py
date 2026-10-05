"""Regression contracts for integer execution bounds and per-request copies."""

import asyncio
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.models import AgentState, QueryPlan
from agent.retrieval_preview import RetrievalPreviewError
from agent.safety import SafetyLimits
from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import SearchResult
from storage.event_log import SQLiteEventLog

COUNT_FIELDS = ("max_source_docs", "max_hops")


class SensitiveCount:
    def __repr__(self) -> str:
        return "sensitive-count-value"

    def __int__(self) -> int:
        return 1


INVALID_COUNTS = [
    pytest.param(field, value, id=f"{field}-{label}")
    for field in COUNT_FIELDS
    for label, value in (
        ("negative", -1),
        ("false", False),
        ("true", True),
        ("zero-float", 0.0),
        ("integral-float", 1.0),
        ("fraction", 1.5),
        ("numeric-string", "1"),
        ("nonnumeric-string", "sensitive-count-value"),
        ("none", None),
        ("nan", float("nan")),
        ("positive-infinity", float("inf")),
        ("negative-infinity", float("-inf")),
        ("sensitive-object", SensitiveCount()),
    )
] + [pytest.param("max_source_docs", 0, id="max_source_docs-zero")]


def count_error(field: str) -> str:
    minimum = 1 if field == "max_source_docs" else 0
    return f"{field} must be an integer >= {minimum}."


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    send = AsyncMock(side_effect=AssertionError("Count tests must not make live HTTP calls."))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield
    send.assert_not_called()


@pytest.mark.parametrize(("field", "value"), INVALID_COUNTS)
@pytest.mark.parametrize("construction", ["constructor", "replace", "mutated-copy"])
def test_invalid_count_configuration_is_rejected(
    field: str, value: object, construction: str
) -> None:
    with pytest.raises(ValueError) as caught:
        if construction == "constructor":
            SafetyLimits(**{field: value})
        elif construction == "replace":
            replace(SafetyLimits(), **{field: value})
        else:
            limits = SafetyLimits()
            setattr(limits, field, value)
            replace(limits)
    assert str(caught.value) == count_error(field)


@pytest.fixture
def application(tmp_path: Path) -> FastAPI:
    return create_app(offline_settings(tmp_path / "counts.sqlite3"))


@pytest.fixture
def work_spies(application: FastAPI, monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, ...]:
    container: AppContainer = application.state.container
    runner = container.runner
    analyze = Mock(wraps=runner._analyzer.analyze)
    plan = Mock(wraps=runner._planner.plan)
    retrieve = AsyncMock(wraps=runner._executor.retrieve)
    prepare = AsyncMock(wraps=runner._executor.prepare_context)
    generate = AsyncMock(wraps=container.llm.generate)
    monkeypatch.setattr(runner._analyzer, "analyze", analyze)
    monkeypatch.setattr(runner._planner, "plan", plan)
    monkeypatch.setattr(runner._executor, "retrieve", retrieve)
    monkeypatch.setattr(runner._executor, "prepare_context", prepare)
    monkeypatch.setattr(container.llm, "generate", generate)
    return analyze, plan, retrieve, prepare, generate


@pytest.mark.parametrize(("field", "value"), INVALID_COUNTS)
async def test_mutated_counts_return_durable_run_error_before_work(
    application: FastAPI,
    work_spies: tuple[Mock, ...],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    container: AppContainer = application.state.container
    setattr(container.runner._safety_limits, field, value)
    result = await asyncio.wait_for(container.runner.run("GraphRAG evidence"), timeout=2)
    assert result.state == AgentState.ERROR
    assert result.error == count_error(field)
    assert result.observation is None and result.plan is None and result.answer is None
    for spy in work_spies:
        spy.assert_not_called()
    events = SQLiteEventLog(tmp_path / "counts.sqlite3").list_events(result.run_id)
    assert len(events) == 1
    assert events[0]["payload"]["from_state"] == "IDLE"
    assert events[0]["payload"]["to_state"] == "ERROR"
    assert events[0]["payload"]["payload"] == {"error": count_error(field)}


@pytest.mark.parametrize(("field", "value"), INVALID_COUNTS)
async def test_mutated_counts_preserve_python_preview_error_contract(
    application: FastAPI,
    work_spies: tuple[Mock, ...],
    field: str,
    value: object,
) -> None:
    container: AppContainer = application.state.container
    setattr(container.runner._safety_limits, field, value)
    with pytest.raises(RetrievalPreviewError) as caught:
        await asyncio.wait_for(container.runner.preview("GraphRAG evidence"), timeout=2)
    assert caught.value.code == "planning_failed"
    assert caught.value.status_code == 500
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == count_error(field)
    assert str(caught.value) == (
        "Retrieval preview failed during planning; no partial evidence was returned."
    )
    for spy in work_spies:
        spy.assert_not_called()
    assert container.event_log.list_events() == []


@pytest.mark.parametrize(("field", "value"), INVALID_COUNTS)
def test_mutated_counts_preserve_http_preview_error_contract(
    application: FastAPI,
    work_spies: tuple[Mock, ...],
    field: str,
    value: object,
) -> None:
    container: AppContainer = application.state.container
    setattr(container.runner._safety_limits, field, value)
    with TestClient(application) as client:
        response = client.post("/retrieve", json={"query": "GraphRAG evidence"})
    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "planning_failed",
        "message": "Retrieval preview failed during planning; no partial evidence was returned.",
    }
    assert response.headers["cache-control"] == "no-store"
    assert field not in response.text
    assert "sensitive-count-value" not in response.text
    for spy in work_spies:
        spy.assert_not_called()
    assert container.event_log.list_events() == []


@pytest.mark.parametrize(("max_sources", "max_hops"), [(1, 0), (50, 5), (100, 8)])
def test_valid_count_bounds_and_requested_clamping_are_preserved(
    max_sources: int, max_hops: int
) -> None:
    original = SafetyLimits(max_source_docs=max_sources, max_hops=max_hops)
    for limits in (original, replace(original)):
        assert limits.max_source_docs == max_sources
        assert limits.max_hops == max_hops
        assert type(limits.max_source_docs) is int
        assert type(limits.max_hops) is int
        assert limits.clamp_sources(max_sources) == max_sources
        assert limits.clamp_sources(max_sources + 1) == max_sources
        assert limits.clamp_hops(max_hops) == max_hops
        assert limits.clamp_hops(max_hops + 1) == max_hops
        assert limits.clamp_sources(-3) == 1
        assert limits.clamp_sources(0) == 1
        assert limits.clamp_hops(-3) == 0
        assert limits.clamp_hops(0) == 0


def test_safety_defaults_are_unchanged() -> None:
    limits = SafetyLimits()
    assert limits.max_source_docs == 50
    assert limits.max_hops == 5
    assert limits.retrieval_timeout_seconds == 30.0
    assert limits.reasoning_timeout_seconds == 60.0


@pytest.mark.parametrize("entrypoint", ["run", "preview"])
@pytest.mark.parametrize(("max_sources", "max_hops"), [(1, 0), (50, 5), (100, 8)])
async def test_valid_count_configuration_reaches_work_and_snapshot(
    application: FastAPI,
    work_spies: tuple[Mock, ...],
    entrypoint: str,
    max_sources: int,
    max_hops: int,
) -> None:
    container: AppContainer = application.state.container
    container.runner._safety_limits = SafetyLimits(max_source_docs=max_sources, max_hops=max_hops)
    if entrypoint == "run":
        result = await asyncio.wait_for(container.runner.run("GraphRAG evidence"), timeout=2)
        assert result.state == AgentState.DONE, result.error
        bundle = container.evidence_exporter.export(result.run_id)
        configuration = bundle.configuration
        tasks = bundle.plan.tasks
        work_spies[4].assert_called_once()
    else:
        preview = await asyncio.wait_for(container.runner.preview("GraphRAG evidence"), timeout=2)
        configuration = preview.configuration
        tasks = preview.plan.tasks
        work_spies[4].assert_not_called()
        assert container.event_log.list_events() == []
    assert configuration.max_source_docs == max_sources
    assert configuration.max_hops == max_hops
    assert tasks and all(task.max_hops <= max_hops for task in tasks)
    for spy in work_spies[:4]:
        spy.assert_called_once()
    assert work_spies[2].call_args.args[1] == max_sources


@pytest.mark.parametrize("entrypoint", ["run", "preview"])
async def test_inflight_count_copy_survives_later_invalid_mutations(
    application: FastAPI, monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    container: AppContainer = application.state.container
    runner = container.runner
    limits = runner._safety_limits
    limits.max_source_docs = 2
    limits.max_hops = 1
    retrieve = runner._executor.retrieve

    async def mutate_shared_limits(plan: QueryPlan, max_results: int) -> list[SearchResult]:
        assert max_results == 2
        assert plan.tasks and all(task.max_hops <= 1 for task in plan.tasks)
        limits.max_source_docs = 0
        limits.max_hops = -1
        return await retrieve(plan, max_results)

    monkeypatch.setattr(runner._executor, "retrieve", mutate_shared_limits)
    if entrypoint == "run":
        result = await asyncio.wait_for(runner.run("GraphRAG evidence"), timeout=2)
        assert result.state == AgentState.DONE, result.error
        configuration = container.evidence_exporter.export(result.run_id).configuration
    else:
        preview = await asyncio.wait_for(runner.preview("GraphRAG evidence"), timeout=2)
        configuration = preview.configuration
        assert container.event_log.list_events() == []
    assert configuration.max_source_docs == 2
    assert configuration.max_hops == 1
    assert limits.max_source_docs == 0
    assert limits.max_hops == -1
