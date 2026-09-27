"""Acceptance contracts for scoped, model-free, per-paper research worksheets."""

import asyncio
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.evidence import text_digest
from agent.models import QueryPlan
from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document, SearchResult

ENDPOINT = "/research/worksheet"
QUESTIONS = ["  How does GraphRAG retrieve passages?  ", "Compare local retrieval methods."]


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Worksheets must not generate, journal, or use external HTTP.")


@dataclass
class WorksheetAPI:
    application: FastAPI
    container: AppContainer
    client: TestClient
    database: Path

    def add(self, document_id: str, text: str = "GraphRAG local retrieval.") -> Chunk:
        chunk = Chunk(
            chunk_id=f"chunk-{document_id}",
            document_id=document_id,
            title=f"Synthetic {document_id}",
            text=text,
            source="synthetic:worksheet",
        )
        self.container.document_store.add_documents(
            [Document(**chunk.model_dump(exclude={"chunk_id"}))], [chunk]
        )
        self.container.hybrid_retriever.add_chunks([chunk])
        self.container.graph_builder.index_chunks([chunk])
        return chunk

    def post(self, **updates: object) -> httpx.Response:
        return self.client.post(
            ENDPOINT, json={"questions": QUESTIONS, "document_ids": ["a", "b"], **updates}
        )


@pytest.fixture
def worksheet_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[WorksheetAPI]:
    database = tmp_path / "worksheet.sqlite3"
    settings = offline_settings(database).model_copy(update={"max_source_docs": 2, "max_hops": 1})
    application = create_app(settings)
    container: AppContainer = application.state.container
    for target, method in (
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (container.llm, "generate"),
        (container.runner, "run"),
        (container.runner._executor, "answer"),
        (container.runner._executor._grounder, "ground"),
        (container.event_log, "append_event"),
        (container.event_log, "append_transition"),
    ):
        monkeypatch.setattr(target, method, denied)
    with TestClient(application) as client:
        api = WorksheetAPI(application, container, client, database)
        api.add("a")
        api.add("b")
        api.add("excluded", "GraphRAG EXCLUDED_MARKER.")
        yield api
    assert container.event_log.list_events() == []


def assert_private(response: httpx.Response) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_each_question_gets_each_paper_not_a_global_top_k(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    api.container.runner._safety_limits.max_source_docs = 1
    preview = AsyncMock(wraps=api.container.runner.preview)
    monkeypatch.setattr(api.container.runner, "preview", preview)
    response = api.post(document_ids=[" b ", "a", "b"])
    assert response.status_code == 200, response.text
    assert_private(response)
    assert response.headers["content-disposition"] == 'attachment; filename="worksheet.json"'
    body = response.json()
    assert body["kind"] == "retrieval_passages"
    assert [document["document_id"] for document in body["documents"]] == ["b", "a"]
    assert [row["question"] for row in body["rows"]] == QUESTIONS
    assert len(body["rows"]) == 2
    assert [(call.args[0], call.kwargs["document_ids"]) for call in preview.await_args_list] == [
        (question, (document_id,)) for question in QUESTIONS for document_id in ("b", "a")
    ]
    for row in body["rows"]:
        assert [cell["document_id"] for cell in row["cells"]] == ["b", "a"]
        for cell in row["cells"]:
            assert cell["status"] == "passages_returned"
            assert cell["preview_source_count"] == 1
            assert cell["passages_omitted"] == 0
            assert {passage["document_id"] for passage in cell["passages"]} == {cell["document_id"]}
            assert cell["configuration"]["max_source_docs"] == 1
    assert "EXCLUDED_MARKER" not in response.text
    assert "answer" not in body and "run_id" not in body
    assert "support" not in {cell["status"] for row in body["rows"] for cell in row["cells"]}
    assert len(response.content) <= 262144
    assert api.client.get("/runs").json()["runs"] == []


@pytest.mark.parametrize(
    "updates",
    [
        {"questions": []},
        {"questions": ["x"] * 6},
        {"questions": [" "]},
        {"questions": ["x" * 501]},
        {"questions": [42]},
        {"questions": "retrieval"},
        {"document_ids": None},
        {"document_ids": []},
        {"document_ids": ["a"] * 11},
        {"document_ids": [42]},
        {"document_ids": ["x" * 129]},
        {"passages_per_cell": 0},
        {"passages_per_cell": 4},
        {"passages_per_cell": True},
        {"passages_per_cell": "2"},
        {"collection_id": "col_" + "0" * 32},
        {"collection_id": None},
        {"unexpected": True},
    ],
)
def test_invalid_request_rejected_before_preview(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, updates: dict[str, object]
) -> None:
    monkeypatch.setattr(worksheet_api.container.runner, "preview", denied)
    response = worksheet_api.post(**updates)
    assert response.status_code == 422, response.text


def test_missing_unknown_and_broken_selections_do_not_widen(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    collection = api.container.paper_collections.create(name="Review", document_ids=["a", "b"])
    monkeypatch.setattr(api.container.runner, "preview", denied)
    assert api.client.post(ENDPOINT, json={"questions": QUESTIONS}).status_code == 422
    unknown = api.post(document_ids=["a", "unknown"])
    assert unknown.status_code == 422
    assert unknown.json()["detail"]["code"] == "unknown_documents"
    assert unknown.json()["detail"]["missing_document_ids"] == ["unknown"]
    assert_private(unknown)
    missing = api.client.post(
        ENDPOINT, json={"questions": QUESTIONS, "collection_id": "col_" + "0" * 32}
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "collection_not_found"
    with closing(sqlite3.connect(api.database)) as connection, connection:
        connection.execute("DELETE FROM documents WHERE document_id = 'b'")
    broken = api.client.post(
        ENDPOINT, json={"questions": QUESTIONS, "collection_id": collection.collection_id}
    )
    assert broken.status_code == 409
    assert broken.json()["detail"]["code"] == "collection_documents_missing"


def test_excerpts_match_preview_rank_score_path_and_digest(
    worksheet_api: WorksheetAPI,
) -> None:
    api = worksheet_api
    text = "GraphRAG caf\u00e9 \u7814\u7a76\n" * 100
    original = api.add("long", text)
    preview_response = api.client.post(
        "/retrieve", json={"query": QUESTIONS[0], "document_ids": ["long"]}
    )
    assert preview_response.status_code == 200
    expected = preview_response.json()["sources"][0]
    response = api.post(document_ids=["long"], questions=[QUESTIONS[0]], passages_per_cell=1)
    assert response.status_code == 200
    passage = response.json()["rows"][0]["cells"][0]["passages"][0]
    for field in ("rank", "score", "path", "retriever", "text_sha256"):
        assert passage[field] == expected[field]
    assert passage["chunk_id"] == original.chunk_id
    assert passage["excerpt"] == text[:800]
    assert passage["excerpt_truncated"] is True
    assert passage["text_characters"] == len(text)
    assert passage["text_sha256"] == text_digest(text)
    inspection = response.json()["documents"][0]["inspection_url"]
    inspected = api.client.get(inspection)
    assert inspected.status_code == 200
    assert inspected.json()["chunks"][0]["chunk_id"] == passage["chunk_id"]


def test_no_passages_and_zero_score_are_not_scientific_findings(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    empty = AsyncMock(return_value=[])
    monkeypatch.setattr(api.container.runner._executor, "retrieve", empty)
    response = api.post()
    assert response.status_code == 200
    assert all(
        cell["status"] == "no_passages" and cell["passages"] == []
        for row in response.json()["rows"]
        for cell in row["cells"]
    )
    source = api.add("zero")
    monkeypatch.setattr(
        api.container.runner._executor,
        "retrieve",
        AsyncMock(return_value=[SearchResult(chunk=source, score=0, retriever="synthetic")]),
    )
    monkeypatch.setattr(
        api.container.runner._executor._reranker,
        "rerank",
        AsyncMock(return_value=[SearchResult(chunk=source, score=0, retriever="synthetic")]),
    )
    response = api.post(document_ids=["zero"])
    assert response.status_code == 200
    cell = response.json()["rows"][0]["cells"][0]
    assert cell["status"] == "passages_returned"
    assert cell["passages"][0]["score"] == 0


def test_failure_after_a_completed_cell_is_sanitized_not_partial_success(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    retrieve = api.container.runner._executor.retrieve
    calls = 0

    async def fail_second(
        plan: QueryPlan, max_results: int = 8, *, document_ids: tuple[str, ...] | None = None
    ) -> list[SearchResult]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("PRIVATE_SOURCE secret-key")
        return await retrieve(plan, max_results, document_ids=document_ids)

    monkeypatch.setattr(api.container.runner._executor, "retrieve", fail_second)
    response = api.post()
    assert response.status_code == 500
    assert_private(response)
    assert response.json()["detail"]["code"] == "retrieval_failed"
    assert response.json()["detail"]["question_index"] == 0
    assert response.json()["detail"]["document_index"] == 1
    assert "rows" not in response.json() and "PRIVATE_SOURCE" not in response.text
    assert "secret-key" not in response.text
    assert calls == 2


def test_markdown_keeps_user_and_corpus_content_literal(
    worksheet_api: WorksheetAPI,
) -> None:
    api = worksheet_api
    attack = '```</script><img src="https://invalid.example/track">[x](javascript:alert(1))\n|x|'
    api.add("literal", attack)
    response = api.client.post(
        ENDPOINT,
        params={"format": "markdown"},
        json={"questions": [attack], "document_ids": ["literal"]},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"] == 'attachment; filename="worksheet.md"'
    assert_private(response)
    assert "| Question | Paper 1 |" in response.text
    assert f"````text\n{attack}\n````" in response.text
    assert len(response.content) <= 262144
    assert api.client.post(ENDPOINT, params={"format": "html"}, json={}).status_code == 422


def test_all_fifty_cells_and_collection_membership_bound(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    identifiers = [f"paper-{index}" for index in range(10)]
    for identifier in identifiers:
        api.add(identifier)
    collection = api.container.paper_collections.create(name="Ten papers", document_ids=identifiers)
    response = api.client.post(
        ENDPOINT, json={"questions": ["retrieval"] * 5, "collection_id": collection.collection_id}
    )
    assert response.status_code == 200, response.text
    assert sum(len(row["cells"]) for row in response.json()["rows"]) == 50
    assert len(response.content) <= 262144
    assert response.json()["collection_id"] == collection.collection_id
    api.add("eleventh")
    api.container.paper_collections.replace(
        collection.collection_id,
        name="Eleven papers",
        document_ids=[*identifiers, "eleventh"],
        expected_revision=1,
    )
    monkeypatch.setattr(api.container.runner, "preview", denied)
    response = api.client.post(
        ENDPOINT, json={"questions": ["retrieval"], "collection_id": collection.collection_id}
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "worksheet_selection_too_large"


async def test_python_request_and_collection_are_frozen_before_first_await(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.research_worksheet import WorksheetRequest

    api = worksheet_api
    supplied_questions = ["retrieval", "GraphRAG"]
    supplied_ids = ["a", "b"]
    request = WorksheetRequest(questions=supplied_questions, document_ids=supplied_ids)
    supplied_questions[:] = ["caller changed query"]
    supplied_ids[:] = ["excluded"]
    collection = api.container.paper_collections.create(name="Frozen", document_ids=["b", "a"])
    resolve = Mock(wraps=api.container.paper_collections.resolve)
    monkeypatch.setattr(api.container.paper_collections, "resolve", resolve)
    retrieve = api.container.runner._executor.retrieve
    started, release = asyncio.Event(), asyncio.Event()
    scopes: list[tuple[str, ...]] = []

    async def held(
        plan: QueryPlan, max_results: int = 8, *, document_ids: tuple[str, ...] | None = None
    ) -> list[SearchResult]:
        assert document_ids is not None
        scopes.append(document_ids)
        started.set()
        await release.wait()
        return await retrieve(plan, max_results, document_ids=document_ids)

    monkeypatch.setattr(api.container.runner._executor, "retrieve", held)
    selected = WorksheetRequest(questions=request.questions, collection_id=collection.collection_id)
    pending = asyncio.create_task(api.container.worksheets.build(selected))
    try:
        await asyncio.wait_for(started.wait(), 2)
        api.container.paper_collections.delete(collection.collection_id, expected_revision=1)
    finally:
        release.set()
    worksheet = await asyncio.wait_for(pending, 2)
    assert resolve.call_count == 1
    assert scopes == [("b",), ("a",), ("b",), ("a",)]
    assert [row.question for row in worksheet.rows] == ["retrieval", "GraphRAG"]
    assert request.document_ids == ("a", "b")
    assert [document.document_id for document in worksheet.documents] == ["b", "a"]


async def test_cancellation_propagates_without_a_result_or_journal(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.research_worksheet import WorksheetRequest

    started = asyncio.Event()
    cancelled: list[bool] = []

    async def block(*args: object, **kwargs: object) -> NoReturn:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
        raise AssertionError("Cancelled worksheet continued")

    monkeypatch.setattr(worksheet_api.container.runner._executor, "retrieve", block)
    pending = asyncio.create_task(
        worksheet_api.container.worksheets.build(
            WorksheetRequest(questions=QUESTIONS, document_ids=["a", "b"])
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        pending.cancel("worksheet cancelled")
        with pytest.raises(asyncio.CancelledError, match="worksheet cancelled"):
            await pending
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    assert cancelled == [True]


def test_overall_deadline_cancels_current_cell_and_stops_later_cells(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.research_worksheet as worksheet_module

    cancelled: list[bool] = []

    async def block(*args: object, **kwargs: object) -> NoReturn:
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
        raise AssertionError("Worksheet deadline did not fire")

    monkeypatch.setattr(worksheet_module, "WORKSHEET_TIMEOUT_SECONDS", 0.02)
    retrieve = AsyncMock(side_effect=block)
    monkeypatch.setattr(worksheet_api.container.runner._executor, "retrieve", retrieve)
    response = worksheet_api.post()
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "worksheet_timeout"
    assert_private(response)
    assert retrieve.await_count == 1 and cancelled == [True]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_fail_closed(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, score: float
) -> None:
    source = worksheet_api.add("invalid-score")
    monkeypatch.setattr(
        worksheet_api.container.runner._executor._reranker,
        "rerank",
        AsyncMock(return_value=[SearchResult(chunk=source, score=score, retriever="broken")]),
    )
    response = worksheet_api.post(document_ids=["invalid-score"])
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "context_preparation_failed"
    assert "rows" not in response.json()


def test_generative_retrieval_rejected_without_fake_or_live_generation(
    worksheet_api: WorksheetAPI,
) -> None:
    from retrieval.hyde import HyDEExpander

    worksheet_api.container.hybrid_retriever._hyde_expander = HyDEExpander(
        worksheet_api.container.llm
    )
    response = worksheet_api.post()
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "generative_retrieval"


@pytest.mark.parametrize("stage", ["retrieve", "rerank"])
def test_misscoped_results_are_errors_not_filtered_success(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    source = worksheet_api.add("excluded", "PRIVATE_SOURCE GraphRAG")
    executor = worksheet_api.container.runner._executor
    target = executor if stage == "retrieve" else executor._reranker
    monkeypatch.setattr(
        target,
        stage,
        AsyncMock(return_value=[SearchResult(chunk=source, score=1, retriever="bad")]),
    )
    response = worksheet_api.post()
    assert response.status_code == 500
    assert "PRIVATE_SOURCE" not in response.text and "rows" not in response.json()


def test_request_does_not_write_database_and_repeats_after_restart(
    worksheet_api: WorksheetAPI,
) -> None:
    api = worksheet_api
    before = api.database.read_bytes()
    first = api.post()
    second = api.post()
    assert first.status_code == 200 and second.content == first.content
    assert api.database.read_bytes() == before
    reopened = create_app(
        offline_settings(api.database).model_copy(update={"max_source_docs": 2, "max_hops": 1})
    )
    with TestClient(reopened) as client:
        response = client.post(ENDPOINT, json={"questions": QUESTIONS, "document_ids": ["a", "b"]})
        assert response.status_code == 200 and response.content == first.content
        assert client.get("/runs").json()["runs"] == []


def test_aggregate_serialization_limit_is_not_a_per_cell_proxy(
    worksheet_api: WorksheetAPI,
) -> None:
    api = worksheet_api
    api.container.runner._safety_limits.max_source_docs = 3
    identifiers = []
    for index in range(10):
        identifier = f"large-{index}"
        identifiers.append(identifier)
        chunks = [
            Chunk(
                chunk_id=f"{identifier}-{ordinal}",
                document_id=identifier,
                title="\U0001f52c" * 300,
                text="GraphRAG retrieval " + "\U0001f52c" * 1000,
                source="s" * 512,
            )
            for ordinal in range(3)
        ]
        api.container.document_store.add_documents(
            [Document(**chunks[0].model_dump(exclude={"chunk_id"}))], chunks
        )
        api.container.hybrid_retriever.add_chunks(chunks)
        api.container.graph_builder.index_chunks(chunks)
    response = api.post(questions=["retrieval"] * 5, document_ids=identifiers, passages_per_cell=3)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "worksheet_too_large"
    assert len(response.content) < 1024 and "rows" not in response.json()
    assert_private(response)


async def test_serializers_enforce_exact_utf8_byte_boundary(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.research_worksheet as worksheet_module
    from agent.research_worksheet import WorksheetError, WorksheetRequest

    result = await worksheet_api.container.worksheets.build(
        WorksheetRequest(questions=["caf\u00e9 retrieval"], document_ids=["a"])
    )
    for serializer in (result.to_json, result.to_markdown):
        monkeypatch.setattr(worksheet_module, "MAX_RESPONSE_BYTES", 262144)
        text = serializer()
        size = len(text.encode("utf-8"))
        monkeypatch.setattr(worksheet_module, "MAX_RESPONSE_BYTES", size)
        assert serializer() == text
        monkeypatch.setattr(worksheet_module, "MAX_RESPONSE_BYTES", size - 1)
        with pytest.raises(WorksheetError) as captured:
            serializer()
        assert captured.value.code == "worksheet_too_large"


def test_storage_unavailable_or_corrupt_collection_is_not_empty_success(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    collection = api.container.paper_collections.create(name="Corrupt", document_ids=["a"])
    monkeypatch.setattr(api.container.runner, "preview", denied)
    with closing(sqlite3.connect(api.database)) as connection, connection:
        connection.execute(
            "UPDATE paper_collections SET document_ids = ? WHERE collection_id = ?",
            (json.dumps(["a", "a"]), collection.collection_id),
        )
    response = api.client.post(
        ENDPOINT, json={"questions": QUESTIONS, "collection_id": collection.collection_id}
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_collection_record"
    api.database.rename(api.database.with_suffix(".moved"))
    unavailable = api.post()
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"]["code"] == "collection_storage_error"
    assert not api.database.exists()
    api.database.with_suffix(".moved").rename(api.database)


@pytest.mark.parametrize("length", [799, 800, 801])
def test_excerpt_and_label_prefixes_have_truthful_truncation_flags(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, length: int
) -> None:
    api = worksheet_api
    chunk = api.add("a", "x" * length)
    chunk.title = "t" * 301
    chunk.source = "s" * 513
    result = SearchResult(chunk=chunk, score=0.25, retriever="custom", path=["one", "two"])
    monkeypatch.setattr(
        api.container.runner._executor, "retrieve", AsyncMock(return_value=[result])
    )
    response = api.post(document_ids=["a"], questions=["retrieval"])
    assert response.status_code == 200
    passage = response.json()["rows"][0]["cells"][0]["passages"][0]
    assert passage["excerpt"] == chunk.text[:800]
    assert passage["excerpt_truncated"] is (length > 800)
    assert passage["title"] == "t" * 300 and passage["title_truncated"] is True
    assert passage["source"] == "s" * 512 and passage["source_truncated"] is True
    assert passage["text_characters"] == length
    assert passage["path"] == ["one", "two", "custom"]


def test_passage_limit_preserves_ranks_and_reports_omitted_results(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    chunk = api.add("a")
    api.container.runner._safety_limits.max_source_docs = 3
    results = [
        SearchResult(
            chunk=chunk.model_copy(update={"chunk_id": f"chunk-a-{rank}"}),
            score=1 / rank,
            retriever="custom",
        )
        for rank in range(1, 4)
    ]
    monkeypatch.setattr(api.container.runner._executor, "retrieve", AsyncMock(return_value=results))
    response = api.post(document_ids=["a"], questions=["retrieval"], passages_per_cell=1)
    assert response.status_code == 200
    cell = response.json()["rows"][0]["cells"][0]
    assert cell["preview_source_count"] == 3 and cell["passages_omitted"] == 2
    assert [passage["rank"] for passage in cell["passages"]] == [1]
    assert [passage["chunk_id"] for passage in cell["passages"]] == ["chunk-a-1"]


def test_all_configured_providers_and_fake_remain_unused(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm.fake import FakeLLMAdapter
    from llm.providers import AnthropicAdapter, GeminiAdapter, KimiAdapter, OpenAIAdapter
    from llm.router import RoutingLLMAdapter

    generators = []
    for adapter in (
        FakeLLMAdapter,
        RoutingLLMAdapter,
        OpenAIAdapter,
        AnthropicAdapter,
        GeminiAdapter,
        KimiAdapter,
    ):
        generate = AsyncMock(side_effect=denied)
        monkeypatch.setattr(adapter, "generate", generate)
        generators.append(generate)
    settings = offline_settings(worksheet_api.database).model_copy(
        update={
            "openai_api_key": "synthetic-unused-openai",
            "anthropic_api_key": "synthetic-unused-anthropic",
            "gemini_api_key": "synthetic-unused-gemini",
            "moonshot_api_key": "synthetic-unused-kimi",
        }
    )
    application = create_app(settings)
    with TestClient(application) as client:
        response = client.post(ENDPOINT, json={"questions": QUESTIONS, "document_ids": ["a", "b"]})
        assert response.status_code == 200, response.text
        assert "synthetic-unused" not in response.text
    for generate in generators:
        generate.assert_not_awaited()
    assert application.state.container.event_log.list_events() == []


def test_deleted_paper_during_work_is_not_a_successful_stale_selection(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    retrieve = api.container.runner._executor.retrieve

    async def delete_first(
        plan: QueryPlan, max_results: int = 8, *, document_ids: tuple[str, ...] | None = None
    ) -> list[SearchResult]:
        with closing(sqlite3.connect(api.database)) as connection, connection:
            connection.execute("DELETE FROM documents WHERE document_id = 'a'")
        return await retrieve(plan, max_results, document_ids=document_ids)

    monkeypatch.setattr(api.container.runner._executor, "retrieve", delete_first)
    response = api.post()
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "worksheet_documents_changed"
    assert "rows" not in response.json()


def test_corrupt_preview_digest_is_not_relabelled_as_valid_evidence(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.retrieval_preview import RetrievalPreview

    preview = worksheet_api.container.runner.preview

    async def corrupt(
        query: str, *, document_ids: tuple[str, ...] | None = None
    ) -> RetrievalPreview:
        result = await preview(query, document_ids=document_ids)
        return result.model_copy(update={"context_sha256": "0" * 64})

    monkeypatch.setattr(worksheet_api.container.runner, "preview", corrupt)
    response = worksheet_api.post()
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "invalid_worksheet_evidence"


@pytest.mark.parametrize("phase", ["retrieval", "context_preparation"])
def test_preview_phase_deadline_is_preserved(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    async def block(*args: object, **kwargs: object) -> NoReturn:
        await asyncio.sleep(10)
        raise AssertionError("Phase timeout did not fire")

    runner = worksheet_api.container.runner
    runner._safety_limits.retrieval_timeout_seconds = 0.01
    runner._safety_limits.reasoning_timeout_seconds = 0.01
    target = runner._executor if phase == "retrieval" else runner._executor._reranker
    monkeypatch.setattr(target, "retrieve" if phase == "retrieval" else "rerank", block)
    response = worksheet_api.post()
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == f"{phase}_timeout"


def test_deadline_overrun_in_synchronous_selection_cannot_return_success(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.research_worksheet as worksheet_module

    validate = worksheet_api.container.paper_collections.validate_document_ids

    def slow_validate(document_ids: tuple[str, ...]) -> tuple[str, ...]:
        time.sleep(0.03)
        return validate(document_ids)

    monkeypatch.setattr(worksheet_module, "WORKSHEET_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        worksheet_api.container.paper_collections, "validate_document_ids", slow_validate
    )
    monkeypatch.setattr(worksheet_api.container.runner, "preview", denied)
    response = worksheet_api.post()
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "worksheet_timeout"


def test_openapi_exposes_bounded_request_downloads_and_keeps_existing_routes(
    worksheet_api: WorksheetAPI,
) -> None:
    schema = worksheet_api.application.openapi()
    assert {ENDPOINT, "/query", "/retrieve", "/documents", "/collections"} <= schema["paths"].keys()
    request_schema = schema["components"]["schemas"]["WorksheetRequest"]
    assert request_schema["properties"]["questions"]["maxItems"] == 5
    assert request_schema["properties"]["document_ids"]["maxItems"] == 10
    assert request_schema["properties"]["passages_per_cell"]["maximum"] == 3
    assert {"application/json", "text/markdown"} <= (
        schema["paths"][ENDPOINT]["post"]["responses"]["200"]["content"].keys()
    )


def test_invalid_unicode_request_is_a_safe_private_validation_error(
    worksheet_api: WorksheetAPI,
) -> None:
    response = worksheet_api.client.post(
        ENDPOINT,
        content=json.dumps({"questions": ["bad \ud800"], "document_ids": ["a"]}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert_private(response)
    assert all("input" not in error for error in response.json()["detail"])
