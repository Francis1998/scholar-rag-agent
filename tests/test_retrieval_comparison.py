"""Compare actual preview policies before generation, without journaling or corpus writes."""

import asyncio
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings
from scripts.demo_near_duplicate_evidence import seed_corpus

from agent.evidence import text_digest
from agent.models import QueryPlan
from api.application import create_app
from api.dependencies import AppContainer
from retrieval.hyde import HyDEExpander
from retrieval.models import Chunk, SearchResult

ENDPOINT = "/research/compare-retrieval"
QUERY = "  retrieval evidence  "
POLICY = {
    "near_duplicate_threshold": 0.8,
    "max_chunks_per_document": 1,
    "min_evidence_documents": 4,
}


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Comparison must not generate, ground, journal, write, or use HTTP.")


@dataclass
class ComparisonAPI:
    application: FastAPI
    container: AppContainer
    client: TestClient
    database: Path

    def post(self, **updates: object) -> httpx.Response:
        before = self.database.read_bytes()
        response = self.client.post(
            ENDPOINT,
            json={
                "query": QUERY,
                "baseline": {},
                "candidate": {"evidence_policy": POLICY},
                **updates,
            },
        )
        assert self.database.read_bytes() == before
        return response


@pytest.fixture
def comparison_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ComparisonAPI]:
    database = tmp_path / "retrieval-comparison.sqlite3"
    application = create_app(
        offline_settings(database).model_copy(update={"max_source_docs": 5, "max_hops": 1})
    )
    container: AppContainer = application.state.container
    seed_corpus(container)
    for target, method in (
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (container.llm, "generate"),
        (container.runner, "run"),
        (container.runner._executor, "answer"),
        (container.runner._executor, "answer_prepared"),
        (container.runner._executor._grounder, "ground"),
        (container.event_log, "append_event"),
        (container.event_log, "append_transition"),
        (container.document_store, "add_documents"),
        (container.hybrid_retriever, "add_chunks"),
        (container.graph_builder, "index_chunks"),
    ):
        monkeypatch.setattr(target, method, denied)
    with TestClient(application) as client:
        yield ComparisonAPI(application, container, client, database)
    assert container.event_log.list_events() == []


def assert_private(response: httpx.Response) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_compare_before_query_preserves_real_previews_and_measured_deltas(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = comparison_api
    expected = [
        api.client.post("/retrieve", json={"query": QUERY, **options}).json()
        for options in ({}, POLICY)
    ]
    preview = AsyncMock(wraps=api.container.runner.preview)
    monkeypatch.setattr(api.container.runner, "preview", preview)
    response = api.post()
    assert response.status_code == 200, response.text
    assert_private(response)
    body = response.json()
    assert set(body) == {
        "schema_version",
        "kind",
        "query",
        "document_ids",
        "collection",
        "baseline",
        "candidate",
        "evidence",
        "documents",
        "source_changes",
        "changes",
        "any_changes",
        "context_bytes_delta",
        "limits",
        "notices",
    }
    assert body["schema_version"] == "1.0"
    assert body["kind"] == "retrieval_policy_comparison"
    assert body["query"] == QUERY
    assert body["document_ids"] is body["collection"] is None
    assert preview.await_count == 2
    assert preview.await_args_list[0].kwargs == {}
    assert preview.await_args_list[1].kwargs == POLICY
    for name, original in zip(("baseline", "candidate"), expected, strict=True):
        variant = body[name]
        assert variant["preview"] == original
        assert variant["status"] == "passages_returned"
        assert variant["source_count"] == len(original["sources"])
        assert variant["document_count"] == len(
            {source["chunk"]["document_id"] for source in original["sources"]}
        )
        assert variant["context_utf8_bytes"] == len(original["context"].encode("utf-8"))
        assert original["context_sha256"] == text_digest(original["context"])
        assert "run_id" not in original["plan"]
        assert original["configuration"]["max_source_docs"] == 5
        assert all(task["max_hops"] <= 1 for task in original["plan"]["tasks"])
    assert body["baseline"]["source_count"] == 5
    assert body["candidate"]["source_count"] == 2
    assert body["candidate"]["preview"]["evidence_assessment"] == {
        "required_documents": 4,
        "observed_documents": 2,
        "passed": False,
    }
    assert body["evidence"]["statistics"] == {
        "baseline_count": 5,
        "candidate_count": 2,
        "shared_count": 2,
        "added_count": 0,
        "removed_count": 3,
        "union_count": 5,
        "identity_jaccard": 0.4,
    }
    assert body["documents"]["statistics"]["baseline_count"] == 4
    assert body["documents"]["statistics"]["candidate_count"] == 2
    assert body["documents"]["statistics"]["identity_jaccard"] == 0.5
    assert body["source_changes"]["text_changed"] == 0
    assert body["source_changes"]["retriever_changed"] == 2
    assert body["context_bytes_delta"] == (
        body["candidate"]["context_utf8_bytes"] - body["baseline"]["context_utf8_bytes"]
    )
    assert body["context_bytes_delta"] < 0
    assert body["changes"]["evidence_policy_changed"] is True
    assert body["changes"]["configuration_changed"] is False
    assert body["any_changes"] is True
    notices = " ".join(body["notices"])
    for phrase in ("current corpus", "controlled experiment", "ground truth", "concurrent"):
        assert phrase in notices
    assert "answer" not in body and "generation" not in body and "run_id" not in body
    assert len(response.content) <= 262144
    assert api.client.get("/runs").json()["runs"] == []


@pytest.mark.parametrize(
    "updates",
    [
        {"query": ""},
        {"query": " \t\n"},
        {"query": None},
        {"query": 42},
        {"query": True},
        {"query": "x" * 501},
        {"query": "\ud800"},
        {"document_ids": None},
        {"document_ids": []},
        {"document_ids": "paper-a"},
        {"document_ids": [" "]},
        {"document_ids": [42]},
        {"document_ids": ["x" * 129]},
        {"document_ids": ["paper-a"] * 101},
        {"document_ids": ["\ud800"]},
        {"collection_id": None},
        {"collection_id": "not-a-collection"},
        {"collection_id": "col_" + "0" * 32, "document_ids": ["paper-a"]},
        {"baseline": None},
        {"candidate": []},
        {"baseline": {"evidence_policy": None}},
        {"candidate": {"evidence_policy": {}}},
        {"candidate": {"evidence_policy": {"unknown": 1}}},
        {"candidate": {"evidence_policy": {"max_chunks_per_document": None}}},
        {"candidate": {"evidence_policy": {"max_chunks_per_document": 0}}},
        {"candidate": {"evidence_policy": {"max_chunks_per_document": 51}}},
        {"candidate": {"evidence_policy": {"max_chunks_per_document": True}}},
        {"candidate": {"evidence_policy": {"min_evidence_documents": "2"}}},
        {"candidate": {"evidence_policy": {"min_evidence_documents": None}}},
        {"candidate": {"evidence_policy": {"min_evidence_documents": 51}}},
        {"candidate": {"evidence_policy": {"near_duplicate_threshold": None}}},
        {"candidate": {"evidence_policy": {"near_duplicate_threshold": 0}}},
        {"candidate": {"evidence_policy": {"near_duplicate_threshold": 1.1}}},
        {"candidate": {"evidence_policy": {"near_duplicate_threshold": True}}},
        {"candidate": {"evidence_policy": {"near_duplicate_threshold": "0.8"}}},
        {"candidate": {"document_ids": ["paper-a"]}},
        {"candidate": {"query": "different query"}},
        {"third": {}},
        {"max_source_docs": 10},
        {"provider": "fake"},
    ],
)
def test_invalid_inputs_are_private_422_before_any_activity(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch, updates: dict[str, object]
) -> None:
    api = comparison_api
    monkeypatch.setattr(api.container.runner, "preview", denied)
    monkeypatch.setattr(api.container.paper_collections, "resolved_collection", denied)
    payload = {"query": QUERY, "baseline": {}, "candidate": {}, **updates}
    response = api.client.post(
        ENDPOINT, content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422, response.text
    assert_private(response)
    assert all(set(error) == {"loc", "type", "msg"} for error in response.json()["detail"])


@pytest.mark.parametrize("name", ["baseline", "candidate"])
def test_both_named_variants_are_required(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(comparison_api.container.runner, "preview", denied)
    response = comparison_api.client.post(ENDPOINT, json={"query": QUERY, name: {}})
    assert response.status_code == 422
    assert_private(response)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_nested_policies_do_not_break_error_serialization(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    monkeypatch.setattr(comparison_api.container.runner, "preview", denied)
    payload = {
        "query": "PRIVATE_QUERY",
        "baseline": {},
        "candidate": {"evidence_policy": {"near_duplicate_threshold": value}},
    }
    response = comparison_api.client.post(
        ENDPOINT, content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert "PRIVATE_QUERY" not in response.text
    assert_private(response)


def test_defaults_unknown_scope_and_empty_overlap(comparison_api: ComparisonAPI) -> None:
    same = comparison_api.post(candidate={})
    assert same.status_code == 200
    body = same.json()
    assert body["baseline"] == body["candidate"]
    assert body["any_changes"] is False
    assert not any(body["source_changes"].values())
    empty = comparison_api.post(candidate={}, document_ids=["unknown"])
    assert empty.status_code == 200
    body = empty.json()
    for name in ("baseline", "candidate"):
        assert body[name]["status"] == "no_passages"
        assert body[name]["source_count"] == body[name]["context_utf8_bytes"] == 0
        assert body[name]["preview"]["context_sha256"] == text_digest("")
    for field in ("evidence", "documents"):
        assert body[field]["statistics"]["identity_jaccard"] is None
        assert body[field]["statistics"]["union_count"] == 0
    assert body["context_bytes_delta"] == 0
    assert "not an absence" in " ".join(body["notices"])


def test_empty_minimum_assessment_is_preserved(comparison_api: ComparisonAPI) -> None:
    response = comparison_api.post(document_ids=["unknown"])
    assert response.status_code == 200
    body = response.json()
    assert body["candidate"]["status"] == "no_passages"
    assert body["candidate"]["preview"]["evidence_assessment"]["passed"] is False
    assert body["changes"]["evidence_policy_changed"] is True
    assert body["changes"]["context_changed"] is False
    assert body["changes"]["evidence_assessment_changed"] is True
    assert body["any_changes"] is True


def _result(chunk_id: str, document_id: str, text: str = "retrieval evidence") -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title="Synthetic",
            text=text,
            source="synthetic:comparison",
            metadata={"fixture": "true"},
        ),
        score=0.5,
        retriever="fixture",
        path=["upstream"],
    )


def test_exact_identity_rank_score_text_and_provenance_deltas(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = [_result("removed", "doc-a"), _result("shared", "doc-b"), _result("moved", "doc-c")]
    changed = _result("shared", "doc-b", "changed caf\u00e9 evidence")
    changed.score = 0.875
    changed.retriever = "different"
    changed.path = ["different", "upstream"]
    changed.chunk.title = "Changed title"
    changed.chunk.source = "synthetic:changed"
    changed.chunk.metadata = {"fixture": "changed"}
    right = [changed, _result("added", "doc-d"), _result("moved", "new-owner")]
    executor = comparison_api.container.runner._executor
    monkeypatch.setattr(executor, "retrieve", AsyncMock(side_effect=[left, right]))
    monkeypatch.setattr(executor._reranker, "rerank", AsyncMock(side_effect=lambda q, r: r))
    response = comparison_api.post(candidate={})
    assert response.status_code == 200
    body = response.json()
    evidence = body["evidence"]
    assert evidence["statistics"]["identity_jaccard"] == 0.2
    assert [(s["chunk_id"], s["document_id"]) for s in evidence["added"]] == [
        ("added", "doc-d"),
        ("moved", "new-owner"),
    ]
    assert [s["chunk_id"] for s in evidence["removed"]] == ["removed", "moved"]
    assert evidence["reassigned_chunk_ids"] == ["moved"]
    shared = evidence["shared"][0]
    assert shared["rank_delta"] == -1
    assert shared["baseline"]["score"] == 0.5
    assert shared["candidate"]["score"] == 0.875
    assert all(shared["changes"].values())
    assert all(value == 1 for value in body["source_changes"].values())
    assert body["documents"]["shared"] == ["doc-b"]
    assert body["documents"]["added"] == ["doc-d", "new-owner"]
    assert body["documents"]["removed"] == ["doc-a", "doc-c"]


@pytest.mark.parametrize("field", ["text", "source", "metadata", "retriever", "path", "score"])
def test_same_identity_does_not_hide_content_or_provenance_changes(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    left = _result("one", "one")
    right = left.model_copy(deep=True)
    if field in {"retriever", "path", "score"}:
        setattr(right, field, {"retriever": "new", "path": ["new"], "score": 0.0}[field])
    else:
        setattr(right.chunk, field, {"text": "changed", "source": "new", "metadata": {}}[field])
    executor = comparison_api.container.runner._executor
    monkeypatch.setattr(executor, "retrieve", AsyncMock(side_effect=[[left], [right]]))
    monkeypatch.setattr(executor._reranker, "rerank", AsyncMock(side_effect=lambda q, r: r))
    response = comparison_api.post(candidate={})
    assert response.status_code == 200
    body = response.json()
    assert body["evidence"]["statistics"]["identity_jaccard"] == 1
    assert body["changes"]["source_membership_changed"] is False
    assert body["changes"]["context_changed"] is (field == "text")
    assert body["source_changes"][f"{field}_changed"] == 1
    assert body["changes"]["evidence_changed"] is body["any_changes"] is True


async def test_python_scope_policy_and_collection_revision_are_copied_before_await(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.retrieval_comparison import RetrievalComparisonRequest

    api = comparison_api
    store = api.container.paper_collections
    collection = store.create(name="Synthetic selection", document_ids=["paper-b", "paper-a"])
    resolve = Mock(wraps=store.resolved_collection)
    monkeypatch.setattr(store, "resolved_collection", resolve)
    original = api.container.runner._executor.retrieve
    started, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def held(
        plan: QueryPlan, max_results: int = 8, *, document_ids: tuple[str, ...] | None = None
    ) -> list[SearchResult]:
        seen.append((document_ids, plan.observation.evidence_policy))
        started.set()
        await release.wait()
        return await original(plan, max_results, document_ids=document_ids)

    monkeypatch.setattr(api.container.runner._executor, "retrieve", held)
    policy = {"max_chunks_per_document": 1}
    request = RetrievalComparisonRequest(
        query=QUERY,
        collection_id=collection.collection_id,
        baseline={},
        candidate={"evidence_policy": policy},
    )
    pending = asyncio.create_task(api.container.retrieval_comparator.compare(request))
    try:
        await asyncio.wait_for(started.wait(), 2)
        policy["max_chunks_per_document"] = 2
        store.replace(
            collection.collection_id,
            name="Synthetic changed",
            document_ids=["paper-d"],
            expected_revision=1,
        )
    finally:
        release.set()
    result = await asyncio.wait_for(pending, 2)
    assert resolve.call_count == 1
    assert result.collection.collection_id == collection.collection_id
    assert result.collection.revision == 1
    assert result.document_ids == ("paper-b", "paper-a")
    assert [scope for scope, _ in seen] == [("paper-b", "paper-a")] * 2
    assert seen[0][1] is None
    assert seen[1][1].max_chunks_per_document == 1
    assert {s.chunk.document_id for s in result.candidate.preview.sources} <= {"paper-a", "paper-b"}
    store.delete(collection.collection_id, expected_revision=2)


async def test_direct_selection_is_detached_and_effective_limits_are_not_invented(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.retrieval_comparison import RetrievalComparisonRequest

    api = comparison_api
    supplied = [" paper-a ", "paper-b", "paper-a"]
    request = RetrievalComparisonRequest(
        query=QUERY, document_ids=supplied, baseline={}, candidate={}
    )
    supplied[:] = ["paper-d"]
    original = api.container.runner.preview
    observed = []

    async def preview(*args: object, **kwargs: object) -> object:
        result = await original(*args, **kwargs)
        observed.append(result)
        api.container.runner._safety_limits.max_source_docs = 1
        return result

    monkeypatch.setattr(api.container.runner, "preview", preview)
    result = await api.container.retrieval_comparator.compare(request)
    assert result.document_ids == ("paper-a", "paper-b")
    assert result.baseline.preview == observed[0]
    assert result.candidate.preview == observed[1]
    assert result.baseline.preview.configuration.max_source_docs == 5
    assert result.candidate.preview.configuration.max_source_docs == 1
    assert result.changes.configuration_changed is True


def test_collection_errors_never_widen_or_start_preview(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = comparison_api
    monkeypatch.setattr(api.container.runner, "preview", denied)
    response = api.post(collection_id="col_" + "0" * 32)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "collection_not_found"
    assert_private(response)
    from storage.paper_collections import CollectionError

    for code, status in (
        ("collection_documents_missing", 409),
        ("invalid_collection_record", 409),
        ("collection_storage_error", 503),
    ):
        monkeypatch.setattr(
            api.container.paper_collections,
            "resolved_collection",
            Mock(side_effect=CollectionError(code, "Selection unavailable.", status)),
        )
        response = api.post(collection_id="col_" + "0" * 32)
        assert response.status_code == status
        assert response.json()["detail"]["code"] == code
        assert_private(response)


def test_second_preview_failure_is_sanitized_all_or_nothing(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = comparison_api.container.runner._executor
    retrieve = AsyncMock(side_effect=[[], RuntimeError("PRIVATE_SOURCE private-key")])
    monkeypatch.setattr(executor, "retrieve", retrieve)
    response = comparison_api.post()
    assert response.status_code == 500
    assert_private(response)
    assert response.json()["detail"]["code"] == "retrieval_failed"
    assert response.json()["detail"]["variant"] == "candidate"
    assert "baseline" not in response.json() and "PRIVATE_SOURCE" not in response.text
    assert "private-key" not in response.text
    assert retrieve.await_count == 2


def test_generative_or_unsupported_components_fail_without_fallback(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = comparison_api
    api.container.hybrid_retriever._hyde_expander = HyDEExpander(api.container.llm)
    response = api.post()
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "generative_retrieval"
    assert response.json()["detail"]["variant"] == "baseline"
    api.container.hybrid_retriever._hyde_expander = HyDEExpander()

    async def no_scope(plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        raise AssertionError("Unsupported scoped retrieval must not be called without scope.")

    monkeypatch.setattr(api.container.runner._executor, "retrieve", no_scope)
    response = api.post(document_ids=["paper-a"])
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "retrieval_failed"


def test_unsupported_policy_preparation_is_not_retried_without_the_policy(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.evidence import EvidenceSnapshot

    calls = []

    async def no_policy(plan: QueryPlan, retrieved: list[SearchResult]) -> EvidenceSnapshot:
        calls.append(True)
        return EvidenceSnapshot.capture(plan.observation.original_query, retrieved)

    monkeypatch.setattr(comparison_api.container.runner._executor, "prepare_context", no_policy)
    response = comparison_api.post()
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "context_preparation_failed"
    assert response.json()["detail"]["variant"] == "candidate"
    assert calls == [True]


@pytest.mark.parametrize("field", ["context", "context_sha256", "rank", "text_sha256", "scope"])
async def test_invalid_preview_provenance_is_not_a_partial_comparison(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    from agent.retrieval_comparison import RetrievalComparisonError, RetrievalComparisonRequest

    api = comparison_api
    valid = await api.container.runner.preview(QUERY)
    broken = valid.model_copy(deep=True)
    if field == "scope":
        observation = broken.plan.observation.model_copy(update={"document_ids": ("unexpected",)})
        broken = broken.model_copy(
            update={"plan": broken.plan.model_copy(update={"observation": observation})}
        )
    elif field in {"rank", "text_sha256"}:
        broken.sources[0] = broken.sources[0].model_copy(
            update={field: 2 if field == "rank" else "0" * 64}
        )
    else:
        broken = broken.model_copy(
            update={field: "PRIVATE_SOURCE" if field == "context" else "0" * 64}
        )
    preview = AsyncMock(side_effect=[valid, broken])
    monkeypatch.setattr(api.container.runner, "preview", preview)
    with pytest.raises(RetrievalComparisonError) as error:
        await api.container.retrieval_comparator.compare(
            RetrievalComparisonRequest(query=QUERY, baseline={}, candidate={})
        )
    assert error.value.code == "comparison_failed"
    assert error.value.variant == "candidate"
    assert "PRIVATE_SOURCE" not in str(error.value)


async def test_completed_preview_is_detached_before_second_await(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.retrieval_comparison import RetrievalComparisonRequest

    api = comparison_api
    actual = await api.container.runner.preview(QUERY)
    expected = actual.model_copy(deep=True)
    calls = 0

    async def reuse(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            actual.sources[0].chunk.metadata["concurrent"] = "changed after baseline"
        return actual

    monkeypatch.setattr(api.container.runner, "preview", reuse)
    result = await api.container.retrieval_comparator.compare(
        RetrievalComparisonRequest(query=QUERY, baseline={}, candidate={})
    )
    assert result.baseline.preview == expected
    assert result.source_changes.metadata_changed == 1
    assert "concurrent" not in result.baseline.preview.sources[0].chunk.metadata
    assert result.changes.context_changed is False


@pytest.mark.parametrize("stage", ["retrieve", "rerank"])
def test_scope_escapes_are_errors_not_filtered_success(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    executor = comparison_api.container.runner._executor
    target = executor if stage == "retrieve" else executor._reranker
    monkeypatch.setattr(
        target, stage, AsyncMock(return_value=[_result("private", "excluded", "PRIVATE_SOURCE")])
    )
    response = comparison_api.post(document_ids=["paper-a"])
    assert response.status_code == 500
    assert "PRIVATE_SOURCE" not in response.text
    assert "baseline" not in response.json()


async def test_invalid_python_model_is_revalidated_before_preview(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import ValidationError

    from agent.retrieval_comparison import RetrievalComparisonRequest

    monkeypatch.setattr(comparison_api.container.runner, "preview", denied)
    request = RetrievalComparisonRequest(query=QUERY, baseline={}, candidate={}).model_copy(
        update={"query": ""}
    )
    with pytest.raises(ValidationError):
        await comparison_api.container.retrieval_comparator.compare(request)


def test_overall_deadline_covers_both_previews_and_cancels_current_work(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.retrieval_comparison as module

    cancelled = []
    calls = 0

    async def slow(*args: object, **kwargs: object) -> list[SearchResult]:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.025)
            return []
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
        raise AssertionError("Comparison deadline did not cancel candidate retrieval.")

    monkeypatch.setattr(module, "COMPARISON_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(comparison_api.container.runner._executor, "retrieve", slow)
    response = comparison_api.post()
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "comparison_timeout"
    assert response.json()["detail"]["variant"] == "candidate"
    assert "baseline" not in response.json()
    assert_private(response)
    assert calls == 2 and cancelled == [True]


def test_synchronous_overrun_and_preview_phase_timeout_are_not_success(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.retrieval_comparison as module

    async def slow(*args: object, **kwargs: object) -> list[SearchResult]:
        time.sleep(0.03)  # noqa: ASYNC251
        return []

    monkeypatch.setattr(module, "COMPARISON_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(comparison_api.container.runner._executor, "retrieve", slow)
    response = comparison_api.post()
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "comparison_timeout"
    monkeypatch.setattr(module, "COMPARISON_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(
        comparison_api.container.runner._executor, "retrieve", AsyncMock(side_effect=TimeoutError)
    )
    response = comparison_api.post()
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "retrieval_timeout"


async def test_external_cancellation_propagates_after_baseline_without_partial_result(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.retrieval_comparison import RetrievalComparisonRequest

    started = asyncio.Event()
    cancelled = []
    calls = 0

    async def block(*args: object, **kwargs: object) -> list[SearchResult]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
        raise AssertionError("Cancelled comparison continued.")

    monkeypatch.setattr(comparison_api.container.runner._executor, "retrieve", block)
    pending = asyncio.create_task(
        comparison_api.container.retrieval_comparator.compare(
            RetrievalComparisonRequest(query=QUERY, baseline={}, candidate={})
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        pending.cancel("comparison cancelled")
        with pytest.raises(asyncio.CancelledError, match="comparison cancelled"):
            await pending
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    assert calls == 2 and cancelled == [True]


def test_markdown_unicode_and_special_input_are_literal(comparison_api: ComparisonAPI) -> None:
    query = 'caf\u00e9 \u7814\u7a76\n```</script><img src="https://invalid.example/x">[x](javascript:x)\n|x|'
    payload = {"query": query, "baseline": {}, "candidate": {}}
    response = comparison_api.client.post(ENDPOINT, params={"format": "markdown"}, json=payload)
    assert response.status_code == 200
    assert_private(response)
    assert response.headers["content-type"].startswith("text/markdown")
    assert (
        response.headers["content-disposition"] == 'attachment; filename="retrieval-comparison.md"'
    )
    assert f"````text\n{query}\n````" in response.text
    assert "## Baseline" in response.text and "## Candidate" in response.text
    assert len(response.content) <= 262144
    invalid = comparison_api.client.post(ENDPOINT, params={"format": "html"}, json=payload)
    assert invalid.status_code == 422
    assert_private(invalid)


def test_markdown_source_text_metadata_and_identifiers_are_literal(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    attack = '```\n# Untrusted\n<img src="https://invalid.example/track">[x](javascript:x)\n|x|'
    source = _result("id|[x](javascript:x)", "doc<script>", attack + "\ncaf\u00e9 \u7814\u7a76")
    source.chunk.title = attack
    source.chunk.source = attack
    source.chunk.metadata = {"untrusted": attack}
    monkeypatch.setattr(
        comparison_api.container.runner._executor, "retrieve", AsyncMock(return_value=[source])
    )
    payload = {"query": QUERY, "baseline": {}, "candidate": {}}
    response = comparison_api.client.post(ENDPOINT, params={"format": "markdown"}, json=payload)
    assert response.status_code == 200
    assert "````json\n" in response.text
    assert "\n# Untrusted\n" not in response.text
    assert "doc<script>" in response.text and "caf\u00e9 \u7814\u7a76" in response.text
    result = comparison_api.client.post(ENDPOINT, json=payload).json()
    retained = result["baseline"]["preview"]["sources"][0]
    assert retained["chunk"] == source.chunk.model_dump()
    assert retained["text_sha256"] == text_digest(source.chunk.text)


async def test_markdown_is_stable_after_json_roundtrip_with_unordered_metadata(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.retrieval_comparison import RetrievalComparison, RetrievalComparisonRequest

    source = _result("one", "paper-a")
    source.chunk.metadata = {"z": "last", "a": "first"}
    monkeypatch.setattr(
        comparison_api.container.runner._executor, "retrieve", AsyncMock(return_value=[source])
    )
    result = await comparison_api.container.retrieval_comparator.compare(
        RetrievalComparisonRequest(query=QUERY, baseline={}, candidate={})
    )
    reloaded = RetrievalComparison.model_validate_json(result.to_json())
    assert reloaded == result
    assert reloaded.to_markdown() == result.to_markdown()


async def test_exact_utf8_output_boundary_and_aggregate_limit(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.retrieval_comparison as module
    from agent.retrieval_comparison import RetrievalComparisonError, RetrievalComparisonRequest

    api = comparison_api
    result = await api.container.retrieval_comparator.compare(
        RetrievalComparisonRequest(query="caf\u00e9 retrieval", baseline={}, candidate={})
    )
    for serialize in (result.to_json, result.to_markdown):
        monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 262144)
        content = serialize()
        size = len(content.encode("utf-8"))
        monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", size)
        assert serialize() == content
        monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", size - 1)
        with pytest.raises(RetrievalComparisonError) as error:
            serialize()
        assert error.value.code == "comparison_too_large"
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 262144)
    large = _result("long", "paper-a", "\U0001f52c" * 20000)
    monkeypatch.setattr(api.container.runner._executor, "retrieve", AsyncMock(return_value=[large]))
    response = api.post(candidate={})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "comparison_too_large"
    assert "baseline" not in response.json() and len(response.content) < 1024
    assert_private(response)


def test_repeat_after_restart_and_useful_openapi(comparison_api: ComparisonAPI) -> None:
    api = comparison_api
    first = api.post(candidate={})
    assert first.status_code == 200
    assert api.post(candidate={}).content == first.content
    application = create_app(
        offline_settings(api.database).model_copy(update={"max_source_docs": 5, "max_hops": 1})
    )
    with TestClient(application) as client:
        response = client.post(ENDPOINT, json={"query": QUERY, "baseline": {}, "candidate": {}})
        assert response.content == first.content
    schema = api.client.get("/openapi.json").json()
    operation = schema["paths"][ENDPOINT]["post"]
    ref = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    request = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert set(request["required"]) == {"query", "baseline", "candidate"}
    assert request["additionalProperties"] is False
    assert request["properties"]["query"]["maxLength"] == 500
    assert "anyOf" not in request["properties"]["document_ids"]
    assert "anyOf" not in request["properties"]["collection_id"]
    assert {"200", "404", "409", "413", "422", "500", "503", "504"} <= set(operation["responses"])
    assert {"application/json", "text/markdown"} <= set(operation["responses"]["200"]["content"])


def test_all_fake_and_live_provider_adapters_are_unused(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
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
    settings = offline_settings(comparison_api.database).model_copy(
        update=dict.fromkeys(
            ("openai_api_key", "anthropic_api_key", "gemini_api_key", "moonshot_api_key"),
            "synthetic-unused-key",
        )
    )
    application = create_app(settings)
    with TestClient(application) as client:
        response = client.post(ENDPOINT, json={"query": QUERY, "baseline": {}, "candidate": {}})
        assert response.status_code == 200
        assert "synthetic-unused" not in response.text
    for generate in generators:
        generate.assert_not_awaited()
