"""Explicit provider blocks must never become completed research answers."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.evidence import EvidenceSnapshot
from api.application import create_app
from config import Settings
from llm.fake import FakeLLMAdapter
from llm.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    HTTPProviderAdapter,
    KimiAdapter,
    OpenAIAdapter,
    ProviderResponseError,
)
from llm.router import ModelRouter, RoutingLLMAdapter
from llm.schemas import LLMRequest, TaskType
from retrieval.models import Chunk, Document
from storage.event_log import SQLiteEventLog

PARTIAL = "  Synthetic private partial answer [c1].\n"
REFUSAL = "Synthetic private refusal text."
THINKING = "synthetic-private-hidden-thinking"
DETAILS = "synthetic-private-filter-details"
DUMMY_KEY = "offline-dummy-key"
MODEL = "configured-provider-model"
MISSING = object()
PROVIDERS = (OpenAIAdapter, KimiAdapter, AnthropicAdapter, GeminiAdapter)
REQUEST = LLMRequest(
    task_type=TaskType.DEFAULT,
    prompt="What does GraphRAG connect?",
    context="[c1] Synthetic evidence: GraphRAG connects research entities.",
    citation_chunk_ids=["c1"],
)


def chat_body(*, reason: object = MISSING, **message: object) -> dict[str, object]:
    choice: dict[str, object] = {
        "message": {"content": PARTIAL, "reasoning_content": THINKING, **message}
    }
    if reason is not MISSING:
        choice["finish_reason"] = reason
    return {"model": "server-alias", "choices": [choice]}


def claude_body(*, reason: object = MISSING, **fields: object) -> dict[str, object]:
    body: dict[str, object] = {
        "content": [
            {"type": "thinking", "thinking": THINKING},
            {"type": "text", "text": PARTIAL},
        ],
        "stop_details": DETAILS,
        **fields,
    }
    if reason is not MISSING:
        body["stop_reason"] = reason
    return body


def gemini_body(
    *, reason: object = MISSING, prompt_feedback: object = MISSING, **fields: object
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "content": {"parts": [{"thought": True, "text": THINKING}, {"text": PARTIAL}]},
        "finishMessage": DETAILS,
        **fields,
    }
    if reason is not MISSING:
        candidate["finishReason"] = reason
    body: dict[str, object] = {"modelVersion": "server-alias", "candidates": [candidate]}
    if prompt_feedback is not MISSING:
        body["promptFeedback"] = prompt_feedback
    return body


GEMINI_BLOCKED_REASONS = (
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
    "ESCALATION",
    "PUP_LIMITED_DISABLED",
)
BLOCKED_CASES = [
    *[
        pytest.param(provider, chat_body(reason="content_filter"), id=provider.provider_name)
        for provider in (OpenAIAdapter, KimiAdapter)
    ],
    pytest.param(AnthropicAdapter, claude_body(reason="refusal"), id="anthropic-refusal"),
    *[
        pytest.param(GeminiAdapter, gemini_body(reason=reason), id=f"gemini-{reason}")
        for reason in GEMINI_BLOCKED_REASONS
    ],
    *[
        pytest.param(
            provider,
            chat_body(reason=reason, refusal=refusal),
            id=f"{provider.provider_name}-refusal-string-{reason_index}-{refusal_index}",
        )
        for provider in (OpenAIAdapter, KimiAdapter)
        for reason_index, reason in enumerate((MISSING, None, "stop", "unknown-finish"))
        for refusal_index, refusal in enumerate((REFUSAL, " \n"))
    ],
    *[
        pytest.param(
            provider,
            chat_body(
                reason=reason,
                content=[
                    {"type": "reasoning", "text": THINKING},
                    {"type": "text", "text": PARTIAL},
                    {"type": "refusal", **fields},
                    {"type": "text", "text": "Synthetic trailing text."},
                ],
            ),
            id=f"{provider.provider_name}-refusal-part-{reason_index}-{fields_index}",
        )
        for provider in (OpenAIAdapter, KimiAdapter)
        for reason_index, reason in enumerate((MISSING, "stop"))
        for fields_index, fields in enumerate(
            ({}, {"refusal": ""}, {"refusal": None}, {"refusal": REFUSAL, "text": THINKING})
        )
    ],
    *[
        pytest.param(
            GeminiAdapter,
            gemini_body(reason="STOP", prompt_feedback={"blockReason": reason}),
            id=f"gemini-prompt-{reason}",
        )
        for reason in (
            "SAFETY",
            "OTHER",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "IMAGE_SAFETY",
            DETAILS,
        )
    ],
    *[
        pytest.param(
            GeminiAdapter,
            {"promptFeedback": {"blockReason": reason}, "candidates": []},
            id=f"gemini-prompt-no-candidates-{reason}",
        )
        for reason in ("SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY")
    ],
    pytest.param(
        GeminiAdapter,
        gemini_body(
            reason="STOP",
            safetyRatings=[None, {"blocked": False}, {"blocked": True, "category": DETAILS}],
        ),
        id="gemini-candidate-blocked-rating",
    ),
    pytest.param(
        GeminiAdapter,
        gemini_body(
            prompt_feedback={
                "blockReason": "BLOCK_REASON_UNSPECIFIED",
                "safetyRatings": [{"blocked": True, "category": DETAILS}],
            }
        ),
        id="gemini-prompt-blocked-rating",
    ),
]


def blocked_error(provider: type[HTTPProviderAdapter]) -> str:
    return (
        f"{provider.provider_name} returned an invalid response: response was blocked or refused."
    )


@pytest.fixture
def provider_http(
    body: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> tuple[list[httpx.Request], AsyncMock]:
    original_client = httpx.AsyncClient
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=body, headers={"x-private-detail": DETAILS})

    def create_client(*, timeout: float) -> httpx.AsyncClient:
        assert timeout == 60.0
        return original_client(
            timeout=timeout, transport=httpx.MockTransport(respond), trust_env=False
        )

    sleep = AsyncMock()
    monkeypatch.setattr("llm.providers.httpx.AsyncClient", create_client)
    monkeypatch.setattr("llm.rate_limit.asyncio.sleep", sleep)
    return requests, sleep


@pytest.mark.parametrize(("provider", "body"), BLOCKED_CASES)
@pytest.mark.parametrize("visible_text", [True, False], ids=["with-text", "without-text"])
def test_blocked_signal_is_explicit_even_without_text_and_does_not_mutate_input(
    provider: type[HTTPProviderAdapter], body: dict[str, object], visible_text: bool
) -> None:
    selected = deepcopy(body)
    if not visible_text:
        if provider is AnthropicAdapter:
            selected["content"] = []
        elif provider is GeminiAdapter:
            candidates = selected["candidates"]
            assert isinstance(candidates, list)
            if candidates:
                candidates[0]["content"] = None
        else:
            choices = selected["choices"]
            assert isinstance(choices, list)
            message = choices[0]["message"]
            content = message["content"]
            message["content"] = (
                [part for part in content if part.get("type") == "refusal"]
                if isinstance(content, list)
                else None
            )
    original_body = deepcopy(selected)
    request = REQUEST.model_copy(deep=True)
    with pytest.raises(ProviderResponseError) as caught:
        provider(api_key=DUMMY_KEY, model=MODEL).parse_response(selected, request)
    assert str(caught.value) == blocked_error(provider)
    assert selected == original_body
    assert request == REQUEST


@pytest.mark.parametrize(("provider", "body"), BLOCKED_CASES)
async def test_blocked_response_has_one_admitted_attempt_and_no_retry_or_fallback(
    provider: type[HTTPProviderAdapter],
    provider_http: tuple[list[httpx.Request], AsyncMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, sleep = provider_http
    adapter = provider(api_key=DUMMY_KEY, model=MODEL)
    fallback = AsyncMock(side_effect=AssertionError("Unexpected provider fallback."))
    alternatives = [
        FakeLLMAdapter(),
        *(p(api_key=DUMMY_KEY) for p in PROVIDERS if p is not provider),
    ]
    for alternative in alternatives:
        monkeypatch.setattr(alternative, "generate", fallback)
    router = RoutingLLMAdapter(
        ModelRouter(
            adapters={
                **{alternative.provider_name: alternative for alternative in alternatives},
                provider.provider_name: adapter,
            },
            default_provider=provider.provider_name,
        )
    )
    with pytest.raises(ProviderResponseError) as caught:
        await router.generate(REQUEST)
    assert str(caught.value) == blocked_error(provider)
    for private in (PARTIAL.strip(), REFUSAL, THINKING, DETAILS, DUMMY_KEY, "server-alias"):
        assert private not in str(caught.value)
    assert len(requests) == 1
    assert requests[0].url == adapter.endpoint
    assert json.loads(requests[0].content) == adapter.payload(REQUEST)
    assert len(adapter._limiter._timestamps) == 1
    fallback.assert_not_awaited()
    sleep.assert_not_awaited()


@pytest.mark.parametrize(
    ("provider", "body"),
    [
        case
        for case in BLOCKED_CASES
        if case.values[0] is not AnthropicAdapter and "promptFeedback" not in case.values[1]
    ],
)
@pytest.mark.parametrize("blocked_first", [True, False], ids=["first", "later"])
def test_only_first_choice_or_candidate_controls_rejection(
    provider: type[HTTPProviderAdapter], body: dict[str, object], blocked_first: bool
) -> None:
    key = "candidates" if provider is GeminiAdapter else "choices"
    good = gemini_body() if provider is GeminiAdapter else chat_body()
    bad_candidates, good_candidates = body[key], good[key]
    assert isinstance(bad_candidates, list)
    assert isinstance(good_candidates, list)
    selected = {
        key: deepcopy(
            bad_candidates + good_candidates if blocked_first else good_candidates + bad_candidates
        )
    }
    adapter = provider(api_key=DUMMY_KEY, model=MODEL)
    if blocked_first:
        with pytest.raises(ProviderResponseError, match="response was blocked or refused"):
            adapter.parse_response(selected, REQUEST)
    else:
        response = adapter.parse_response(selected, REQUEST)
        assert response.text == PARTIAL
        assert response.model_name == MODEL
        assert response.raw_provider == provider.provider_name
        assert response.citation_chunk_ids == ["c1"]


@pytest.mark.parametrize(("provider", "body"), BLOCKED_CASES)
def test_blocked_query_preserves_input_but_never_grounds_or_exports_an_answer(
    provider: type[HTTPProviderAdapter],
    provider_http: tuple[list[httpx.Request], AsyncMock],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    credentials = {
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
        "MOONSHOT_API_KEY": "",
        "SEMANTIC_SCHOLAR_API_KEY": "",
    }
    key_prefix = "MOONSHOT" if provider is KimiAdapter else provider.provider_name.upper()
    credentials[f"{key_prefix}_API_KEY"] = DUMMY_KEY
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "blocked.sqlite3",
        default_model=provider.provider_name,
        **credentials,
        **{f"{provider.provider_name}_model": MODEL},
    )
    app = create_app(settings)
    container = app.state.container
    grounder = container.runner._executor._grounder
    ground = Mock(wraps=grounder.ground)
    monkeypatch.setattr(grounder, "ground", ground)
    chunk = Chunk(
        chunk_id="c1",
        document_id="synthetic-document",
        title="Synthetic evidence",
        text="GraphRAG connects research entities.",
        source="offline-fixture",
    )
    container.document_store.add_documents(
        [Document(**chunk.model_dump(exclude={"chunk_id"}))], [chunk]
    )
    container.hybrid_retriever.add_chunks([chunk])
    requests, sleep = provider_http
    with TestClient(app) as client:
        response = client.post("/query", json={"query": REQUEST.prompt})
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["state"] == "ERROR"
        assert result["answer"] is None
        assert result["error"] == blocked_error(provider)
        ground.assert_not_called()
        events = SQLiteEventLog(settings.database_path).list_events(result["run_id"])
        assert [event["event_type"] for event in events] == [
            "state_transition",
            "decision_log",
            "state_transition",
            "state_transition",
            "evidence_snapshot",
            "state_transition",
        ]
        snapshot = EvidenceSnapshot.model_validate(events[-2]["payload"])
        assert snapshot.request == REQUEST.model_copy(update={"task_type": TaskType.REASONING})
        assert [source.chunk for source in snapshot.sources] == [chunk]
        adapter = provider(api_key=DUMMY_KEY, model=MODEL)
        assert len(requests) == 1
        assert requests[0].url == adapter.endpoint
        assert json.loads(requests[0].content) == adapter.payload(snapshot.request)
        assert events[-1]["payload"] == {
            "from_state": "REASONING",
            "to_state": "ERROR",
            "payload": {"error": blocked_error(provider)},
        }
        assert not any(event["payload"].get("to_state") == "DONE" for event in events)
        for private in (PARTIAL.strip(), REFUSAL, THINKING, DETAILS, DUMMY_KEY, "server-alias"):
            assert private not in response.text
            assert private not in json.dumps(events)
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/runs/{result['run_id']}/events").json() == events
        for export_format in ("json", "markdown", "html"):
            assert (
                restarted.get(
                    f"/runs/{result['run_id']}/export", params={"format": export_format}
                ).status_code
                == 409
            )
    assert len(requests) == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda provider: provider.provider_name)
@pytest.mark.parametrize(
    "reason", [MISSING, None, "", "stop", "end_turn", "STOP", "unknown", [], {}]
)
def test_nonblocked_text_keeps_whitespace_multipart_citations_and_provenance(
    provider: type[HTTPProviderAdapter], reason: object
) -> None:
    first, last = "  I cannot answer ", "from these synthetic notes [c1].\n"
    if provider is AnthropicAdapter:
        body = claude_body(
            reason=reason,
            content=[
                {"type": "thinking", "text": THINKING},
                {"type": "text", "text": first},
                {"type": "redacted_thinking", "data": THINKING},
                {"type": "text", "text": last},
            ],
        )
    elif provider is GeminiAdapter:
        body = gemini_body(
            reason=reason,
            content={
                "parts": [
                    {"thought": True, "text": THINKING},
                    {"text": first},
                    {"inlineData": {"mimeType": "image/png", "data": ""}},
                    {"text": last},
                ]
            },
        )
    else:
        body = chat_body(
            reason=reason,
            refusal=None,
            content=[
                {"type": "reasoning", "text": THINKING},
                {"type": "text", "text": first, "refusal": REFUSAL},
                {"text": last},
                {"type": [], "text": THINKING},
                None,
            ],
        )
    response = provider(api_key=DUMMY_KEY, model=MODEL).parse_response(body, REQUEST)
    assert response.text == first + last
    assert response.citation_chunk_ids == ["c1"]
    assert response.model_name == MODEL
    assert response.raw_provider == provider.provider_name


@pytest.mark.parametrize("provider", [OpenAIAdapter, KimiAdapter])
@pytest.mark.parametrize("refusal", [None, "", False, 1, [], {}, [REFUSAL], {"text": REFUSAL}])
def test_empty_or_nonstring_refusal_is_not_an_explicit_refusal(
    provider: type[HTTPProviderAdapter], refusal: object
) -> None:
    response = provider(api_key=DUMMY_KEY).parse_response(chat_body(refusal=refusal), REQUEST)
    assert response.text == PARTIAL


@pytest.mark.parametrize(
    "feedback",
    [
        MISSING,
        None,
        [],
        DETAILS,
        {},
        *[
            {"blockReason": reason}
            for reason in (None, "", " \n", "BLOCK_REASON_UNSPECIFIED", [], {}, False, 1)
        ],
    ],
)
def test_gemini_missing_empty_or_unspecified_prompt_reason_is_not_a_block(feedback: object) -> None:
    response = GeminiAdapter(api_key=DUMMY_KEY).parse_response(
        gemini_body(prompt_feedback=feedback), REQUEST
    )
    assert response.text == PARTIAL


@pytest.mark.parametrize(
    "ratings",
    [
        None,
        [],
        {},
        DETAILS,
        [None, {}, "SAFETY"],
        *[
            [{"blocked": blocked, "probability": "HIGH"}]
            for blocked in (False, None, "", "true", 1, [], {})
        ],
    ],
)
@pytest.mark.parametrize("scope", ["prompt", "candidate"])
def test_gemini_only_boolean_true_rating_is_an_explicit_block(ratings: object, scope: str) -> None:
    body = (
        gemini_body(prompt_feedback={"safetyRatings": ratings})
        if scope == "prompt"
        else gemini_body(safetyRatings=ratings)
    )
    assert GeminiAdapter(api_key=DUMMY_KEY).parse_response(body, REQUEST).text == PARTIAL
