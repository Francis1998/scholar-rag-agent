"""Native form -> per-paper worksheet -> exact local sources, with no generation or writes."""

import asyncio
import json
import socket
import sqlite3
from contextlib import closing
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.research_worksheet import ResearchWorksheet, WorksheetRequest
from api.application import create_app
from llm.fake import FakeLLMAdapter
from llm.router import RoutingLLMAdapter
from retrieval.models import Chunk, Document, SearchResult
from tests.test_corpus_explorer import (
    HOSTILE,
    ExplorerDOM,
    assert_headers,
    href,
    identities,
    values,
)
from tests.test_evidence_html import Node
from tests.test_research_worksheet import WorksheetAPI, denied
from tests.test_research_worksheet import worksheet_api as worksheet_api

ENDPOINT = "/explore/worksheet"
INPUT = {
    "questions": "PRIVATE_QUESTION retrieval",
    "document_ids": '["a", "b"]',
    "collection_id": "",
    "passages_per_cell": "1",
    "format": "html",
}


def controls(root: Node) -> dict[str, str]:
    form = root.by_id("worksheet-form")
    result = {node.attrs["name"]: node.attrs.get("value", "") for node in form.by_tag("input")}
    for node in form.by_tag("textarea"):
        # HTML's textarea parsing discards the first literal newline.
        result[node.attrs["name"]] = node.text().removeprefix("\n")
    result["format"] = next(
        node.attrs["value"] for node in form.by_tag("option") if "selected" in node.attrs
    )
    return result


def failure(response: httpx.Response, status: int) -> None:
    assert response.status_code == status, response.text
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    assert not root.by_class("worksheet-cell") and not root.by_tag("form")
    assert not any(secret in response.text for secret in ("PRIVATE_", "EXCLUDED_MARKER"))
    assert "content-disposition" not in response.headers


def seed(api: WorksheetAPI, identifier: str, text: str, *, count: int = 1) -> list[Chunk]:
    chunks = [
        Chunk(
            document_id=identifier,
            chunk_id=f"{identifier}/chunk-{number}",
            title="Synthetic " + HOSTILE + "t" * 300,
            source="https://invalid.example/PRIVATE_SOURCE" + "s" * 512,
            text=text,
            metadata={"chunk_index": str(number)},
        )
        for number in range(count)
    ]
    api.container.document_store.add_documents(
        [Document(**chunks[0].model_dump(exclude={"chunk_id"}))], chunks
    )
    api.container.hybrid_retriever.add_chunks(chunks)
    api.container.graph_builder.index_chunks(chunks)
    return chunks


def test_native_form_builds_a_question_by_paper_worksheet(worksheet_api: WorksheetAPI) -> None:
    api = worksheet_api
    response = api.client.get(ENDPOINT)
    assert response.status_code == 200
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    form = root.by_id("worksheet-form")
    assert form.attrs["method"] == "post"
    assert form.attrs["action"] == ENDPOINT
    assert form.attrs["enctype"] == "application/x-www-form-urlencoded"
    response = api.client.post(
        form.attrs["action"],
        data={
            "questions": "  GraphRAG retrieval?  \r\nWhat are the limitations?",
            "document_ids": json.dumps(["b", "a"]),
            "collection_id": "",
            "passages_per_cell": "1",
            "format": "html",
        },
    )
    assert response.status_code == 200, response.text
    assert_headers(response)
    worksheet = ExplorerDOM(response.text).root
    assert values(worksheet, "worksheet-question") == [
        "  GraphRAG retrieval?  ",
        "What are the limitations?",
    ]
    assert len(worksheet.by_class("worksheet-cell")) == 4
    assert "EXCLUDED_MARKER" not in worksheet.text()
    assert "not answers" in worksheet.text()
    assert "Each submission builds a new worksheet" in worksheet.text()
    assert href(worksheet, "inspect-paper-1").startswith("/explore/document?")


def test_catalog_document_prefill_native_controls_and_return(worksheet_api: WorksheetAPI) -> None:
    api = worksheet_api
    query = {"limit": 1, "source": "synthetic:worksheet", "title": "Synthetic", "cursor": "a"}
    catalog_url = "/explore?" + urlencode(query)
    catalog = ExplorerDOM(api.client.get(catalog_url).text).root
    opened = ExplorerDOM(api.client.get(href(catalog, "research-worksheet")).text).root
    assert href(opened, "back-to-catalog") == catalog_url
    document = ExplorerDOM(api.client.get(href(catalog, "inspect-document-1")).text).root
    opened = ExplorerDOM(api.client.get(href(document, "worksheet-document")).text).root
    fields = controls(opened)
    assert json.loads(fields["document_ids"]) == ["b"]
    assert fields["questions"] == fields["collection_id"] == ""
    labels = {node.attrs["for"] for node in opened.by_tag("label")}
    for node in (*opened.by_tag("input"), *opened.by_tag("select"), *opened.by_tag("textarea")):
        if node.attrs.get("type") != "hidden":
            assert node.attrs["id"] in labels
    response = api.client.post(ENDPOINT, data={**fields, "questions": "retrieval"})
    assert response.status_code == 200
    result = ExplorerDOM(response.text).root
    assert identities(result, "cell-document-id") == ["b"]
    assert href(result, "back-to-catalog") == catalog_url
    assert len(result.by_tag("table")[0].by_tag("td")) == 1
    for node in result.by_tag("a"):
        url = urlsplit(node.attrs["href"])
        assert not url.scheme and not url.netloc
        if url.fragment:
            result.by_id(url.fragment)


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_mount_prefixes_on_forms_sources_exports_and_errors(
    worksheet_api: WorksheetAPI, prefix: str, mounted: bool
) -> None:
    api = worksheet_api
    seed(api, "../paper/\u7814?&", "Synthetic retrieval.")
    app = api.application
    root_path = prefix
    if mounted:
        app = FastAPI()
        app.mount("/library", api.application)
        root_path = "/prefix"
    with TestClient(app, root_path=root_path) as client:
        opened = client.get(prefix + ENDPOINT, params={"document_id": "../paper/\u7814?&"})
        form = ExplorerDOM(opened.text).root
        response = client.post(
            form.by_id("worksheet-form").attrs["action"],
            data={**controls(form), "questions": "retrieval"},
        )
        assert response.status_code == 200
        result = ExplorerDOM(response.text).root
        context = client.get(href(result, "source-1-1-1"))
        assert context.status_code == 200
        assert identities(ExplorerDOM(context.text).root, "chunk-id") == [
            "../paper/\u7814?&/chunk-0"
        ]
        invalid = client.post(prefix + ENDPOINT, data={"questions": "PRIVATE_QUESTION"})
        failure(invalid, 422)
        for root in (form, result, ExplorerDOM(invalid.text).root):
            for anchor in root.by_tag("a"):
                path = urlsplit(anchor.attrs["href"]).path
                assert not path or path.startswith(prefix + "/")
            for native_form in root.by_tag("form"):
                assert native_form.attrs["action"].startswith(prefix + "/")
        downloaded = client.post(prefix + ENDPOINT, data={**controls(result), "format": "json"})
        assert downloaded.status_code == 200
        assert downloaded.json()["documents"][0]["document_id"] == "../paper/\u7814?&"


async def test_all_outputs_reuse_service_and_exports_and_collection_scope(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    collection = api.container.paper_collections.create(name="Synthetic", document_ids=["b", "a"])
    request = WorksheetRequest(questions=["retrieval"], collection_id=collection.collection_id)
    expected = await api.container.worksheets.build(request)
    build = AsyncMock(wraps=api.container.worksheets.build)
    monkeypatch.setattr(api.container.worksheets, "build", build)
    form = ExplorerDOM(
        api.client.get(ENDPOINT, params={"collection_id": collection.collection_id}).text
    ).root
    fields = {**controls(form), "questions": "retrieval"}
    for fmt, content_type, expected_content in (
        ("html", "text/html", None),
        ("json", "application/json", expected.to_json()),
        ("markdown", "text/markdown", expected.to_markdown()),
    ):
        response = api.client.post(ENDPOINT, data={**fields, "format": fmt})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(content_type)
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["cache-control"] == "no-store"
        assert "default-src 'none'" in response.headers["content-security-policy"]
        if expected_content is not None:
            assert response.text == expected_content
            assert response.headers["content-disposition"].startswith(
                'attachment; filename="worksheet.'
            )
        else:
            assert identities(ExplorerDOM(response.text).root, "cell-document-id") == ["b", "a"]
    assert build.await_count == 3
    assert all(call.args == (request,) for call in build.await_args_list)


@pytest.mark.parametrize(
    "updates",
    [
        {"questions": ""},
        {"questions": " "},
        {"questions": "x\n"},
        {"questions": "x\n\nx"},
        {"questions": "x\n" * 5 + "x"},
        {"questions": "x" * 501},
        {"document_ids": ""},
        {"document_ids": "[]"},
        {"document_ids": "null"},
        {"document_ids": "{}"},
        {"document_ids": '[""]'},
        {"document_ids": "[42]"},
        {"document_ids": '["a"'},  # Incomplete JSON.
        {"document_ids": json.dumps(["a"] * 11)},
        {"document_ids": json.dumps(["x" * 129])},
        {"document_ids": '["\\ud800"]'},
        {"document_ids": "[" * 1100},
        {"collection_id": "col_" + "0" * 32},
        {"document_ids": "", "collection_id": " "},
        {"passages_per_cell": "0"},
        {"passages_per_cell": "4"},
        {"passages_per_cell": "1.0"},
        {"passages_per_cell": "true"},
        {"passages_per_cell": "+1"},
        {"format": ""},
        {"format": "HTML"},
        {"unknown": "PRIVATE_VALUE"},
        {"navigation": '{"catalog_limit": 0}'},
        {"navigation": '{"catalog_limit":20,"catalog_limit":1}'},
        {"navigation": '{"return_url":"https://invalid.example/PRIVATE_REDIRECT"}'},
        {"navigation": '{"source":"\\ud800"}'},
        {"navigation": "null"},
    ],
)
def test_invalid_forms_fail_before_build_without_echo(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, updates: dict[str, str]
) -> None:
    build = AsyncMock(side_effect=denied)
    monkeypatch.setattr(worksheet_api.container.worksheets, "build", build)
    failure(worksheet_api.client.post(ENDPOINT, data={**INPUT, **updates}), 422)
    build.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        b"questions=x&questions=y",
        b"questions=x&%71uestions=y",
        b"questions=%FF",
        b"questions=%C3",
        b"questions=%",
        b"questions=%GG",
        b"questions=%00",
        b"questions=\xff",
        b"questions",
        b"questions=x&&format=html",
        b"&".join([b"x=1"] * 7),
        b"x" * 33 + b"=1",
        b"document_ids=" + b"x" * 16385,
    ],
)
def test_malformed_wire_and_duplicate_fields_are_not_repaired(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    monkeypatch.setattr(worksheet_api.container.worksheets, "build", denied)
    failure(
        worksheet_api.client.post(
            ENDPOINT, content=raw, headers={"content-type": "application/x-www-form-urlencoded"}
        ),
        422,
    )


@pytest.mark.parametrize(
    "query",
    [
        "document_id=a&document_id=b",
        "document_id=a&collection_id=col_" + "0" * 32,
        "unknown=PRIVATE_VALUE",
        "catalog_limit=1.0",
        "catalog_cursor=",
        "source=%FF",
        "title=%GG",
        "source=" + "x" * 513,
    ],
)
def test_invalid_open_form_does_not_read_corpus(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    monkeypatch.setattr(worksheet_api.container.worksheets, "build", denied)
    monkeypatch.setattr(worksheet_api.container.paper_collections, "resolve", denied)
    failure(worksheet_api.client.get(ENDPOINT + "?" + query), 422)


def test_body_query_encoding_and_method_bounds(worksheet_api: WorksheetAPI) -> None:
    api = worksheet_api
    headers = {"content-type": "application/x-www-form-urlencoded"}
    failure(api.client.post(ENDPOINT, content=b"x" * 65537, headers=headers), 413)

    def oversized_stream() -> object:
        yield b"questions=x&"
        yield b"x" * 65536

    failure(api.client.post(ENDPOINT, content=oversized_stream(), headers=headers), 413)
    failure(api.client.get(ENDPOINT + "?title=" + "x" * 16385), 422)
    failure(api.client.request("GET", ENDPOINT, content=b"questions=PRIVATE_VALUE"), 422)
    failure(api.client.post(ENDPOINT + "?format=json", data=INPUT), 422)
    failure(api.client.post(ENDPOINT, json=INPUT), 415)
    failure(api.client.post(ENDPOINT, data=INPUT, headers={"content-encoding": "gzip"}), 415)
    for method in ("PUT", "DELETE", "PATCH", "OPTIONS"):
        response = api.client.request(method, ENDPOINT)
        failure(response, 405)
        assert response.headers["allow"] == "GET, POST"
    success = api.client.post(
        ENDPOINT,
        content=urlencode(INPUT),
        headers={"content-type": 'application/x-www-form-urlencoded; charset="UTF-8"'},
    )
    assert success.status_code == 200
    schema = api.client.get("/openapi.json").json()["paths"][ENDPOINT]
    assert schema["get"]["operationId"] != schema["post"]["operationId"]


def test_unknown_empty_stale_oversized_and_unavailable_scopes(worksheet_api: WorksheetAPI) -> None:
    api = worksheet_api
    failure(
        api.client.post(ENDPOINT, data={**INPUT, "document_ids": '["a","PRIVATE_UNKNOWN"]'}), 422
    )
    failure(
        api.client.post(
            ENDPOINT, data={**INPUT, "document_ids": "", "collection_id": "col_" + "0" * 32}
        ),
        404,
    )
    collection = api.container.paper_collections.create(name="Synthetic", document_ids=["a", "b"])
    with closing(sqlite3.connect(api.database)) as connection, connection:
        connection.execute("DELETE FROM documents WHERE document_id = 'b'")
    selected = {**INPUT, "document_ids": "", "collection_id": collection.collection_id}
    failure(api.client.post(ENDPOINT, data=selected), 409)
    with closing(sqlite3.connect(api.database)) as connection, connection:
        connection.execute("UPDATE paper_collections SET document_ids = '[]'")
    failure(api.client.post(ENDPOINT, data=selected), 409)
    for number in range(11):
        api.add(f"new-{number}")
    large = api.container.paper_collections.create(
        name="Eleven", document_ids=[f"new-{number}" for number in range(11)]
    )
    failure(api.client.post(ENDPOINT, data={**selected, "collection_id": large.collection_id}), 422)
    moved = api.database.with_suffix(".unavailable")
    api.database.rename(moved)
    try:
        failure(api.client.post(ENDPOINT, data={**INPUT, "document_ids": '["a"]'}), 503)
        assert not api.database.exists()
    finally:
        moved.rename(api.database)


def test_literals_truncation_exact_source_links_and_return_controls(
    worksheet_api: WorksheetAPI,
) -> None:
    api = worksheet_api
    identifier = "../paper/\u7814?%&#\r\nid"
    text = "\nSynthetic retrieval cafe\u0301 \u7814\u7a76\r\n" + HOSTILE + " long passage" * 100
    chunks = seed(api, identifier, text)
    question = " \u7814\u7a76 retrieval " + HOSTILE
    navigation = {"source": "literal\r\nfilter", "title": "\u7814", "catalog_limit": 1}
    response = api.client.post(
        ENDPOINT,
        data={
            **INPUT,
            "questions": question,
            "document_ids": json.dumps([identifier]),
            "navigation": json.dumps(navigation),
        },
    )
    assert response.status_code == 200
    root = ExplorerDOM(response.text).root
    assert values(root, "worksheet-question") == [question]
    assert identities(root, "document-id") == [identifier]
    assert identities(root, "chunk-id") == [chunks[0].chunk_id]
    assert values(root, "passage-text") == [text[:800]]
    assert values(root, "excerpt-truncated") == ["true"]
    assert "Title truncatedtrue" in root.text() and "Source truncatedtrue" in root.text()
    assert not any(root.by_tag(tag) for tag in ("script", "iframe", "img", "svg", "base", "object"))
    assert not any(name.startswith("on") for node in root.walk() for name in node.attrs)
    source_url = href(root, "source-1-1-1")
    assert parse_qs(urlsplit(source_url).query)["chunk_id"] == [chunks[0].chunk_id]
    context = api.client.get(source_url)
    assert context.status_code == 200
    assert "context-anchor" in context.text
    assert json.loads(controls(root)["navigation"])["source"] == "literal\r\nfilter"
    assert parse_qs(urlsplit(href(root, "back-to-catalog")).query)["source"] == [
        "literal\r\nfilter"
    ]


def test_no_passages_zero_score_and_passage_omissions(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    empty = AsyncMock(return_value=[])
    monkeypatch.setattr(api.container.runner._executor, "retrieve", empty)
    response = api.client.post(ENDPOINT, data=INPUT)
    assert response.status_code == 200
    assert (
        "no_passages" in response.text
        and "not a finding of no scientific evidence" in response.text
    )
    chunks = seed(api, "zero", "retrieval", count=2)
    results = [SearchResult(chunk=chunk, score=0, retriever="synthetic") for chunk in chunks]
    monkeypatch.setattr(api.container.runner._executor, "retrieve", AsyncMock(return_value=results))
    monkeypatch.setattr(
        api.container.runner._executor._reranker, "rerank", AsyncMock(return_value=results)
    )
    response = api.client.post(ENDPOINT, data={**INPUT, "document_ids": '["zero"]'})
    assert response.status_code == 200
    root = ExplorerDOM(response.text).root
    assert values(root, "passage-score") == ["0.0"]
    assert "passages omitted: 1" in root.text()


def test_no_generation_writes_or_external_connections_and_restart(
    worksheet_api: WorksheetAPI,
) -> None:
    from contextlib import ExitStack
    from unittest.mock import patch

    api = worksheet_api
    before = api.database.read_bytes()
    events = api.container.event_log.list_events()
    with ExitStack() as stack:
        guards = [
            stack.enter_context(patch.object(target, method, side_effect=denied))
            for target, method in (
                (FakeLLMAdapter, "generate"),
                (RoutingLLMAdapter, "generate"),
                (socket, "create_connection"),
                (socket.socket, "connect"),
                (api.container.document_store, "add_documents"),
                (api.container.hybrid_retriever, "add_chunks"),
                (api.container.graph_builder, "index_chunks"),
                (api.container.ingestion_pipeline, "ingest_documents"),
                (api.container.event_log, "append_event"),
                (api.container.event_log, "append_transition"),
                (api.container.paper_collections, "create"),
                (api.container.paper_collections, "replace"),
                (api.container.paper_collections, "delete"),
            )
        ]
        first = api.client.post(ENDPOINT, data=INPUT)
        assert first.status_code == 200
        assert not any(guard.call_count for guard in guards)
    assert before == api.database.read_bytes() and events == api.container.event_log.list_events()
    restarted = create_app(
        offline_settings(api.database).model_copy(update={"max_source_docs": 2, "max_hops": 1})
    )
    with TestClient(restarted) as client:
        assert client.post(ENDPOINT, data=INPUT).content == first.content
    assert before == api.database.read_bytes()


@pytest.mark.parametrize("stage", ["retrieve", "rerank"])
def test_scope_escape_is_not_filtered_into_success(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    api = worksheet_api
    chunk = api.add("excluded", "PRIVATE_SOURCE_EXCLUDED")
    executor = api.container.runner._executor
    monkeypatch.setattr(
        executor if stage == "retrieve" else executor._reranker,
        stage,
        AsyncMock(return_value=[SearchResult(chunk=chunk, score=1, retriever="broken")]),
    )
    failure(api.client.post(ENDPOINT, data=INPUT), 500)


def test_generative_retrieval_rejected(worksheet_api: WorksheetAPI) -> None:
    from retrieval.hyde import HyDEExpander

    api = worksheet_api
    api.container.hybrid_retriever._hyde_expander = HyDEExpander(api.container.llm)
    response = api.client.post(ENDPOINT, data=INPUT)
    failure(response, 409)
    assert "Generative retrieval" in response.text


def test_timeout_serialization_and_html_overflow_are_all_or_nothing(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.research_worksheet as worksheet_module
    import api.explorer as explorer

    api = worksheet_api
    baseline = api.client.post(ENDPOINT, data=INPUT)
    assert baseline.status_code == 200
    size = len(baseline.content)
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", size)
    assert api.client.post(ENDPOINT, data=INPUT).content == baseline.content
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", size - 1)
    failure(api.client.post(ENDPOINT, data=INPUT), 413)
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", 1_048_576)
    monkeypatch.setattr(worksheet_module, "MAX_RESPONSE_BYTES", 100)
    failure(api.client.post(ENDPOINT, data=INPUT), 413)
    monkeypatch.setattr(worksheet_module, "MAX_RESPONSE_BYTES", 262144)
    monkeypatch.setattr(worksheet_module, "WORKSHEET_TIMEOUT_SECONDS", 0.01)
    retrieve = AsyncMock(side_effect=lambda *args, **kwargs: None)

    async def slow(*args: object, **kwargs: object) -> list[SearchResult]:
        await asyncio.sleep(10)
        return []

    retrieve.side_effect = slow
    monkeypatch.setattr(api.container.runner._executor, "retrieve", retrieve)
    failure(api.client.post(ENDPOINT, data=INPUT), 504)
    assert retrieve.await_count == 1


def test_second_cell_failure_does_not_expose_first_cell_or_exception(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    preview = asyncio.run(api.container.runner.preview("retrieval", document_ids=("a",)))
    monkeypatch.setattr(
        api.container.runner,
        "preview",
        AsyncMock(side_effect=[preview, RuntimeError("PRIVATE_DIAGNOSTIC")]),
    )
    failure(api.client.post(ENDPOINT, data={**INPUT, "questions": "retrieval"}), 500)


def test_unrepresentable_corpus_text_is_explicit_not_replaced(worksheet_api: WorksheetAPI) -> None:
    api = worksheet_api
    seed(api, "nul", "retrieval PRIVATE_TEXT\0not-representable")
    selected = {**INPUT, "document_ids": '["nul"]'}
    failure(api.client.post(ENDPOINT, data=selected), 409)
    exported = api.client.post(ENDPOINT, data={**selected, "format": "json"})
    assert exported.status_code == 200
    assert (
        "\0"
        in ResearchWorksheet.model_validate_json(exported.content)
        .rows[0]
        .cells[0]
        .passages[0]
        .excerpt
    )
