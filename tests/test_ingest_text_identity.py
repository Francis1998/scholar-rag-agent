"""API document identities preserve exact field boundaries and persisted ownership."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import NoReturn

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import IngestResponse
from ingestion.chunking import TextChunker, stable_id
from retrieval.dense import DenseRetriever
from retrieval.models import Chunk, Document
from retrieval.sparse import BM25Retriever
from storage.document_catalog import DocumentCatalogPage
from storage.document_chunks import DocumentChunksPage

SHARED_TEXT = "shared synthetic content"
GRAPH_TEXT = "GraphRAG connects Alpha and Beta."


def deny_network(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Ingestion identity tests must not use the network.")


@pytest.fixture(autouse=True)
def offline_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_network)


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "identity.sqlite3"


@pytest.fixture
def identity_client(database_path: Path) -> Iterator[TestClient]:
    application = create_app(offline_settings(database_path))
    with TestClient(application) as client:
        yield client


def ingest(client: TestClient, payload: dict[str, str]) -> IngestResponse:
    response = client.post("/ingest/text", json=payload)
    assert response.status_code == 200, response.text
    assert set(response.json()) == {"document_id", "chunk_ids"}
    return IngestResponse.model_validate_json(response.content)


def assert_persisted_papers(
    client: TestClient,
    database_path: Path,
    papers: list[tuple[dict[str, str], IngestResponse]],
) -> None:
    documents = [
        Document(document_id=result.document_id, **payload, metadata={"source_type": "api"})
        for payload, result in papers
    ]
    expected_chunks = [chunk for document in documents for chunk in TextChunker().chunk(document)]
    with closing(sqlite3.connect(database_path)) as connection:
        assert connection.execute(
            "SELECT document_id, source, title, text, metadata FROM documents ORDER BY document_id"
        ).fetchall() == sorted(
            (doc.document_id, doc.source, doc.title, doc.text, '{"source_type": "api"}')
            for doc in documents
        )
        for query in (
            "SELECT chunk_id, document_id, title, text, source, metadata FROM chunks "
            "ORDER BY chunk_id",
            "SELECT chunk_id, document_id, title, text, source, metadata FROM graph_chunks "
            "ORDER BY chunk_id",
        ):
            assert connection.execute(query).fetchall() == sorted(
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.title,
                    chunk.text,
                    chunk.source,
                    json.dumps(chunk.metadata, sort_keys=True),
                )
                for chunk in expected_chunks
            )

    response = client.get("/documents")
    assert response.status_code == 200, response.text
    catalog = DocumentCatalogPage.model_validate_json(response.content)
    assert catalog.next_cursor is None
    assert [
        (row.document_id, row.source, row.title, row.chunk_count) for row in catalog.documents
    ] == sorted(
        (result.document_id, payload["source"], payload["title"], len(result.chunk_ids))
        for payload, result in papers
    )
    for _, result in papers:
        expected = [chunk for chunk in expected_chunks if chunk.document_id == result.document_id]
        assert result.chunk_ids == [chunk.chunk_id for chunk in expected]
        response = client.get(f"/documents/{result.document_id}/chunks")
        assert response.status_code == 200, response.text
        page = DocumentChunksPage.model_validate_json(response.content)
        assert page.document_id == result.document_id
        assert page.next_cursor is None
        assert [
            (row.chunk_id, row.document_id, row.title, row.source, row.text, row.chunk_index)
            for row in page.chunks
        ] == sorted(
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.title,
                chunk.source,
                chunk.text,
                int(chunk.metadata["chunk_index"]),
            )
            for chunk in expected
        )
        assert all(
            not row.text_truncated and not row.title_truncated and not row.source_truncated
            for row in page.chunks
        )
        preview = client.post(
            "/retrieve",
            json={"query": "What does GraphRAG connect?", "document_ids": [result.document_id]},
        )
        assert preview.status_code == 200, preview.text
        retrieved = [Chunk.model_validate(source["chunk"]) for source in preview.json()["sources"]]
        assert sorted(retrieved, key=lambda chunk: chunk.chunk_id) == sorted(
            expected, key=lambda chunk: chunk.chunk_id
        )
    preview = client.post("/retrieve", json={"query": "What does GraphRAG connect?"})
    assert preview.status_code == 200, preview.text
    assert {source["chunk"]["document_id"] for source in preview.json()["sources"]} == {
        document.document_id for document in documents
    }


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            ("synthetic:a", "b", SHARED_TEXT),
            ("synthetic", "a:b", SHARED_TEXT),
            id="source-title",
        ),
        pytest.param(
            ("synthetic", "a:b", SHARED_TEXT),
            ("synthetic", "a", f"b:{SHARED_TEXT}"),
            id="title-text",
        ),
        pytest.param(
            ("synthetic:a", "b:c", f"d:{SHARED_TEXT}"),
            ("synthetic", "a:b", f"c:d:{SHARED_TEXT}"),
            id="all-fields",
        ),
        pytest.param(
            ("synthetic:", "", SHARED_TEXT),
            ("synthetic", ":", SHARED_TEXT),
            id="empty-title",
        ),
        pytest.param(
            (":synthetic", "note", SHARED_TEXT),
            ("", "synthetic:note", SHARED_TEXT),
            id="empty-source",
        ),
        pytest.param(
            ("synthetic:\u7814\u7a76", "caf\u00e9", f"\U0001f52c:{SHARED_TEXT}"),
            ("synthetic", "\u7814\u7a76:caf\u00e9", f"\U0001f52c:{SHARED_TEXT}"),
            id="unicode",
        ),
        pytest.param(
            ('synthetic:["a",\n\\', '"b"]', SHARED_TEXT),
            ("synthetic", '["a",\n\\:"b"]', SHARED_TEXT),
            id="json-punctuation",
        ),
    ],
)
def test_separator_boundaries_do_not_alias_documents(
    identity_client: TestClient, first: tuple[str, str, str], second: tuple[str, str, str]
) -> None:
    assert first != second
    assert ":".join(first) == ":".join(second)
    ingested = []
    for source, title, text in (first, second):
        response = identity_client.post(
            "/ingest/text", json={"source": source, "title": title, "text": text}
        )
        assert response.status_code == 200, response.text
        ingested.append(IngestResponse.model_validate_json(response.content))

    assert ingested[0].document_id != ingested[1].document_id, (
        "Two distinct valid API ingestions returned the same document identity: "
        f"{ingested[0].model_dump()} == {ingested[1].model_dump()}"
    )
    assert set(ingested[0].chunk_ids).isdisjoint(ingested[1].chunk_ids)


@pytest.mark.parametrize("field", ["source", "title", "text"])
@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param("Note", "note", id="case"),
        pytest.param(" Note\t ", "Note", id="whitespace"),
        pytest.param("caf\u00e9", "cafe\u0301", id="unicode-composition"),
        pytest.param("\U0001f52c", "\U0001f9ec", id="supplementary-unicode"),
        pytest.param("\u00e9", r"\u00e9", id="literal-unicode-escape"),
        pytest.param("Alpha\nBeta", r"Alpha\nBeta", id="literal-newline-escape"),
        pytest.param("Alpha:Beta", "Alpha::Beta", id="colon-count"),
    ],
)
def test_exact_field_values_are_not_normalized_for_identity(
    identity_client: TestClient, database_path: Path, field: str, first: str, second: str
) -> None:
    original = {"source": "synthetic", "title": "Note", "text": GRAPH_TEXT, field: first}
    changed = {**original, field: second}
    left = ingest(identity_client, original)
    right = ingest(identity_client, changed)
    assert left.document_id != right.document_id
    assert set(left.chunk_ids).isdisjoint(right.chunk_ids)
    with closing(sqlite3.connect(database_path)) as connection:
        assert connection.execute(
            "SELECT document_id, source, title, text FROM documents ORDER BY document_id"
        ).fetchall() == sorted(
            (result.document_id, payload["source"], payload["title"], payload["text"])
            for payload, result in ((original, left), (changed, right))
        )


def test_identity_is_stable_across_json_encodings_order_and_default_source(
    identity_client: TestClient,
) -> None:
    payload = {
        "title": '\u7814\u7a76:"quoted"\\paper\n',
        "text": "\tGraphRAG  connects\nAlpha:Beta. caf\u00e9 \U0001f52c\u2003",
    }
    expected = IngestResponse(
        document_id="doc-api-v2-88ef8120ab15fa70", chunk_ids=["chunk-4c33f963133cf29c"]
    )
    assert ingest(identity_client, payload) == expected
    explicit_source = {"source": "api", **dict(reversed(payload.items()))}
    for ascii_escaped in (True, False):
        response = identity_client.post(
            "/ingest/text",
            content=json.dumps(explicit_source, ensure_ascii=ascii_escaped, indent=2).encode(
                "utf-8"
            ),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200, response.text
        assert response.json() == expected.model_dump()


def test_both_papers_retain_chunks_retrieval_and_catalog_after_retries_and_restart(
    database_path: Path,
) -> None:
    settings = offline_settings(database_path)
    text = "\tGraphRAG  connects\nAlpha and Beta. " * 35 + "caf\u00e9 \u7814\u7a76 \U0001f52c\u2003"
    payloads = [
        {"source": "synthetic:a", "title": "b", "text": text},
        {"source": "synthetic", "title": "a:b", "text": text},
    ]
    with TestClient(create_app(settings)) as client:
        papers = [(payload, ingest(client, payload)) for payload in payloads]
        assert len({result.document_id for _, result in papers}) == 2
        assert all(len(result.chunk_ids) > 1 for _, result in papers)
        for payload, result in reversed(papers):
            assert ingest(client, payload) == result
        assert_persisted_papers(client, database_path, papers)

    with TestClient(create_app(settings)) as restarted:
        assert_persisted_papers(restarted, database_path, papers)
        for payload, result in papers:
            assert ingest(restarted, payload) == result
        assert_persisted_papers(restarted, database_path, papers)


def test_preview_ranks_and_context_survive_restart_with_new_identities(
    database_path: Path,
) -> None:
    settings = offline_settings(database_path)
    with TestClient(create_app(settings)) as client:
        papers = [
            ingest(
                client,
                {
                    "source": "synthetic:collections",
                    "title": f"Synthetic {title}",
                    "text": f"Synthetic data, not a publication. Local retrieval {marker}.",
                },
            )
            for title, marker in (
                ("methods", "METHODS_MARKER"),
                ("comparison", "COMPARISON_MARKER"),
                ("background", "EXCLUDED_MARKER"),
            )
        ]
        query = {"query": "retrieval", "document_ids": [paper.document_id for paper in papers[:2]]}
        before = client.post("/retrieve", json=query)
        assert before.status_code == 200, before.text
    with TestClient(create_app(settings)) as restarted:
        after = restarted.post("/retrieve", json=query)
        assert after.status_code == 200, after.text
        assert after.json() == before.json()


@pytest.mark.parametrize("scoped", [False, True])
def test_replayed_ingestion_preserves_preview_before_and_after_restart(
    database_path: Path, scoped: bool
) -> None:
    settings = offline_settings(database_path).model_copy(
        update={"max_source_docs": 2, "max_hops": 1}
    )
    payloads = [
        {"source": "synthetic:replay", "title": title, "text": "retrieval"}
        for title in ("First paper", "Second paper")
    ]
    with TestClient(create_app(settings)) as client:
        papers = [ingest(client, payload) for payload in payloads]
        document_ids = [paper.document_id for paper in papers]
        scope = {"document_ids": document_ids} if scoped else {}
        query = {"query": "retrieval", **scope}
        before = client.post("/retrieve", json=query)
        assert before.status_code == 200, before.text
        assert len(before.json()["sources"]) == 2
        assert {source["chunk"]["document_id"] for source in before.json()["sources"]} == set(
            document_ids
        )
        first_id = before.json()["sources"][0]["chunk"]["document_id"]
        first_index = document_ids.index(first_id)
        for _ in range(3):
            assert ingest(client, payloads[first_index]) == papers[first_index]
        replayed = client.post("/retrieve", json=query)
        assert replayed.status_code == 200, replayed.text
        assert len(client.get("/documents").json()["documents"]) == 2

    with TestClient(create_app(settings)) as restarted:
        after_restart = restarted.post("/retrieve", json=query)
        assert after_restart.status_code == 200, after_restart.text
        assert ingest(restarted, payloads[first_index]) == papers[first_index]
        replayed_after_restart = restarted.post("/retrieve", json=query)
        assert replayed_after_restart.status_code == 200, replayed_after_restart.text

    assert after_restart.json() == before.json()
    assert replayed.json() == before.json()
    assert replayed_after_restart.json() == before.json()


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
async def test_tied_chunk_ranks_are_insertion_independent_before_scope_and_limit(
    retriever_type: type[DenseRetriever] | type[BM25Retriever],
) -> None:
    chunks = [
        Chunk(
            chunk_id=identifier,
            document_id="excluded" if identifier == "chunk-a" else "selected",
            title="Synthetic note",
            text="unrelated" if identifier == "chunk-0" else "retrieval",
            source="synthetic",
        )
        for identifier in ("chunk-c", "chunk-b", "chunk-a", "chunk-0")
    ]
    for insertion_order in (chunks, list(reversed(chunks))):
        retriever = retriever_type()
        retriever.add_chunks(insertion_order)
        results = await retriever.retrieve("retrieval", limit=2)
        assert [result.chunk.chunk_id for result in results] == ["chunk-a", "chunk-b"]
        assert results[0].score == results[1].score
        scoped = await retriever.retrieve("retrieval", limit=1, document_ids=["selected"])
        assert [result.chunk.chunk_id for result in scoped] == ["chunk-b"]
        all_selected = await retriever.retrieve("retrieval", document_ids=["selected"])
        assert [result.chunk.chunk_id for result in all_selected] == [
            "chunk-b",
            "chunk-c",
            "chunk-0",
        ]
        assert all_selected[-1].score == 0


@pytest.mark.parametrize(
    ("legacy_source", "legacy_title"),
    [
        pytest.param("synthetic:a", "b:c", id="same-payload"),
        pytest.param("synthetic", "a:b:c", id="ambiguous-legacy-payload"),
    ],
)
def test_legacy_documents_collections_and_saved_runs_are_not_remapped(
    database_path: Path, legacy_source: str, legacy_title: str
) -> None:
    settings = offline_settings(database_path)
    payload = {"source": "synthetic:a", "title": "b:c", "text": GRAPH_TEXT}
    legacy_payload = {"source": legacy_source, "title": legacy_title, "text": GRAPH_TEXT}
    legacy_id = stable_id(f"{legacy_source}:{legacy_title}:{GRAPH_TEXT}", "doc")
    legacy = Document(document_id=legacy_id, **legacy_payload, metadata={"source_type": "api"})
    application = create_app(settings)
    container: AppContainer = application.state.container
    chunks = container.ingestion_pipeline.ingest_documents([legacy])
    legacy_result = IngestResponse(
        document_id=legacy_id, chunk_ids=[chunk.chunk_id for chunk in chunks]
    )
    with TestClient(application) as client:
        response = client.post(
            "/collections", json={"name": "Legacy paper", "document_ids": [legacy_id]}
        )
        assert response.status_code == 201, response.text
        collection = response.json()
        collection_path = f"/collections/{collection['collection_id']}"
        query = {
            "query": "What does GraphRAG connect?",
            "collection_id": collection["collection_id"],
        }
        response = client.post("/query", json=query)
        assert response.status_code == 200, response.text
        run = response.json()["result"]
        assert run["state"] == "DONE", run
        assert {citation["document_id"] for citation in run["answer"]["citations"]} == {legacy_id}
        export_path = f"/runs/{run['run_id']}/export"
        exports = {}
        for fmt in ("json", "markdown"):
            response = client.get(export_path, params={"format": fmt})
            assert response.status_code == 200, response.text
            exports[fmt] = response.content
        events = container.event_log.list_events()

    previous: IngestResponse | None = None
    for _ in range(2):
        restarted = create_app(settings)
        restarted_container: AppContainer = restarted.state.container
        with TestClient(restarted) as client:
            current = ingest(client, payload)
            assert current.document_id.startswith("doc-api-v2-")
            assert current.document_id != legacy_id
            assert set(current.chunk_ids).isdisjoint(legacy_result.chunk_ids)
            if previous is not None:
                assert current == previous
            assert ingest(client, payload) == current
            previous = current
            assert_persisted_papers(
                client, database_path, [(legacy_payload, legacy_result), (payload, current)]
            )
            assert client.get(collection_path).json() == collection
            preview = client.post("/retrieve", json=query)
            assert preview.status_code == 200, preview.text
            assert {source["chunk"]["document_id"] for source in preview.json()["sources"]} == {
                legacy_id
            }
            for fmt, content in exports.items():
                response = client.get(export_path, params={"format": fmt})
                assert response.status_code == 200, response.text
                assert response.content == content
            assert restarted_container.event_log.list_events() == events


def test_new_namespace_is_disjoint_even_when_legacy_and_structured_hash_inputs_match(
    database_path: Path,
) -> None:
    payload = {"source": "synthetic:a", "title": "b:c", "text": GRAPH_TEXT}
    encoded = f'["synthetic:a","b:c","{GRAPH_TEXT}"]'
    legacy_source, legacy_title, legacy_text = encoded.split(":", 2)
    legacy_payload = {"source": legacy_source, "title": legacy_title, "text": legacy_text}
    assert f"{legacy_source}:{legacy_title}:{legacy_text}" == encoded
    legacy_id = stable_id(encoded, "doc")
    application = create_app(offline_settings(database_path))
    container: AppContainer = application.state.container
    legacy = Document(document_id=legacy_id, **legacy_payload, metadata={"source_type": "api"})
    chunks = container.ingestion_pipeline.ingest_documents([legacy])
    legacy_result = IngestResponse(
        document_id=legacy_id, chunk_ids=[chunk.chunk_id for chunk in chunks]
    )
    with TestClient(application) as client:
        current = ingest(client, payload)
        assert current.document_id == f"doc-api-v2-{legacy_id.removeprefix('doc-')}"
        assert current.document_id != legacy_id
        assert_persisted_papers(
            client, database_path, [(legacy_payload, legacy_result), (payload, current)]
        )
