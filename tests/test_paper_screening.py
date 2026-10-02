"""Human screening is revisioned metadata, never an automatic research judgment."""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from storage import paper_screening


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Screening must not retrieve, generate, journal, or use the network.")


@pytest.fixture
def screening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, AppContainer, str, list[str]]]:
    app = create_app(offline_settings(tmp_path / "screening.sqlite3"))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    with TestClient(app) as client:
        ids = []
        for title in ("Methods", "Background", "Comparison"):
            response = client.post(
                "/ingest/text",
                json={
                    "title": title,
                    "text": f"Synthetic GraphRAG {title}.",
                    "source": "synthetic",
                },
            )
            response.raise_for_status()
            ids.append(response.json()["document_id"])
        created = client.post("/collections", json={"name": "Review", "document_ids": ids})
        created.raise_for_status()
        path = f"/collections/{created.json()['collection_id']}/screening"
        yield client, app.state.container, path, ids


def submission(
    decision: str = "include", *, collection: int = 1, revision: int = 0
) -> dict[str, object]:
    return {
        "collection_revision": collection,
        "expected_decision_revision": revision,
        "decision": decision,
        "reason": "Human opinion about this synthetic note.",
    }


def queue(client: TestClient, path: str, revision: int = 1, **params: object) -> dict[str, Any]:
    response = client.get(path, params={"collection_revision": revision, **params})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    result: dict[str, Any] = response.json()
    return result


def test_screening_queue_decisions_restart_and_explicit_retrieval(
    screening: tuple[TestClient, AppContainer, str, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, container, path, ids = screening
    chunks = container.document_store.list_chunks()
    with monkeypatch.context() as patch:
        for target, name in (
            (container.llm, "generate"),
            (container.runner, "run"),
            (container.runner, "preview"),
            (container.event_log, "append_event"),
            (container.event_log, "append_transition"),
        ):
            patch.setattr(target, name, denied)
        initial = queue(client, path)
        assert [item["document_id"] for item in initial["items"]] == ids
        assert initial["counts"] == {
            "unscreened": 3,
            "include": 0,
            "exclude": 0,
            "unsure": 0,
            "stale": 0,
        }
        assert initial["included_document_ids"] == []
        for identifier, decision in zip(ids, ("include", "exclude", "unsure"), strict=True):
            response = client.put(f"{path}/{identifier}", json=submission(decision))
            assert response.status_code == 200, response.text
            assert response.json()["revision"] == 1
        reviewed = queue(client, path)
        assert reviewed["included_document_ids"] == ids[:1]
        assert reviewed["counts"] == {
            "unscreened": 0,
            "include": 1,
            "exclude": 1,
            "unsure": 1,
            "stale": 0,
        }
    assert container.event_log.list_events() == []
    assert container.document_store.list_chunks() == chunks
    restarted = create_app(offline_settings(tmp_path / "screening.sqlite3"))
    with TestClient(restarted) as other:
        assert queue(other, path) == reviewed
        preview = other.post(
            "/retrieve",
            json={"query": "GraphRAG", "document_ids": reviewed["included_document_ids"]},
        )
        assert preview.status_code == 200, preview.text
        assert {source["chunk"]["document_id"] for source in preview.json()["sources"]} == {ids[0]}
        assert restarted.state.container.event_log.list_events() == []


def test_stale_collection_and_decision_revisions_never_apply_old_inclusions(
    screening: tuple[TestClient, AppContainer, str, list[str]],
) -> None:
    client, _, path, ids = screening
    first = client.put(f"{path}/{ids[0]}", json=submission())
    assert first.status_code == 200
    stale_write = client.put(f"{path}/{ids[0]}", json=submission("exclude"))
    assert stale_write.status_code == 409
    assert stale_write.json()["detail"]["code"] == "screening_revision_conflict"
    collection = client.put(
        path.removesuffix("/screening"),
        json={"name": "Changed criteria", "document_ids": ids, "expected_revision": 1},
    )
    assert collection.status_code == 200
    assert client.get(path, params={"collection_revision": 1}).status_code == 409
    assert client.put(f"{path}/{ids[1]}", json=submission()).status_code == 409
    current = queue(client, path, revision=2)
    assert current["included_document_ids"] == []
    assert current["counts"]["stale"] == 1
    assert current["items"][0]["status"] == "stale"
    assert current["items"][0]["review"]["collection_revision"] == 1
    renewed = client.put(f"{path}/{ids[0]}", json=submission(collection=2, revision=1))
    assert renewed.status_code == 200
    assert renewed.json()["revision"] == 2
    assert queue(client, path, revision=2)["included_document_ids"] == ids[:1]


def test_member_order_paging_and_filters_keep_full_collection_counts(
    screening: tuple[TestClient, AppContainer, str, list[str]],
) -> None:
    client, _, path, ids = screening
    assert client.put(f"{path}/{ids[1]}", json=submission("exclude")).status_code == 200
    first = queue(client, path, limit=1)
    second = queue(client, path, limit=1, cursor=first["next_cursor"])
    third = queue(client, path, limit=1, cursor=second["next_cursor"])
    assert [page["items"][0]["document_id"] for page in (first, second, third)] == ids
    assert third["next_cursor"] is None
    assert first["counts"] == second["counts"] == third["counts"]
    pending = queue(client, path, status="unscreened")
    assert [item["document_id"] for item in pending["items"]] == [ids[0], ids[2]]
    invalid_cursor = client.get(path, params={"collection_revision": 1, "cursor": "unknown"})
    assert invalid_cursor.status_code == 422
    assert invalid_cursor.json()["detail"]["code"] == "screening_cursor_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision", "automatic"),
        ("reason", ""),
        ("reason", " \t\n"),
        ("reason", "x" * 1001),
        ("reason", "\x00"),
        ("reason", "\ud800"),
        ("collection_revision", True),
        ("collection_revision", 1.0),
        ("collection_revision", "1"),
        ("expected_decision_revision", -1),
        ("expected_decision_revision", True),
        ("expected_decision_revision", 2**63),
    ],
)
def test_invalid_decisions_have_no_side_effects(
    screening: tuple[TestClient, AppContainer, str, list[str]], field: str, value: object
) -> None:
    client, container, path, ids = screening
    response = client.put(
        f"{path}/{ids[0]}",
        content=json.dumps({**submission(), field: value}),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert queue(client, path)["counts"]["unscreened"] == 3
    assert container.event_log.list_events() == []


def test_missing_member_and_deleted_corpus_fail_closed(
    screening: tuple[TestClient, AppContainer, str, list[str]], tmp_path: Path
) -> None:
    client, _, path, ids = screening
    response = client.put(f"{path}/unknown", json=submission())
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "screening_document_not_member"
    with sqlite3.connect(tmp_path / "screening.sqlite3") as connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", (ids[1],))
    assert client.get(path, params={"collection_revision": 1}).status_code == 409
    assert client.put(f"{path}/{ids[0]}", json=submission()).status_code == 409


def test_no_inclusions_cannot_become_an_unscoped_query(
    screening: tuple[TestClient, AppContainer, str, list[str]],
) -> None:
    client, container, path, _ = screening
    included = queue(client, path)["included_document_ids"]
    assert included == []
    response = client.post("/retrieve", json={"query": "GraphRAG", "document_ids": included})
    assert response.status_code == 422
    assert container.event_log.list_events() == []


def test_response_limit_measures_utf8_bytes_without_truncation(
    screening: tuple[TestClient, AppContainer, str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, path, ids = screening
    reason = "\U0001f4da" * 1000
    saved = client.put(f"{path}/{ids[0]}", json={**submission(), "reason": reason})
    assert saved.status_code == 200
    original = client.get(path, params={"collection_revision": 1})
    assert original.status_code == 200
    assert original.json()["items"][0]["review"]["reason"] == reason
    assert len(original.content) > len(original.text)
    monkeypatch.setattr(paper_screening, "MAX_RESPONSE_BYTES", len(original.content))
    assert client.get(path, params={"collection_revision": 1}).content == original.content
    monkeypatch.setattr(paper_screening, "MAX_RESPONSE_BYTES", len(original.content) - 1)
    too_large = client.get(path, params={"collection_revision": 1})
    assert too_large.status_code == 413
    assert too_large.json()["detail"]["code"] == "screening_response_too_large"
    assert "items" not in too_large.json()


def test_invalid_store_and_deleted_collection_return_private_errors(
    screening: tuple[TestClient, AppContainer, str, list[str]], tmp_path: Path
) -> None:
    client, _, path, ids = screening
    assert client.put(f"{path}/{ids[0]}", json=submission()).status_code == 200
    with sqlite3.connect(tmp_path / "screening.sqlite3") as connection:
        connection.execute("UPDATE paper_screening_decisions SET record = 'private-invalid-data'")
    corrupt = client.get(path, params={"collection_revision": 1})
    assert corrupt.status_code == 409
    assert corrupt.json()["detail"]["code"] == "invalid_screening_record"
    assert "private-invalid-data" not in corrupt.text
    with sqlite3.connect(tmp_path / "screening.sqlite3") as connection:
        connection.execute("DROP TABLE paper_screening_decisions")
    unavailable = client.get(path, params={"collection_revision": 1})
    assert unavailable.status_code == 503
    assert unavailable.headers["cache-control"] == "no-store"
    assert "paper_screening_decisions" not in unavailable.text
    deleted = client.delete(path.removesuffix("/screening"), params={"expected_revision": 1})
    assert deleted.status_code == 204
    assert client.get(path, params={"collection_revision": 1}).status_code == 404


def test_api_preserves_slashes_in_existing_document_identities(
    screening: tuple[TestClient, AppContainer, str, list[str]],
) -> None:
    client, container, _, _ = screening
    identifier = "doi:10.1234/synthetic-paper"
    container.document_store.add_documents(
        [Document(document_id=identifier, title="Synthetic", text="Synthetic", source="synthetic")],
        [],
    )
    created = client.post(
        "/collections", json={"name": "Opaque identities", "document_ids": [identifier]}
    )
    assert created.status_code == 201
    path = f"/collections/{created.json()['collection_id']}/screening"
    response = client.put(f"{path}/{quote(identifier, safe='')}", json=submission())
    assert response.status_code == 200, response.text
    assert response.json()["document_id"] == identifier
    assert queue(client, path)["included_document_ids"] == [identifier]
