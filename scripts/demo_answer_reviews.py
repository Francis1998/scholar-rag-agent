"""Measure persistent human opinions on a real, synthetic, offline saved answer."""

import argparse
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from agent.review_models import ReviewDecision, ReviewHistoryPage, ReviewSubmission
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import IngestResponse, QueryResponse
from scripts.demo_evidence_export import offline_settings

PANEL_TITLES = (
    "1. Create a real offline saved answer",
    "2. Append an opinion, safely retry its UUID",
    "3. Review frozen evidence, not today's corpus",
    "4. Reopen SQLite and read ordered history",
)
OUTPUT_NAMES = (
    "bundle.json",
    "bundle.md",
    "events.json",
    "review-created.json",
    "review-retry.json",
    "review-conflict.json",
    "invalid-reference.json",
    "accepted-review.json",
    "page-1.json",
    "page-2.json",
    "transcript.txt",
)
DEMO_TEXT = (
    "Synthetic data, not a real paper or finding. GraphRAG connects research entities. "
    "The example uses local retrieval and a fake answer to demonstrate evidence review. "
    "A human's acceptance of this workflow does not establish scientific correctness."
)


def _expect(response: httpx.Response, status: int) -> None:
    if response.status_code != status:
        raise RuntimeError(f"Offline demo expected HTTP {status}, got {response.status_code}.")


def _event_rows(database_path: Path) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(database_path)) as connection:
        return connection.execute("SELECT * FROM agent_events ORDER BY id").fetchall()


def _block_work(stack: ExitStack, container: AppContainer) -> list[Mock]:
    return [
        stack.enter_context(
            patch.object(
                component, method, side_effect=AssertionError("Reviews must not run agent work.")
            )
        )
        for component, method in (
            (container.runner, "run"),
            (container.runner, "preview"),
            (container.llm, "generate"),
            (container.hybrid_retriever, "retrieve"),
            (container.document_store, "list_chunks"),
            (container.graph_store, "chunks_for_entities"),
            (container.event_log, "append_event"),
            (container.event_log, "append_transition"),
        )
    ]


def run_demo(output_dir: Path) -> str:
    """Execute actual HTTP requests; leave response artifacts, never the temporary database."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        if (output_dir / name).exists():
            raise FileExistsError(
                f"Refusing to overwrite {output_dir / name}; choose a new directory."
            )
    artifacts: dict[str, bytes] = {}
    with TemporaryDirectory(prefix="scholar-answer-reviews-") as temporary, ExitStack() as stack:
        for transport, method in (
            (httpx.HTTPTransport, "handle_request"),
            (httpx.AsyncHTTPTransport, "handle_async_request"),
        ):
            stack.enter_context(
                patch.object(
                    transport, method, side_effect=AssertionError("The demo forbids live HTTP.")
                )
            )
        database_path = Path(temporary) / "reviews.sqlite3"
        settings = offline_settings(database_path)
        application = create_app(settings)
        container: AppContainer = application.state.container
        with TestClient(application) as client:
            ingested = client.post(
                "/ingest/text",
                json={
                    "title": "Synthetic review note",
                    "text": DEMO_TEXT,
                    "source": "synthetic:review-demo",
                },
            )
            _expect(ingested, 200)
            completed = client.post("/query", json={"query": "What does GraphRAG connect?"})
            _expect(completed, 200)
            result = QueryResponse.model_validate_json(completed.content).result
            if result.state != "DONE":
                raise RuntimeError("The synthetic query did not finish.")
            export_path = f"/runs/{result.run_id}/export"
            review_path = f"/runs/{result.run_id}/reviews"
            original = client.get(export_path)
            markdown = client.get(export_path, params={"format": "markdown"})
            events = client.get(f"/runs/{result.run_id}/events")
            for response in (original, markdown, events):
                _expect(response, 200)
            bundle = EvidenceBundle.model_validate_json(original.content)
            if bundle.generation.provider != "fake" or len(bundle.snapshot.sources) != 1:
                raise RuntimeError("Expected a fake-adapter answer with one synthetic passage.")
            chunk_id = bundle.snapshot.sources[0].chunk.chunk_id
            before_events = _event_rows(database_path)
            before_runs = client.get("/runs")
            _expect(before_runs, 200)
            other = client.post(
                "/ingest/text",
                json={
                    "title": "Later corpus note",
                    "text": "A current-only synthetic note.",
                    "source": "synthetic:later",
                },
            )
            _expect(other, 200)
            current_only = IngestResponse.model_validate_json(other.content).chunk_ids[0]
            blocked = _block_work(stack, container)
            draft = ReviewSubmission(
                review_id=uuid4(),
                decision=ReviewDecision.NEEDS_REVISION,
                comment="Explain the limitations before reusing this synthetic answer.",
                cited_chunk_ids=(chunk_id,),
            )
            created = client.post(review_path, json=draft.model_dump(mode="json"))
            retry = client.post(review_path, json=draft.model_dump(mode="json"))
            conflict = client.post(
                review_path, json={**draft.model_dump(mode="json"), "comment": "Changed opinion."}
            )
            _expect(created, 201)
            _expect(retry, 200)
            _expect(conflict, 409)
            identical_retry = created.content == retry.content
            if not identical_retry or conflict.json()["detail"]["code"] != "review_id_conflict":
                raise RuntimeError("The retry/conflict contract changed.")
            invalid = client.post(
                review_path,
                json={
                    **draft.model_dump(mode="json"),
                    "review_id": str(uuid4()),
                    "cited_chunk_ids": [current_only],
                },
            )
            _expect(invalid, 422)
            if invalid.json()["detail"]["code"] != "invalid_review_references":
                raise RuntimeError("Expected a frozen-evidence reference rejection.")
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute("UPDATE chunks SET text = 'Replaced synthetic corpus'")
                connection.execute("DELETE FROM entity_edges")
                connection.execute("DELETE FROM entity_mentions")
                connection.execute("DELETE FROM graph_chunks")
                connection.execute("DELETE FROM chunks")
                connection.execute("DELETE FROM documents")
                remaining = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]

        reopened = create_app(settings)
        blocked.extend(_block_work(stack, reopened.state.container))
        with TestClient(reopened) as client:
            accepted = client.post(
                review_path,
                json={
                    **draft.model_dump(mode="json"),
                    "review_id": str(uuid4()),
                    "decision": "accepted",
                    "comment": "Accepted as a workflow demonstration only, not scientific proof.",
                },
            )
            _expect(accepted, 201)
            first = client.get(review_path, params={"limit": 1})
            _expect(first, 200)
            first_page = ReviewHistoryPage.model_validate_json(first.content)
            if first_page.next_cursor is None:
                raise RuntimeError("Expected a lookahead cursor for the older review.")
            second = client.get(review_path, params={"limit": 1, "cursor": first_page.next_cursor})
            _expect(second, 200)
            second_page = ReviewHistoryPage.model_validate_json(second.content)
            if (
                [r.decision for r in first_page.reviews] != ["accepted"]
                or [r.decision for r in second_page.reviews] != ["needs_revision"]
                or second_page.next_cursor is not None
                or second_page.reviews[0].review_id != draft.review_id
            ):
                raise RuntimeError("Restarted history did not preserve both ordered opinions.")
            after_json = client.get(export_path)
            after_markdown = client.get(export_path, params={"format": "markdown"})
            after_runs = client.get("/runs")
            for response in (after_json, after_markdown, after_runs):
                _expect(response, 200)
            json_equal = original.content == after_json.content
            markdown_equal = markdown.content == after_markdown.content
            events_equal = before_events == _event_rows(database_path)
            runs_equal = before_runs.content == after_runs.content
            calls = sum(mock.call_count for mock in blocked)
            if (
                not all((json_equal, markdown_equal, events_equal, runs_equal))
                or remaining
                or calls
            ):
                raise RuntimeError("The saved answer changed, work ran, or corpus cleanup failed.")
        artifacts = {
            "bundle.json": original.content,
            "bundle.md": markdown.content,
            "events.json": events.content,
            "review-created.json": created.content,
            "review-retry.json": retry.content,
            "review-conflict.json": conflict.content,
            "invalid-reference.json": invalid.content,
            "accepted-review.json": accepted.content,
            "page-1.json": first.content,
            "page-2.json": second.content,
        }
    removed = not database_path.exists()
    if not removed:
        raise RuntimeError("The temporary demo database was not removed.")
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"POST /ingest/text -> {ingested.status_code}",
                        f"POST /query -> {completed.status_code}; state={result.state}",
                        f"Provider: {bundle.generation.provider}; "
                        f"saved passages: {len(bundle.snapshot.sources)}",
                        f"Saved agent events: {len(before_events)}",
                        "Synthetic data and a fake answer, not scientific findings.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"needs_revision POST: {created.status_code}; "
                        f"same UUID retry: {retry.status_code}",
                        f"Retry returns byte-identical record: {identical_retry}",
                        f"Changed comment with same UUID: {conflict.status_code}",
                        f"Existing current-only chunk rejected: {invalid.status_code}",
                        "Comments are human opinions, not factual verification.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Replace/delete demo corpus; remaining chunks: {remaining}",
                        f"After restart, cite saved chunk with new UUID: {accepted.status_code}",
                        f"JSON export unchanged: {json_equal}; "
                        f"Markdown unchanged: {markdown_equal}",
                        f"Agent event rows unchanged: {events_equal}",
                        f"Recorded run catalog unchanged: {runs_equal}",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Page 1: {first_page.reviews[0].decision}; "
                        f"exclusive cursor: {first_page.next_cursor}",
                        f"Page 2: {second_page.reviews[0].decision}; "
                        f"next cursor: {second_page.next_cursor}",
                        f"Post-save runtime/corpus-read/event-write calls: {calls}",
                        f"Temporary demo database removed: {removed}",
                        "Saved: real API JSON, evidence Markdown, and this transcript.",
                        "No reviewer authentication, ranking changes, or training.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    artifacts["transcript.txt"] = transcript.encode("utf-8")
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as output:
            output.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
