"""Measure exact frozen-quote annotations through the real API, entirely offline."""

import argparse
import hashlib
import json
import socket
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from scripts.demo_evidence_export import offline_settings
from storage.evidence_annotations import (
    AnnotationPage,
    AnnotationSubmission,
    SavedEvidenceAnnotation,
)

DEMO_TEXT = (
    "Synthetic note; caf\u00e9 \U0001f52c e\u0301. GraphRAG connects evidence. "
    "Again: GraphRAG connects evidence."
)
DEMO_QUOTE = "GraphRAG connects evidence"
PANEL_TITLES = (
    "1. Freeze a synthetic source once",
    "2. Annotate the second exact occurrence",
    "3. Retry safely and page human notes",
    "4. Resume after corpus removal and restart",
)
OUTPUT_NAMES = (
    "evidence.json",
    "submission.json",
    "created.json",
    "replayed.json",
    "conflict.json",
    "first-page.json",
    "second-page.json",
    "annotations.json",
    "restarted.json",
    "checks.json",
    "transcript.txt",
)


def _block_annotation_work(stack: ExitStack, container: AppContainer) -> list[Mock]:
    return [
        stack.enter_context(
            patch.object(
                component,
                method,
                side_effect=AssertionError("An annotation attempted agent or corpus work."),
            )
        )
        for component, method in (
            (container.runner, "run"),
            (container.runner, "preview"),
            (container.llm, "generate"),
            (container.hybrid_retriever, "retrieve"),
            (container.document_store, "list_chunks"),
            (container.document_store, "add_documents"),
            (container.graph_store, "chunks_for_entities"),
            (container.event_log, "append_event"),
            (container.event_log, "append_transition"),
            (container.event_log, "list_events"),
        )
    ]


def _events(path: Path) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        return connection.execute("SELECT * FROM agent_events ORDER BY id LIMIT 101").fetchall()


def run_demo(output_dir: Path) -> str:
    """Write actual API responses to a NEW directory; never use or change a user's database."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, bytes] = {}
    with TemporaryDirectory(prefix="scholar-annotations-") as temporary, ExitStack() as stack:
        network = [
            stack.enter_context(
                patch.object(
                    target,
                    method,
                    side_effect=AssertionError("This synthetic demo forbids network access."),
                )
            )
            for target, method in (
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
                (socket, "create_connection"),
                (socket.socket, "connect"),
            )
        ]
        database_path = Path(temporary) / "annotations.sqlite3"
        settings = offline_settings(database_path)
        app = create_app(settings)
        container: AppContainer = app.state.container
        with TestClient(app) as client:
            ingested = client.post(
                "/ingest/text",
                json={
                    "title": "Synthetic annotation workshop note",
                    "source": "synthetic:evidence-annotations",
                    "text": DEMO_TEXT,
                },
            )
            ingested.raise_for_status()
            query = client.post("/query", json={"query": "What does GraphRAG connect?"})
            query.raise_for_status()
            result = QueryResponse.model_validate_json(query.content).result
            if result.state != "DONE":
                raise RuntimeError(f"The synthetic setup query failed: {result.error}")
            exported = client.get(f"/runs/{result.run_id}/export")
            exported.raise_for_status()
            artifacts["evidence.json"] = exported.content
            bundle = EvidenceBundle.model_validate_json(exported.content)
            if (
                bundle.generation.provider != "fake"
                or len(bundle.snapshot.sources) != 1
                or bundle.snapshot.sources[0].chunk.text != DEMO_TEXT
            ):
                raise RuntimeError("Expected one exact synthetic source and a fake setup answer.")
            source = bundle.snapshot.sources[0]
            start = source.chunk.text.rindex(DEMO_QUOTE)
            submission = AnnotationSubmission(
                annotation_id=uuid4(),
                document_id=source.chunk.document_id,
                chunk_id=source.chunk.chunk_id,
                source_text_sha256=source.text_sha256,
                start=start,
                end=start + len(DEMO_QUOTE),
                quote=DEMO_QUOTE,
                note="Human opinion: inspect this occurrence; not proof of the claim.",
            )
            artifacts["submission.json"] = submission.model_dump_json().encode("utf-8")
            events_before = _events(database_path)
            blocked = _block_annotation_work(stack, container)
            path = f"/runs/{result.run_id}/annotations"
            body = submission.model_dump(mode="json")
            created = client.post(path, json=body)
            created.raise_for_status()
            saved = SavedEvidenceAnnotation.model_validate_json(created.content)
            replay = client.post(path, json=body)
            replay.raise_for_status()
            conflict = client.post(path, json={**body, "note": "A changed human opinion."})
            if (
                created.status_code != 201
                or replay.status_code != 200
                or replay.content != created.content
                or conflict.status_code != 409
                or conflict.json()["detail"]["code"] != "annotation_id_conflict"
                or saved.start != DEMO_TEXT.rindex(DEMO_QUOTE)
                or saved.start == DEMO_TEXT.index(DEMO_QUOTE)
                or DEMO_TEXT[saved.start : saved.end] != saved.quote
            ):
                raise RuntimeError("Exact span, created/replay status, or retry identity failed.")
            artifacts["created.json"] = created.content
            artifacts["replayed.json"] = replay.content
            artifacts["conflict.json"] = conflict.content
            first_start = DEMO_TEXT.index(DEMO_QUOTE)
            second = client.post(
                path,
                json={
                    **body,
                    "annotation_id": str(uuid4()),
                    "start": first_start,
                    "end": first_start + len(DEMO_QUOTE),
                    "note": "A separate opinion on the first occurrence.",
                },
            )
            second.raise_for_status()
            first_page = client.get(path, params={"limit": 1})
            first_page.raise_for_status()
            page = AnnotationPage.model_validate_json(first_page.content)
            if page.next_cursor is None:
                raise RuntimeError("Expected a continuation after the newest annotation.")
            second_page = client.get(path, params={"limit": 1, "cursor": page.next_cursor})
            second_page.raise_for_status()
            older = AnnotationPage.model_validate_json(second_page.content)
            history = client.get(path)
            history.raise_for_status()
            complete = AnnotationPage.model_validate_json(history.content)
            if (
                second.status_code != 201
                or len(complete.annotations) != 2
                or page.annotations[0].sequence <= saved.sequence
                or page.next_cursor != page.annotations[0].sequence
                or older.annotations != [saved]
                or older.next_cursor is not None
                or complete.annotations != [*page.annotations, *older.annotations]
            ):
                raise RuntimeError("Expected two immutable notes with exclusive sequence paging.")
            artifacts["first-page.json"] = first_page.content
            artifacts["second-page.json"] = second_page.content
            artifacts["annotations.json"] = history.content
            # Only this newly created synthetic database is edited, outside annotation requests.
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute("UPDATE chunks SET text = 'Changed synthetic current corpus'")
            unchanged = client.get(path)
            unchanged.raise_for_status()
            if unchanged.content != history.content:
                raise RuntimeError("Current-corpus edits changed frozen annotations.")
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute("DELETE FROM chunks")
                connection.execute("DELETE FROM documents")
                remaining = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        restarted = create_app(settings)
        blocked.extend(_block_annotation_work(stack, restarted.state.container))
        before_reads = database_path.read_bytes()
        with TestClient(restarted) as client:
            recovered = client.get(path)
            recovered.raise_for_status()
            retried = client.post(path, json=body)
            retried.raise_for_status()
            artifacts["restarted.json"] = recovered.content
            if (
                recovered.content != history.content
                or retried.status_code != 200
                or retried.content != created.content
                or database_path.read_bytes() != before_reads
            ):
                raise RuntimeError("Restart changed annotation history or replay wrote data.")
        events_after = _events(database_path)
        work_calls = sum(guard.call_count for guard in blocked)
        network_calls = sum(guard.call_count for guard in network)
        if work_calls or network_calls or remaining != 0 or events_before != events_after:
            raise RuntimeError("Annotations performed unexpected work or changed saved events.")
    removed = not database_path.exists()
    if not removed:
        raise RuntimeError("The temporary synthetic database was not removed.")
    checks = {
        "synthetic_only": True,
        "setup_provider": bundle.generation.provider,
        "frozen_sources": len(bundle.snapshot.sources),
        "source_text_sha256": source.text_sha256,
        "first_occurrence_start": first_start,
        "selected_start": saved.start,
        "selected_end": saved.end,
        "span_codepoints": saved.end - saved.start,
        "created_status": created.status_code,
        "replay_status": replay.status_code,
        "conflict_status": conflict.status_code,
        "conflict_code": conflict.json()["detail"]["code"],
        "annotation_count": len(complete.annotations),
        "first_page_sequences": [record.sequence for record in page.annotations],
        "next_cursor": page.next_cursor,
        "second_page_sequences": [record.sequence for record in older.annotations],
        "remaining_corpus_chunks": remaining,
        "annotation_agent_or_corpus_calls": work_calls,
        "annotation_event_writes": len(events_after) - len(events_before),
        "events_unchanged": events_before == events_after,
        "network_calls": network_calls,
        "restart_identical": history.content == recovered.content,
        "history_bytes": len(history.content),
        "history_sha256": {
            "before": hashlib.sha256(history.content).hexdigest(),
            "after": hashlib.sha256(recovered.content).hexdigest(),
        },
        "temporary_database_removed": removed,
    }
    artifacts["checks.json"] = (json.dumps(checks, indent=2, sort_keys=True) + "\n").encode()
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"POST /ingest/text -> {ingested.status_code}",
                        f"POST /query -> {query.status_code}; saved state={result.state}",
                        f"Setup provider: {bundle.generation.provider} "
                        "(placeholder, not a finding)",
                        f"Frozen sources: {len(bundle.snapshot.sources)}; "
                        "repeated quote occurs twice",
                        f"Network calls: {network_calls}; ambient keys/settings ignored",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"POST /runs/<run_id>/annotations -> {created.status_code}",
                        f"Selected span: [{saved.start}, {saved.end}); "
                        f"first occurrence: {first_start}",
                        f"Exact quote: {saved.quote}",
                        f"Note: {saved.note}",
                        f"Source SHA-256 prefix: {saved.source_text_sha256[:24]}",
                        "Offsets count Python Unicode characters, not bytes or UTF-16 units.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Identical UUID + payload -> {replay.status_code}; byte-identical record",
                        f"Same UUID + changed note -> {conflict.status_code} "
                        f"({checks['conflict_code']})",
                        f"Saved annotations: {len(complete.annotations)}",
                        f"GET limit=1 -> sequences {checks['first_page_sequences']}; "
                        f"next_cursor={page.next_cursor}",
                        f"GET cursor={page.next_cursor} -> "
                        f"sequences {checks['second_page_sequences']}",
                        "No overwrite, fuzzy matching, or relocation to another occurrence.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Remaining current corpus chunks: {remaining}",
                        f"Recovered history byte-identical: {checks['restart_identical']}",
                        f"Annotation agent/corpus calls: {work_calls}; "
                        f"event writes: {checks['annotation_event_writes']}",
                        f"Saved events unchanged: {checks['events_unchanged']}",
                        f"Temporary database removed: {removed}; raw responses + checks saved",
                        "Human notes are opinions; exact quotes are provenance, not proof.",
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
