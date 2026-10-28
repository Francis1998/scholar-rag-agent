"""Execute a native answer review on a new, original synthetic saved-evidence fixture."""

import argparse
import hashlib
import json
import socket
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.evidence import EvidenceSnapshot, GenerationRecord, RunConfiguration
from agent.models import (
    AgentAnswer,
    AgentState,
    QueryIntent,
    QueryObservation,
    QueryPlan,
    StateTransition,
)
from agent.review_models import ReviewHistoryPage, ReviewSubmission
from api.application import create_app
from api.dependencies import AppContainer
from llm.schemas import TaskType
from retrieval.models import Chunk, Document, SearchResult
from scripts.demo_corpus_explorer import _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_frozen_annotation import AuthoringPage

FIXTURE = "scholar-saved-answer-review-synthetic-v1"
RUN_ID = "synthetic-answer-review"
CHUNK_ID = "chunk/../synthetic?\u7814#v1"
TEXT = (
    "Original synthetic workshop note, not a publication.\r\n"
    "A saved answer can be inspected without repeating retrieval or generation.\r\n"
    "Human acceptance is an opinion, not proof. Cafe\u0301 \U0001f52c.\r\n"
    "Literal <script>source text</script> & context."
)
COMMENT = (
    "  Needs revision: distinguish recorded grounding from scientific support.\r\n"
    "Human opinion only. Cafe\u0301 \U0001f52c; <b>literal comment</b>.  "
)
FRAME_PAGES = ("saved-answer", "compose", "preview", "saved-review")
ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}


@contextmanager
def authoring_guards(container: AppContainer) -> Iterator[dict[str, Mock]]:
    with browsing_guards(container) as guards, ExitStack() as stack:
        for target, method in (
            (container.document_catalog, "list_documents"),
            (container.document_chunks, "list_chunks"),
            (container.source_context, "read"),
            (container.event_log, "list_events"),
            (container.event_log, "append_transition"),
            (container.evidence_exporter, "export"),
            (container.answer_reviews, "create"),
            (container.evidence_annotations, "create"),
            (container.paper_screening, "submit"),
            (socket, "create_connection"),
            (socket.socket, "connect"),
        ):
            name = f"{getattr(target, '__name__', type(target).__name__)}.{method}"
            guards[name] = stack.enter_context(patch.object(target, method, side_effect=_no_work))
        yield guards


def _seed(container: AppContainer) -> None:
    document = Document(
        document_id="paper/../synthetic?\u7814#v1",
        title="Original synthetic answer-review source",
        source="synthetic:answer-review",
        text=TEXT,
    )
    chunk = Chunk(**document.model_dump(), chunk_id=CHUNK_ID)
    container.document_store.add_documents([document], [chunk])
    query = "What does this saved answer establish?"
    snapshot = EvidenceSnapshot.capture(
        query, [SearchResult(chunk=chunk, score=1.0, retriever="synthetic-fixture")]
    )
    plan = QueryPlan(
        run_id=RUN_ID,
        observation=QueryObservation(original_query=query, intent=QueryIntent.FACTUAL_LOOKUP),
        tasks=[],
        rationale_trace=["Manually seeded original fixture, not model reasoning."],
    )
    answer = AgentAnswer(
        answer="Synthetic saved answer: recorded evidence can be revisited without rerunning.",
        claims=[],
        citations=[],
        ungrounded=True,
        warnings=["Original synthetic fixture, not a model output or a scientific finding."],
    )
    configuration = RunConfiguration(
        max_source_docs=5, max_hops=2, retrieval_timeout_seconds=30, reasoning_timeout_seconds=60
    )
    log = container.event_log

    def transition(before: AgentState, after: AgentState, payload: dict[str, object]) -> None:
        log.append_transition(
            StateTransition(
                agent_id="synthetic-review-fixture",
                run_id=RUN_ID,
                from_state=before,
                to_state=after,
                payload=payload,
            )
        )

    transition(
        AgentState.IDLE,
        AgentState.PLANNING,
        {"query": query, "configuration": configuration.model_dump(mode="json")},
    )
    log.append_event(
        "synthetic-review-fixture", RUN_ID, "decision_log", plan.model_dump(mode="json")
    )
    transition(AgentState.PLANNING, AgentState.RETRIEVING, plan.model_dump(mode="json"))
    transition(AgentState.RETRIEVING, AgentState.REASONING, {"chunk_ids": [CHUNK_ID]})
    log.append_event(
        "synthetic-review-fixture", RUN_ID, "evidence_snapshot", snapshot.model_dump(mode="json")
    )
    log.append_event(
        "synthetic-review-fixture",
        RUN_ID,
        "generation_record",
        GenerationRecord(
            provider="synthetic-fixture",
            model_name=None,
            task_type=TaskType.REASONING,
            claim_chunk_ids=[],
        ).model_dump(mode="json"),
    )
    transition(AgentState.REASONING, AgentState.ANSWERING, answer.model_dump(mode="json"))
    transition(AgentState.ANSWERING, AgentState.DONE, {"ungrounded": True})


def run_demo(output_dir: Path) -> str:
    """Save measured artifacts without touching any existing database or output directory."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    database = output_dir / "review.sqlite3"
    settings = offline_settings(database)
    application = create_app(settings)
    container: AppContainer = application.state.container
    _seed(container)
    events = container.event_log.list_events()
    bundle = container.saved_run_evidence.export(RUN_ID)
    artifacts: dict[str, bytes] = {}
    with (
        TestClient(application, root_path="/review") as client,
        authoring_guards(container) as guards,
        patch.object(
            container.notebook_reviews, "create", wraps=container.notebook_reviews.create
        ) as create,
    ):

        def get(name: str, url: str) -> AuthoringPage:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path.startswith("/review/"):
                raise RuntimeError("Demo navigation left the local mounted app.")
            response = client.get(url)
            response.raise_for_status()
            if response.headers["cache-control"] != "no-store":
                raise RuntimeError("An authoring response lost no-store.")
            artifacts[name + ".html"] = response.content
            return AuthoringPage(response.text)

        before = database.read_bytes()
        history = get("history", "/review/explore/runs?limit=1&state=DONE")
        source_url = history.links["inspect-run-1"]
        reader = get("saved-answer", source_url)
        form_url = reader.links["review-answer"]
        opened = get("compose", form_url)
        controls = opened.forms["review-draft"]
        draft = {
            "selection": controls["selection"],
            "review_id": controls["review_id"],
            "decision": "needs_revision",
            "comment": COMMENT,
            "cite_1": controls["cite_1"],
        }
        response = client.post(opened.actions["review-draft"], data=draft, headers=ORIGIN)
        response.raise_for_status()
        artifacts["preview.html"] = response.content
        page = AuthoringPage(response.text)
        fields = page.forms["review-save"]
        payload = ReviewSubmission.model_validate_json(fields["submission"])
        if (
            database.read_bytes() != before
            or create.call_count != 0
            or payload.comment != COMMENT
            or payload.cited_chunk_ids != (CHUNK_ID,)
            or page.values["review-comment"] != [COMMENT]
            or page.passages != [TEXT]
        ):
            raise RuntimeError("Preview changed storage or lost the exact human opinion.")
        saved = client.post(
            page.actions["review-save"], data=fields, headers=ORIGIN, follow_redirects=False
        )
        if saved.status_code != 303:
            raise RuntimeError("The explicit save did not return 303.")
        location = saved.headers["location"]
        if any(name + "=" in location for name in ("comment", "decision", "submission")):
            raise RuntimeError("Private opinion text leaked into a redirect.")
        persisted = get("saved-review", location)
        original = container.notebook_reviews.get_review(RUN_ID, payload.review_id)
        if persisted.values["review-comment"] != [COMMENT] or original.comment != COMMENT:
            raise RuntimeError("The precise persisted reader lost the comment.")
        before_rejections = database.read_bytes()
        retry = client.post(
            page.actions["review-save"], data=fields, headers=ORIGIN, follow_redirects=False
        )
        rejected = {
            "cross_origin": client.post(
                page.actions["review-save"],
                data=fields,
                headers={"origin": "https://external.invalid"},
            ).status_code,
            "unknown_source": client.post(
                page.actions["review-save"],
                data={
                    **fields,
                    "submission": json.dumps(
                        {**json.loads(fields["submission"]), "cited_chunk_ids": ["unknown"]}
                    ),
                },
                headers=ORIGIN,
            ).status_code,
            "changed_uuid_payload": client.post(
                page.actions["review-save"],
                data={
                    **fields,
                    "submission": json.dumps(
                        {**json.loads(fields["submission"]), "comment": "A different opinion."}
                    ),
                },
                headers=ORIGIN,
            ).status_code,
        }
        reviews = container.notebook_reviews.list_reviews(RUN_ID)
        if (
            retry.status_code != 303
            or retry.headers["location"] != location
            or rejected != {"cross_origin": 403, "unknown_source": 422, "changed_uuid_payload": 409}
            or database.read_bytes() != before_rejections
            or reviews.reviews != [original]
        ):
            raise RuntimeError("Retry or rejected-write invariants failed.")
        counts = {name: guard.call_count for name, guard in guards.items()}
        attempts = create.call_count
    with TestClient(application) as client:
        api_response = client.get(f"/runs/{RUN_ID}/reviews")
        api_response.raise_for_status()
        if ReviewHistoryPage.model_validate_json(api_response.content) != reviews:
            raise RuntimeError("The unchanged JSON API disagrees with Python/native persistence.")
        artifacts["reviews.json"] = api_response.content
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE chunks SET text = 'CHANGED CURRENT CORPUS'")
    for stage in ("changed", "deleted"):
        if stage == "deleted":
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("DELETE FROM chunks")
                connection.execute("DELETE FROM documents")
        restarted = create_app(settings)
        before = database.read_bytes()
        with (
            TestClient(restarted, root_path="/review") as client,
            authoring_guards(restarted.state.container) as guards,
        ):
            response = client.post(opened.actions["review-draft"], data=draft, headers=ORIGIN)
            if response.content != artifacts["preview.html"]:
                raise RuntimeError("Frozen preview changed after corpus edit/deletion/restart.")
            response = client.post(
                page.actions["review-save"], data=fields, headers=ORIGIN, follow_redirects=False
            )
            if response.status_code != 303 or response.headers["location"] != location:
                raise RuntimeError("Unchanged retry failed after restart.")
            persisted_response = client.get(location)
            if (
                persisted_response.content != artifacts["saved-review.html"]
                or restarted.state.container.notebook_reviews.get_review(RUN_ID, payload.review_id)
                != original
                or restarted.state.container.saved_run_evidence.export(RUN_ID) != bundle
            ):
                raise RuntimeError("The exact persisted opinion or frozen evidence changed.")
            artifacts[stage + "-restart.html"] = persisted_response.content
            counts.update({f"{stage}:{name}": guard.call_count for name, guard in guards.items()})
        if database.read_bytes() != before:
            raise RuntimeError("Read/preview/retry changed the restarted database.")
    if any(counts.values()) or container.event_log.list_events() != events or attempts != 3:
        raise RuntimeError("Unexpected work or service calls occurred.")
    checks = {
        "fixture": FIXTURE,
        "run_id": RUN_ID,
        "comment": COMMENT,
        "review_id": str(original.review_id),
        "created_at": original.created_at.isoformat(),
        "review_count": 1,
        "save_status": saved.status_code,
        "retry_status": retry.status_code,
        "rejected_statuses": rejected,
        "get_preview_database_unchanged": True,
        "rejections_database_unchanged": True,
        "retry_record_unchanged": True,
        "python_api_equal": True,
        "corpus_change_delete_restart_identical": True,
        "events_before": len(events),
        "events_after": len(container.event_log.list_events()),
        "forbidden_call_counts": counts,
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
    }
    transcript = (
        "Saved-answer review workbench / executed original synthetic workflow\n"
        "Setup: one manually recorded eight-event fixture; no model or retrieval setup.\n"
        "Mounted native route: saved answer -> compose -> preview -> explicit save -> exact UUID.\n"
        "Explicit verdict: needs_revision; one actual frozen chunk selected by the human.\n"
        "Comment: leading/trailing spaces, CRLF, combining mark, emoji "
        "and literal markup retained.\n"
        "GET and preview: no service writes; database bytes unchanged.\n"
        "Save: 303 to exact persisted opinion; comment absent from GET URLs.\n"
        "Unchanged retry: 303; one record with original UUID, sequence and timestamp.\n"
        "Cross-origin: 403; unknown frozen source: 422; changed UUID payload: 409.\n"
        "Current corpus changed, deleted, app restarted: preview, opinion "
        "and frozen evidence identical.\n"
        "Python/API records equal; agent events unchanged: 8; forbidden calls: 0.\n"
        "GIF: four actual local browser screenshots, not a continuous recording.\n"
        "Capture: isolated Chrome renderings of guarded HTTP responses, "
        "not a live-click workflow.\n"
        "Synthetic human opinions are not scientific validation or authenticated approval.\n"
    )
    artifacts["transcript.txt"] = transcript.encode()
    artifacts["checks.json"] = (json.dumps(checks, indent=2, ensure_ascii=True) + "\n").encode()
    artifacts["pages.json"] = (
        json.dumps({"source": source_url, "compose": form_url, "saved": location}, indent=2) + "\n"
    ).encode()
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as output:
            output.write(content)
    return transcript


def serve_demo(output_dir: Path, port: int) -> None:
    database = output_dir / "review.sqlite3"
    checks = json.loads((output_dir / "checks.json").read_text(encoding="utf-8"))
    if (
        checks.get("fixture") != FIXTURE
        or database.is_symlink()
        or not database.is_file()
        or hashlib.sha256(database.read_bytes()).hexdigest() != checks["database_sha256"]
    ):
        raise ValueError("Expected an unchanged saved-answer-review synthetic demo directory.")
    application = create_app(offline_settings(database))
    mounted = FastAPI()
    mounted.mount("/review", application)
    with authoring_guards(application.state.container):
        uvicorn.run(mounted, host="127.0.0.1", port=port, access_log=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", type=Path)
    action.add_argument("--serve", type=Path)
    parser.add_argument("--port", type=int, default=8777)
    arguments = parser.parse_args()
    if arguments.serve is not None:
        serve_demo(arguments.serve, arguments.port)
    else:
        print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
