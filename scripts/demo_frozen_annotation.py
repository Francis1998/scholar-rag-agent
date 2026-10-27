"""Execute native annotation authoring on an isolated, original synthetic frozen passage."""

import argparse
import hashlib
import json
import socket
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlencode, urlsplit

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
from api.application import create_app
from api.dependencies import AppContainer
from llm.schemas import TaskType
from retrieval.models import Chunk, Document, SearchResult
from scripts.demo_corpus_explorer import _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_run_comparison_workbench import ComparisonPage
from storage.evidence_annotations import AnnotationPage, AnnotationSubmission

FIXTURE = "scholar-frozen-annotation-synthetic-v1"
RUN_ID = "synthetic-frozen-annotation"
DOCUMENT_ID = "paper/../synthetic\r\nid\0?draft#v1"
CHUNK_ID = "chunk/../synthetic\r\nid\0?part#v1"
QUOTE = "GraphRAG connects synthetic evidence"
TEXT = (
    "Original synthetic note, not a publication. Cafe\u0301 \U0001f52c.\r\n"
    f"{QUOTE}. Repetition is not independent support.\r\n"
    f"Again: {QUOTE}.\r\n"
    "Literal <script>source text</script> & context."
)
NOTE = "Second occurrence only.\r\nHuman opinion, not scientific validation. <b>Literal note</b>."
FRAME_PAGES = ("frozen-source", "compose", "preview", "saved-quote")
ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}


class AuthoringPage(ComparisonPage):
    """Inspect actual independent native forms, not invented request payloads."""

    def __init__(self, html: str) -> None:
        self.forms: dict[str, dict[str, str]] = {}
        self.actions: dict[str, str] = {}
        self._active_form = ""
        super().__init__(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id"):
            self._active_form = str(attributes["id"])
            self.forms[self._active_form] = {}
            self.actions[self._active_form] = str(attributes["action"])
        if tag == "input" and self._active_form and attributes.get("name"):
            self.forms[self._active_form][str(attributes["name"])] = str(
                attributes.get("value", "")
            )

    def handle_endtag(self, tag: str) -> None:
        super().handle_endtag(tag)
        if tag == "form":
            self._active_form = ""


@contextmanager
def authoring_guards(container: AppContainer) -> Iterator[dict[str, Mock]]:
    """Allow only frozen readers and the existing explicit annotation append service."""
    with browsing_guards(container) as guards, ExitStack() as stack:
        for target, method in (
            (container.document_catalog, "list_documents"),
            (container.document_chunks, "list_chunks"),
            (container.source_context, "read"),
            (container.event_log, "list_events"),
            (container.event_log, "append_transition"),
            (container.evidence_exporter, "export"),
            (container.answer_reviews, "create"),
            (container.paper_screening, "submit"),
            (socket, "create_connection"),
            (socket.socket, "connect"),
        ):
            guards[f"{getattr(target, '__name__', type(target).__name__)}.{method}"] = (
                stack.enter_context(patch.object(target, method, side_effect=_no_work))
            )
        yield guards


def _seed(container: AppContainer) -> None:
    """Record a labeled synthetic trace through real contracts; no model or retrieval setup."""
    document = Document(
        document_id=DOCUMENT_ID,
        title="Original synthetic annotation source",
        source="synthetic:frozen-annotation",
        text=TEXT,
    )
    chunk = Chunk(**document.model_dump(), chunk_id=CHUNK_ID)
    container.document_store.add_documents([document], [chunk])
    query = "Which exact occurrence does the human note refer to?"
    snapshot = EvidenceSnapshot.capture(
        query, [SearchResult(chunk=chunk, score=1.0, retriever="synthetic-fixture")]
    )
    plan = QueryPlan(
        run_id=RUN_ID,
        observation=QueryObservation(original_query=query, intent=QueryIntent.FACTUAL_LOOKUP),
        tasks=[],
        rationale_trace=["Manually seeded synthetic trace, not model reasoning."],
    )
    configuration = RunConfiguration(
        max_source_docs=5,
        max_hops=2,
        retrieval_timeout_seconds=30,
        reasoning_timeout_seconds=60,
    )
    answer = AgentAnswer(
        answer="Synthetic saved answer fixture. No scientific claim or model output.",
        claims=[],
        citations=[],
        ungrounded=True,
    )
    generation = GenerationRecord(
        provider="synthetic-fixture",
        model_name=None,
        task_type=TaskType.REASONING,
        claim_chunk_ids=[],
    )
    log = container.event_log

    def transition(before: AgentState, after: AgentState, payload: dict[str, object]) -> None:
        log.append_transition(
            StateTransition(
                agent_id="synthetic-annotation-fixture",
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
        "synthetic-annotation-fixture", RUN_ID, "decision_log", plan.model_dump(mode="json")
    )
    transition(AgentState.PLANNING, AgentState.RETRIEVING, plan.model_dump(mode="json"))
    transition(AgentState.RETRIEVING, AgentState.REASONING, {"chunk_ids": [CHUNK_ID]})
    log.append_event(
        "synthetic-annotation-fixture",
        RUN_ID,
        "evidence_snapshot",
        snapshot.model_dump(mode="json"),
    )
    log.append_event(
        "synthetic-annotation-fixture",
        RUN_ID,
        "generation_record",
        generation.model_dump(mode="json"),
    )
    transition(AgentState.REASONING, AgentState.ANSWERING, answer.model_dump(mode="json"))
    transition(AgentState.ANSWERING, AgentState.DONE, {"ungrounded": True})


def run_demo(output_dir: Path) -> str:
    """Save executed HTML, exact records and stable measurements; refuse existing destinations."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    database = output_dir / "annotation.sqlite3"
    settings = offline_settings(database)
    application = create_app(settings)
    container: AppContainer = application.state.container
    _seed(container)
    events = container.event_log.list_events()
    artifacts: dict[str, bytes] = {}
    with (
        TestClient(application, root_path="/review") as client,
        authoring_guards(container) as guards,
        patch.object(
            container.evidence_annotations, "create", wraps=container.evidence_annotations.create
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
        source_url = history.links["inspect-run-1"] + "#source-1"
        reader = get("frozen-source", source_url)
        form_url = reader.links["annotate-source-1"]
        opened = get("compose", form_url)
        start = TEXT.rindex(QUOTE)
        draft = {
            **opened.forms["annotation-draft"],
            "start": str(start),
            "end": str(start + len(QUOTE)),
            "note": NOTE,
        }
        response = client.post(opened.actions["annotation-draft"], data=draft, headers=ORIGIN)
        response.raise_for_status()
        artifacts["preview.html"] = response.content
        page = AuthoringPage(response.text)
        fields = page.forms["annotation-save"]
        payload = AnnotationSubmission.model_validate_json(fields["submission"])
        if (
            database.read_bytes() != before
            or create.call_count != 0
            or payload.start != start
            or payload.quote != QUOTE
            or payload.note != NOTE
            or payload.document_id != DOCUMENT_ID
            or payload.chunk_id != CHUNK_ID
            or page.highlights != [QUOTE]
            or page.passages != [TEXT]
        ):
            raise RuntimeError("Preview changed storage or lost the exact frozen selection.")
        saved = client.post(
            page.actions["annotation-save"], data=fields, headers=ORIGIN, follow_redirects=False
        )
        if saved.status_code != 303:
            raise RuntimeError("The explicit save did not return 303.")
        location = saved.headers["location"]
        if any(name + "=" in location for name in ("note", "quote", "submission")):
            raise RuntimeError("A private annotation field leaked into a redirect.")
        persisted = get("saved-quote", location)
        if persisted.highlights != [QUOTE] or persisted.values["annotation-note"] != [NOTE]:
            raise RuntimeError("The persisted exact-quote view lost the opinion.")
        before_rejections = database.read_bytes()
        retry = client.post(
            page.actions["annotation-save"], data=fields, headers=ORIGIN, follow_redirects=False
        )
        changed = json.loads(fields["submission"])
        changed["note"] = "Different opinion under the same UUID must conflict."
        rejected = {
            "cross_origin": client.post(
                page.actions["annotation-save"],
                data=fields,
                headers={"origin": "https://external.invalid"},
            ).status_code,
            "duplicate_field": client.post(
                page.actions["annotation-save"],
                content=urlencode(fields) + "&submission=duplicate",
                headers={**ORIGIN, "content-type": "application/x-www-form-urlencoded"},
            ).status_code,
            "changed_uuid_payload": client.post(
                page.actions["annotation-save"],
                data={**fields, "submission": json.dumps(changed)},
                headers=ORIGIN,
            ).status_code,
        }
        annotations = container.evidence_annotations.list_annotations(RUN_ID)
        api_response = client.get(f"/review/runs/{RUN_ID}/annotations")
        api_response.raise_for_status()
        if (
            retry.status_code != 303
            or retry.headers["location"] != location
            or rejected
            != {"cross_origin": 403, "duplicate_field": 422, "changed_uuid_payload": 409}
            or database.read_bytes() != before_rejections
            or len(annotations.annotations) != 1
            or AnnotationPage.model_validate_json(api_response.content) != annotations
        ):
            raise RuntimeError("Retry, rejected writes or API/Python parity failed.")
        original = annotations.annotations[0]
        artifacts["annotations.json"] = api_response.content
        counts = {name: guard.call_count for name, guard in guards.items()}
        attempts = create.call_count
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE chunks SET text = 'CHANGED CURRENT CORPUS'")
    for stage in ("changed", "deleted"):
        if stage == "deleted":
            with closing(sqlite3.connect(database)) as connection, connection:
                for table in (
                    "entity_edges",
                    "entity_mentions",
                    "graph_chunks",
                    "chunks",
                    "documents",
                ):
                    connection.execute(f"DELETE FROM {table}")  # noqa: S608
        restarted = create_app(settings)
        before = database.read_bytes()
        with (
            TestClient(restarted, root_path="/review") as client,
            authoring_guards(restarted.state.container) as guards,
        ):
            response = client.post(opened.actions["annotation-draft"], data=draft, headers=ORIGIN)
            if response.content != artifacts["preview.html"]:
                raise RuntimeError("Frozen preview changed after corpus edit/deletion/restart.")
            response = client.post(
                page.actions["annotation-save"], data=fields, headers=ORIGIN, follow_redirects=False
            )
            if response.status_code != 303 or response.headers["location"] != location:
                raise RuntimeError("Frozen retry failed after restart.")
            persisted_response = client.get(location)
            if persisted_response.content != artifacts["saved-quote.html"]:
                raise RuntimeError("Persisted annotation changed after restart.")
            artifacts[stage + "-restart.html"] = persisted_response.content
            if restarted.state.container.evidence_annotations.list_annotations(
                RUN_ID
            ).annotations != [original]:
                raise RuntimeError("Retry changed original UUID, sequence, timestamp or payload.")
            counts.update({f"{stage}:{name}": guard.call_count for name, guard in guards.items()})
        if database.read_bytes() != before:
            raise RuntimeError("Read/preview/retry changed the restarted database.")
    if any(counts.values()) or container.event_log.list_events() != events or attempts != 3:
        raise RuntimeError("Unexpected authoring work or service calls occurred.")
    checks = {
        "fixture": FIXTURE,
        "run_id": RUN_ID,
        "quote": QUOTE,
        "note": NOTE,
        "start": start,
        "end": start + len(QUOTE),
        "first_occurrence": TEXT.index(QUOTE),
        "source_text_sha256": original.source_text_sha256,
        "annotation_id": str(original.annotation_id),
        "created_at": original.created_at.isoformat(),
        "annotations": 1,
        "save_status": saved.status_code,
        "retry_status": retry.status_code,
        "rejected_statuses": rejected,
        "create_service_attempts_before_restart": attempts,
        "get_preview_database_unchanged": True,
        "rejected_writes_database_unchanged": True,
        "retry_record_unchanged": True,
        "python_api_equal": True,
        "corpus_change_delete_restart_identical": True,
        "events_before": len(events),
        "events_after": len(container.event_log.list_events()),
        "forbidden_call_counts": counts,
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
    }
    transcript = (
        "Frozen annotation workbench / executed original synthetic workflow\n"
        "Setup: one manually recorded eight-event fixture; no model or retrieval setup.\n"
        "Mounted browser route: saved runs -> frozen source -> compose -> preview -> save.\n"
        f"Explicit second occurrence [{start}, {start + len(QUOTE)}): {QUOTE}\n"
        "Unicode code points, not bytes or UTF-16 units; one occurrence highlighted.\n"
        "GET and preview: no annotation/service writes; database bytes unchanged.\n"
        "Save: 303 to exact persisted quote; note absent from GET URLs.\n"
        "Unchanged retry: 303; one record with original UUID, sequence and timestamp.\n"
        "Cross-origin: 403; ambiguous form: 422; changed UUID payload: 409.\n"
        "Current corpus changed, deleted, app restarted: preview and saved view identical.\n"
        f"Python/API records equal; agent events unchanged: {len(events)}; forbidden calls: 0.\n"
        "GIF: four actual local browser screenshots, not a continuous recording.\n"
        "Synthetic human opinions are not scientific validation or authenticated reviews.\n"
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
    """Serve only the measured synthetic fixture; native browser writes are explicit."""
    database = output_dir / "annotation.sqlite3"
    checks = json.loads((output_dir / "checks.json").read_text(encoding="utf-8"))
    if (
        checks.get("fixture") != FIXTURE
        or database.is_symlink()
        or not database.is_file()
        or hashlib.sha256(database.read_bytes()).hexdigest() != checks["database_sha256"]
    ):
        raise ValueError("Expected an unchanged frozen-annotation synthetic demo directory.")
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
    parser.add_argument("--port", type=int, default=8776)
    arguments = parser.parse_args()
    if arguments.serve is not None:
        serve_demo(arguments.serve, arguments.port)
    else:
        print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
