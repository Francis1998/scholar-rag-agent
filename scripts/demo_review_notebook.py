"""Measure the native human-review notebook using a new, fake-only synthetic saved run."""

import argparse
import hashlib
import json
import socket
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from html import escape
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import uvicorn
from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from agent.review_models import ReviewDecision, ReviewHistoryPage
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from scripts.demo_corpus_explorer import _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from scripts.demo_saved_run_browser import _get
from storage.evidence_annotations import AnnotationPage

FIXTURE = "scholar-human-review-notebook-synthetic-v1"
FRAME_PAGES = ("saved-runs", "answer-reviews", "evidence-annotations", "exact-quote")
QUOTE = "GraphRAG connects synthetic evidence"
TEXT = (
    "Original synthetic workshop note, not a publication. Cafe\u0301 \U0001f52c. "
    f"{QUOTE}. A repeated sentence is not independent support. Again: {QUOTE}. "
    "Inspect only the second occurrence when following its annotation. "
    "Literal source markup: <script>synthetic text, never executed</script> & <b>not HTML</b>."
)


@contextmanager
def notebook_guards(container: AppContainer) -> Iterator[dict[str, Mock]]:
    """Fail on model, corpus, opinion or agent-event work during browser reads."""
    with browsing_guards(container) as guards, ExitStack() as stack:
        for target, method in (
            (container.event_log, "list_events"),
            (container.event_log, "append_transition"),
            (container.evidence_exporter, "export"),
            (container.answer_reviews, "create"),
            (container.answer_reviews._store, "append"),
            (container.evidence_annotations, "create"),
        ):
            guards[f"{type(target).__name__}.{method}"] = stack.enter_context(
                patch.object(target, method, side_effect=_no_work)
            )
        yield guards


def _seed(client: TestClient, container: AppContainer) -> EvidenceBundle:
    response = client.post(
        "/ingest/text",
        json={"title": "Synthetic notebook source", "text": TEXT, "source": "synthetic:notebook"},
    )
    response.raise_for_status()
    response = client.post("/query", json={"query": "What does GraphRAG connect?"})
    response.raise_for_status()
    result = QueryResponse.model_validate_json(response.content).result
    if result.state != "DONE":
        raise RuntimeError("The fake-only synthetic setup did not complete.")
    bundle = container.saved_run_evidence.export(result.run_id)
    if bundle.generation.provider != "fake" or len(bundle.snapshot.sources) != 1:
        raise RuntimeError("Expected one original synthetic source and the fake adapter.")
    source = bundle.snapshot.sources[0]
    for decision, comment in (
        ("accepted", "Accepted only as a software demonstration, not as scientific validation."),
        (
            "needs_revision",
            "Human opinion: distinguish repeated wording from independent evidence.",
        ),
        (
            "rejected",
            "Do not use this synthetic answer as a scientific result. <b>Literal comment</b>.",
        ),
    ):
        response = client.post(
            f"/runs/{bundle.run_id}/reviews",
            json={
                "review_id": str(uuid4()),
                "decision": decision,
                "comment": comment,
                "cited_chunk_ids": [source.chunk.chunk_id],
            },
        )
        if response.status_code != 201:
            raise RuntimeError("The synthetic answer review was not recorded.")
    text = source.chunk.text
    for start, note in (
        (text.index(QUOTE), "First occurrence: retained separately, not overwritten."),
        (text.rindex(QUOTE), "Inspect this second occurrence. Exact quote, not proof of support."),
    ):
        response = client.post(
            f"/runs/{bundle.run_id}/annotations",
            json={
                "annotation_id": str(uuid4()),
                "document_id": source.chunk.document_id,
                "chunk_id": source.chunk.chunk_id,
                "source_text_sha256": source.text_sha256,
                "start": start,
                "end": start + len(QUOTE),
                "quote": QUOTE,
                "note": note,
            },
        )
        if response.status_code != 201:
            raise RuntimeError("The synthetic exact-quote annotation was not recorded.")
    return bundle


def run_demo(output_dir: Path) -> str:
    """Persist real HTML and measurements early; optional browser capture uses this same DB."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    path = output_dir / "notebook.sqlite3"
    settings = offline_settings(path)
    artifacts: dict[str, bytes] = {}
    with ExitStack() as stack:
        network = [
            stack.enter_context(patch.object(target, method, side_effect=_no_work))
            for target, method in ((socket, "create_connection"), (socket.socket, "connect"))
        ]
        app = create_app(settings)
        container: AppContainer = app.state.container
        with TestClient(app) as client:
            bundle = _seed(client, container)
            reviews_response = _get(client, f"/runs/{bundle.run_id}/reviews")
            reviews = ReviewHistoryPage.model_validate_json(reviews_response.content)
            annotations_response = _get(client, f"/runs/{bundle.run_id}/annotations")
            annotations = AnnotationPage.model_validate_json(annotations_response.content)
            events = container.event_log.list_events()
            before = path.read_bytes()
            history_url = "/explore/runs?limit=1&state=DONE&catalog_limit=2"
            with notebook_guards(container) as guards:
                history = _get(client, history_url)
                notebook_url = SearchPage(history.text).links["review-notebook-1"]
                notebook_url += "&review_limit=1&annotation_limit=1"
                notebook = _get(client, notebook_url)
                page = SearchPage(notebook.text)
                if [review.decision for review in reviews.reviews] != list(
                    reversed(ReviewDecision)
                ):
                    raise RuntimeError("Not all three recorded review decisions were retained.")
                if len(annotations.annotations) != 2:
                    raise RuntimeError("The exact-span opinion history was not append-only.")
                latest = annotations.annotations[0]
                selected_url = page.links[f"annotation-source-{latest.sequence}"]
                selected = _get(client, selected_url)
                selected_page = SearchPage(selected.text)
                if (
                    latest.start != bundle.snapshot.sources[0].chunk.text.rindex(QUOTE)
                    or selected_page.highlights != [QUOTE]
                    or selected_page.passages != [bundle.snapshot.sources[0].chunk.text]
                    or selected_page.links["back-to-notebook"] != notebook_url
                    or page.links["back-to-runs"] != history_url
                ):
                    raise RuntimeError("Exact frozen selection or return navigation was lost.")
                for name, response in (("notebook", notebook), ("exact-quote", selected)):
                    artifacts[name + ".html"] = response.content
                older_reviews = _get(client, page.links["reviews-next-page"])
                older_page = SearchPage(older_reviews.text)
                older_both_url = older_page.links["annotations-next-page"]
                older_both = _get(client, older_both_url)
                if (
                    escape(reviews.reviews[1].comment) not in older_both.text
                    or escape(annotations.annotations[1].note) not in older_both.text
                    or "not one shared snapshot" not in notebook.text
                ):
                    raise RuntimeError("Independent history continuation lost an exact opinion.")
                artifacts["older-histories.html"] = older_both.content
                counts = {name: guard.call_count for name, guard in guards.items()}
            if path.read_bytes() != before or container.event_log.list_events() != events:
                raise RuntimeError("Notebook reads changed saved events or database bytes.")
        # Only this newly generated synthetic corpus is changed; retained opinions/events stay.
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE chunks SET text = 'CHANGED CURRENT CORPUS'")
        for stage in ("changed", "deleted"):
            if stage == "deleted":
                with closing(sqlite3.connect(path)) as connection, connection:
                    for table in (
                        "entity_edges",
                        "entity_mentions",
                        "graph_chunks",
                        "chunks",
                        "documents",
                    ):
                        connection.execute(f"DELETE FROM {table}")  # noqa: S608
                    remaining = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            restarted = create_app(settings)
            before = path.read_bytes()
            with (
                TestClient(restarted) as client,
                notebook_guards(restarted.state.container) as guards,
            ):
                for url, expected in (
                    (notebook_url, notebook.content),
                    (selected_url, selected.content),
                    (older_both_url, older_both.content),
                ):
                    if _get(client, url).content != expected:
                        raise RuntimeError(
                            "A corpus change or restart changed frozen notebook HTML."
                        )
                counts.update(
                    {f"{stage}:{name}": guard.call_count for name, guard in guards.items()}
                )
            if path.read_bytes() != before:
                raise RuntimeError("Restarted notebook reads changed the synthetic database.")
        if (
            any(counts.values())
            or any(guard.call_count for guard in network)
            or remaining != 0
            or restarted.state.container.event_log.list_events() != events
        ):
            raise RuntimeError("Read-only notebook invariants failed.")
    urls = {
        "saved-runs": history_url,
        "answer-reviews": notebook_url + "#reviews",
        "evidence-annotations": notebook_url + "#annotations",
        "exact-quote": selected_url,
    }
    checks = {
        "fixture": FIXTURE,
        "run_id": bundle.run_id,
        "provider": bundle.generation.provider,
        "review_decisions": [review.decision.value for review in reviews.reviews],
        "annotations": len(annotations.annotations),
        "selected_annotation_id": str(latest.annotation_id),
        "quote": latest.quote,
        "start": latest.start,
        "end": latest.end,
        "first_occurrence": bundle.snapshot.sources[0].chunk.text.index(QUOTE),
        "source_text_sha256": latest.source_text_sha256,
        "history_return_preserved": True,
        "independent_pagination": True,
        "corpus_change_delete_restart_identical": True,
        "database_bytes_unchanged_during_reads": True,
        "database_sha256": hashlib.sha256(before).hexdigest(),
        "events_before": len(events),
        "events_after": len(restarted.state.container.event_log.list_events()),
        "remaining_corpus_chunks": remaining,
        "forbidden_call_counts": counts,
        "socket_connection_calls": sum(guard.call_count for guard in network),
    }
    transcript = (
        "Human-review notebook / actual executed synthetic offline workflow\n"
        "Setup: one real fake-adapter query, three review decisions and two exact-span notes.\n"
        "Browser: saved runs -> notebook -> independent history pages -> exact frozen quote.\n"
        f"Selected second occurrence [{latest.start}, {latest.end}): {latest.quote}\n"
        "Full frozen passage and exact opinion text retained; no guessed occurrence.\n"
        "State/cursor return navigation preserved; histories are not one shared snapshot.\n"
        "Current corpus changed, deleted, then app restarted: notebook/source HTML identical.\n"
        f"Events before/after reads: {len(events)}/{checks['events_after']}; "
        f"corpus chunks: {remaining}.\n"
        "Forbidden generation/retrieval/opinion writes/corpus writes/events/network calls: 0.\n"
        "Synthetic opinions are not scientific validation or authenticated peer review.\n"
    )
    artifacts.update(
        {
            "saved-runs.html": history.content,
            "answer-reviews.html": notebook.content,
            "evidence-annotations.html": notebook.content,
            "reviews.json": reviews_response.content,
            "annotations.json": annotations_response.content,
            "transcript.txt": transcript.encode("utf-8"),
        }
    )
    for name, value in (("pages.json", urls), ("checks.json", checks)):
        artifacts[name] = (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as output:
            output.write(content)
    return transcript


def serve_demo(output_dir: Path, port: int) -> None:
    """Serve only the generated fake-only fixture on loopback, with forbidden work guarded."""
    path = output_dir / "notebook.sqlite3"
    checks = json.loads((output_dir / "checks.json").read_text(encoding="utf-8"))
    if checks.get("fixture") != FIXTURE or path.is_symlink() or not path.is_file():
        raise ValueError("Expected an existing human-review notebook demo directory.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != checks["database_sha256"]:
        raise ValueError("The synthetic notebook database changed after its measured demo.")
    app = create_app(offline_settings(path))
    before = path.read_bytes()
    with notebook_guards(app.state.container):
        uvicorn.run(app, host="127.0.0.1", port=port, access_log=False)
    if path.read_bytes() != before:
        raise RuntimeError("The synthetic database changed during browser inspection.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", type=Path, help="Create new measured synthetic output.")
    action.add_argument("--serve", type=Path, help="Serve that existing fixture on loopback.")
    parser.add_argument("--port", type=int, default=8770)
    arguments = parser.parse_args()
    if arguments.serve is not None:
        serve_demo(arguments.serve, arguments.port)
    else:
        print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
