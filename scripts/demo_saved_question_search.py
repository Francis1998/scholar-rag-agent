"""Measure original-question search with synthetic records; optional illustration, not footage."""

import argparse
import hashlib
import json
import socket
import sqlite3
from contextlib import ExitStack, closing
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.models import AgentState, StateTransition
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from scripts.demo_corpus_explorer import _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from storage.run_history import RunHistoryPage, SQLiteRunHistory

NEEDLE = "Literal 100%_[x]"
LONG_QUESTION = "Original synthetic question context. " * 12 + NEEDLE + "?"
PREFIX = "/lab"
PANEL_TITLES = (
    "1. Search the complete original question",
    "2. Filter before creation-order pagination",
    "3. Keep the native return selection",
    "4. Restart without generation or writes",
)


def _mounted(container_app: FastAPI) -> FastAPI:
    application = FastAPI()
    application.mount(PREFIX, container_app)
    return application


def _get(client: TestClient, url: str, **params: str | int) -> httpx.Response:
    parts = urlsplit(url)
    if parts.netloc or parts.scheme or not parts.path.startswith(PREFIX + "/"):
        raise RuntimeError("Demo navigation lost its local mount prefix.")
    response = client.get(url, params=params) if params else client.get(url)
    response.raise_for_status()
    if (
        response.headers.get("cache-control") != "no-store"
        or response.headers.get("x-content-type-options") != "nosniff"
    ):
        raise RuntimeError("Demo read lost privacy headers.")
    return response


def _ids(response: httpx.Response) -> list[str]:
    return [
        parse_qs(urlsplit(url).query)["run_id"][0]
        for name, url in SearchPage(response.text).links.items()
        if name.startswith("inspect-run-")
    ]


def _seed(client: TestClient, container: AppContainer) -> list[str]:
    response = client.post(
        PREFIX + "/ingest/text",
        json={
            "title": "Synthetic question-search note",
            "source": "synthetic:question-search",
            "text": "Original fictional workshop note, not a scientific finding. "
            "Saved questions help locate an earlier record; lexical matches are not relevance.",
        },
    )
    response.raise_for_status()
    completed = []
    for question in (LONG_QUESTION, "Another synthetic question: " + NEEDLE):
        response = client.post(PREFIX + "/query", json={"query": question})
        response.raise_for_status()
        run = QueryResponse.model_validate_json(response.content).result
        if run.state != AgentState.DONE:
            raise RuntimeError("A synthetic fake-adapter query did not complete.")
        completed.append(run.run_id)
    for run_id, before, after, payload in (
        ("synthetic-partial", AgentState.IDLE, AgentState.PLANNING, {"query": LONG_QUESTION}),
        ("synthetic-partial", AgentState.PLANNING, AgentState.REASONING, {}),
        ("synthetic-no-query", AgentState.ANSWERING, AgentState.DONE, {}),
        ("synthetic-empty", AgentState.IDLE, AgentState.PLANNING, {"query": ""}),
        ("synthetic-answer-only", AgentState.IDLE, AgentState.PLANNING, {"query": "Unrelated"}),
    ):
        container.event_log.append_transition(
            StateTransition(
                agent_id="synthetic-fixture",
                run_id=run_id,
                from_state=before,
                to_state=after,
                payload=payload,
            )
        )
    container.event_log.append_event(
        "synthetic-fixture", "synthetic-answer-only", "generation_record", {"answer": NEEDLE}
    )
    return completed


def run_demo(output_dir: Path) -> str:
    """Persist real measured responses; all writes are setup or requested demo artifacts."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    path = output_dir / "history.sqlite3"
    settings = offline_settings(path)
    artifacts: dict[str, bytes] = {}
    with ExitStack() as stack:
        network = [
            stack.enter_context(patch.object(target, method, side_effect=_no_work))
            for target, method in (
                (socket, "create_connection"),
                (socket.socket, "connect"),
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
            )
        ]
        app = create_app(settings)
        container: AppContainer = app.state.container
        with TestClient(_mounted(app)) as client:
            completed = _seed(client, container)
            events_before = container.event_log.list_events()
            bytes_before = path.read_bytes()
            with closing(sqlite3.connect(path)) as connection:
                schema_before = connection.execute(
                    "SELECT sql FROM sqlite_master ORDER BY name"
                ).fetchall()
            with (
                browsing_guards(container) as guards,
                patch.object(container.event_log, "list_events", side_effect=_no_work),
                patch.object(container.document_catalog, "list_documents", side_effect=_no_work),
                patch.object(container.graph_store, "chunks_for_entities", side_effect=_no_work),
            ):
                all_runs = _get(client, PREFIX + "/runs")
                matches = _get(client, PREFIX + "/runs", question=NEEDLE)
                matched = RunHistoryPage.model_validate_json(matches.content)
                if [run.run_id for run in matched.runs] != [
                    "synthetic-partial",
                    *reversed(completed),
                ]:
                    raise RuntimeError("Full-question search included an answer or missing query.")
                if matched != SQLiteRunHistory(path).list_runs(question=NEEDLE):
                    raise RuntimeError("Python and HTTP question search disagree.")
                form = _get(
                    client,
                    PREFIX + "/explore/runs",
                    catalog_limit=2,
                    source="synthetic:question-search",
                )
                controls = SearchPage(form.text)
                first = _get(
                    client,
                    controls.form_action,
                    **{**controls.form_values, "question": NEEDLE, "state": "DONE", "limit": 1},
                )
                next_url = SearchPage(first.text).links["next-page"]
                second = _get(client, next_url)
                if _ids(first) != [completed[1]] or _ids(second) != [completed[0]]:
                    raise RuntimeError("Native filtered keyset pages lost exact order.")
                if "question-match-beyond-preview" not in second.text:
                    raise RuntimeError("The beyond-preview match was not explained.")
                second_fields = parse_qs(urlsplit(next_url).query)
                page = _get(
                    client,
                    PREFIX + "/runs",
                    question=NEEDLE,
                    state="DONE",
                    limit=1,
                    cursor=second_fields["cursor"][0],
                )
                typed_page = RunHistoryPage.model_validate_json(page.content)
                if (
                    [run.run_id for run in typed_page.runs] != _ids(second)
                    or typed_page.next_cursor is not None
                    or NEEDLE in (typed_page.runs[0].query_summary or "")
                    or not typed_page.runs[0].query_truncated
                ):
                    raise RuntimeError("Full-question filtering or preview bounds changed.")
                lower = _get(client, PREFIX + "/runs", question=NEEDLE.lower())
                regex = _get(client, PREFIX + "/runs", question=".*")
                if lower.json()["runs"] or regex.json()["runs"]:
                    raise RuntimeError("Question search became case-folded or pattern-based.")
                invalid = client.get(PREFIX + "/runs", params={"question": ""})
                if invalid.status_code != 422:
                    raise RuntimeError("Explicit empty API search did not fail validation.")
                if (
                    _get(client, PREFIX + "/explore/runs", question="").content
                    != _get(client, PREFIX + "/explore/runs").content
                ):
                    raise RuntimeError("Empty native search did not mean omission.")
                reader_url = SearchPage(second.text).links["inspect-run-1"]
                reader = _get(client, reader_url)
                links = SearchPage(reader.text).links
                notebook = _get(client, links["review-notebook"])
                review = _get(client, links["review-answer"])
                annotation = _get(client, links["annotate-source-1"])
                chooser = _get(client, SearchPage(second.text).links["compare-run-1"])
                selected = SearchPage(chooser.text)
                comparison = _get(
                    client,
                    selected.form_action,
                    **{**selected.form_values, "candidate": json.dumps(completed[1])},
                )
                for response in (reader, notebook, review, annotation, chooser, comparison):
                    if SearchPage(response.text).links["back-to-runs"] != next_url:
                        raise RuntimeError("A saved-run consumer discarded the question selection.")
                if _get(client, links["back-to-runs"]).content != second.content:
                    raise RuntimeError("Returning from evidence changed the filtered history.")
                counts = {name: guard.call_count for name, guard in guards.items()}
            if path.read_bytes() != bytes_before:
                raise RuntimeError("Search or native return navigation changed the database.")

        restarted = create_app(settings)
        after_startup = path.read_bytes()
        with (
            TestClient(_mounted(restarted)) as client,
            browsing_guards(restarted.state.container) as guards,
        ):
            after = _get(client, next_url)
            after_reader = _get(client, reader_url)
            if (
                after.content != second.content
                or after_reader.content != reader.content
                or _get(client, PREFIX + "/runs", question=NEEDLE).content != matches.content
            ):
                raise RuntimeError("Restart changed saved-question discovery or frozen evidence.")
            counts.update({f"restart:{name}": guard.call_count for name, guard in guards.items()})
        events_after = restarted.state.container.event_log.list_events()
        with closing(sqlite3.connect(path)) as connection:
            schema_after = connection.execute(
                "SELECT sql FROM sqlite_master ORDER BY name"
            ).fetchall()
        if (
            any(counts.values())
            or any(guard.call_count for guard in network)
            or events_before != events_after
            or path.read_bytes() != after_startup
            or bytes_before != after_startup
            or schema_before != schema_after
        ):
            raise RuntimeError("A read-only, no-network or restart invariant failed.")

    checks = {
        "fixture": "synthetic-saved-question-search-v1",
        "search": NEEDLE,
        "all_runs": len(all_runs.json()["runs"]),
        "question_matches": len(matched.runs),
        "done_pages": [_ids(first), _ids(second)],
        "long_question_match_start": LONG_QUESTION.index(NEEDLE),
        "preview_characters": len(typed_page.runs[0].query_summary or ""),
        "lowercase_matches": len(lower.json()["runs"]),
        "regex_like_matches": len(regex.json()["runs"]),
        "empty_api_status": invalid.status_code,
        "native_return_consumers": 6,
        "mount_prefix": PREFIX,
        "restart_identical": True,
        "schema_and_database_unchanged": True,
        "database_sha256": hashlib.sha256(after_startup).hexdigest(),
        "events_before": len(events_before),
        "events_after": len(events_after),
        "forbidden_call_counts": counts,
        "network_connection_calls": sum(guard.call_count for guard in network),
    }
    transcript = (
        "\n\n".join(
            (
                PANEL_TITLES[0]
                + "\nExecuted synthetic API/Python/native HTML demo; not scientific evidence.\n"
                + f"Saved runs: {checks['all_runs']}; question matches: {len(matched.runs)}.\n"
                + f"Literal needle: {NEEDLE}; case-sensitive, no pattern syntax.\n"
                + f"Older match starts at code point {LONG_QUESTION.index(NEEDLE)}; preview: "
                + f"{checks['preview_characters']} code points.\n"
                + "Answer-only, empty and missing-query fixtures are not matches.",
                PANEL_TITLES[1]
                + f"\nDONE pages: {len(_ids(first))} + {len(_ids(second))}; no duplicates.\n"
                + "Both filters precede the exclusive first-event-ID continuation.\n"
                + f"Lowercase matches: {checks['lowercase_matches']}; "
                + f"regex-like .* matches: {checks['regex_like_matches']}.\n"
                + f"Empty API question: HTTP {invalid.status_code}; empty native field clears it.\n"
                + "Python, JSON and native result IDs agree.",
                PANEL_TITLES[2]
                + "\nSubmitted the actual native form and followed its Next page link.\n"
                + "The long-question page explains the match beyond its preview.\n"
                + "Reader, notebook, review, annotation, chooser and comparison keep selection.\n"
                + f"Exact question, state, cursor and catalog context survive under {PREFIX}.\n"
                + "No review or annotation was submitted; those are separate explicit writes.",
                PANEL_TITLES[3]
                + "\nRecreated the application on the same saved SQLite database.\n"
                + "Search and frozen evidence responses remain byte-identical.\n"
                + f"Events before/after: {len(events_before)}/{len(events_after)}; "
                + "schema and database bytes unchanged.\n"
                + f"Guarded runtime/network calls during reads: {sum(counts.values())}/"
                + f"{checks['network_connection_calls']}.\n"
                + "Two fake-adapter queries and four inserted traces preceded measurements.",
            )
        )
        + "\n"
    )
    artifacts["transcript.txt"] = transcript.encode()
    artifacts["checks.json"] = (json.dumps(checks, indent=2) + "\n").encode()
    for name, response in (
        ("matches.json", matches),
        ("done-next.json", page),
        ("history-first.html", first),
        ("history-next.html", second),
        ("reader.html", reader),
        ("comparison.html", comparison),
        ("notebook.html", notebook),
        ("review.html", review),
        ("annotation.html", annotation),
        ("restarted.html", after),
    ):
        artifacts[name] = response.content
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as destination:
            destination.write(content)
    return transcript


def create_preview(output_dir: Path) -> None:
    """Illustrate only the measured transcript using the repository's existing renderer."""
    from PIL import Image

    from scripts.create_evidence_gif import render_panels

    gif = output_dir / "saved-question-search.gif"
    still = output_dir / "saved-question-search.png"
    for path in (gif, still):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new directory.")
    transcript = (output_dir / "transcript.txt").read_text(encoding="utf-8")
    panels = transcript.strip().split("\n\n")
    if len(panels) != 4 or any(
        panel.splitlines()[0] != title for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected the four measured saved-question search panels.")
    render_panels(panels, gif, banner="SCHOLAR RAG / SAVED ORIGINAL-QUESTION SEARCH")
    with Image.open(gif) as image, BytesIO() as preview:
        image.convert("RGB").save(preview, format="PNG")
        with still.open("xb") as output:
            output.write(preview.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--illustrate", action="store_true", help="Render measured GIF and still.")
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")
    if arguments.illustrate:
        create_preview(arguments.output_dir)


if __name__ == "__main__":
    main()
