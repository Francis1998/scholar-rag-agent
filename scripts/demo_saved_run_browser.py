"""Exercise native saved-run discovery and frozen evidence using original offline fixtures."""

import argparse
import hashlib
import json
import socket
import sqlite3
from contextlib import ExitStack, closing
from html import escape
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import httpx
import uvicorn
from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from agent.models import AgentState, StateTransition
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from scripts.demo_corpus_explorer import _no_work as _no_work
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from storage.evidence_export import SQLiteEvidenceReader
from storage.run_history import RunHistoryPage, SQLiteRunHistory

FIXTURE = "scholar-saved-run-browser-synthetic-v1"
FRAME_PAGES = ("history-first", "history-next", "frozen-answer", "frozen-source")
SOURCE = "synthetic:saved-run-browser"
LEGACY = "synthetic-legacy-done"
INTERRUPTED = "synthetic-interrupted"


def _run_ids(html: str) -> list[str]:
    return [
        parse_qs(urlsplit(url).query)["run_id"][0]
        for identifier, url in SearchPage(html).links.items()
        if identifier.startswith("inspect-run-")
    ]


def _get(client: TestClient, url: str, *, status: int = 200) -> httpx.Response:
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path.startswith(("/explore", "/runs")):
        raise RuntimeError("Saved-run navigation escaped local read-only routes.")
    response = client.get(url)
    if (
        response.status_code != status
        or response.headers["cache-control"] != "no-store"
        or response.headers["x-content-type-options"] != "nosniff"
    ):
        raise RuntimeError("A saved-run response lost its expected outcome or privacy headers.")
    return response


def _seed(client: TestClient, container: AppContainer) -> list[str]:
    for title, text in (
        (
            "Synthetic graph methods",
            "GraphRAG connects synthetic research evidence. This original workshop note is "
            "not a paper or scientific finding. Frozen passages retain literal source markup: "
            "<script>synthetic example only</script> & <b>not HTML</b>.",
        ),
        (
            "Synthetic graph limitations",
            "GraphRAG connects a second synthetic evidence passage. A citation association "
            "and token overlap do not establish scientific validity or independent support.",
        ),
    ):
        response = client.post(
            "/ingest/text", json={"title": title, "text": text, "source": SOURCE}
        )
        response.raise_for_status()
    run_ids = []
    for question in (
        "How does GraphRAG connect synthetic evidence?",
        "What are the limitations of the synthetic GraphRAG evidence?",
    ):
        response = client.post("/query", json={"query": question})
        response.raise_for_status()
        result = QueryResponse.model_validate_json(response.content).result
        if result.state != AgentState.DONE:
            raise RuntimeError("An offline fake-adapter demo query did not complete.")
        run_ids.append(result.run_id)
    for run_id, from_state, to_state, payload in (
        (LEGACY, AgentState.ANSWERING, AgentState.DONE, {}),
        (
            INTERRUPTED,
            AgentState.IDLE,
            AgentState.PLANNING,
            {
                "query": "Inserted incomplete trace fixture, not a live worker. "
                + "Synthetic long query preview. " * 15
            },
        ),
        (INTERRUPTED, AgentState.PLANNING, AgentState.REASONING, {}),
    ):
        container.event_log.append_transition(
            StateTransition(
                agent_id="synthetic-fixture",
                run_id=run_id,
                from_state=from_state,
                to_state=to_state,
                payload=payload,
            )
        )
    return run_ids


def run_demo(output_dir: Path) -> str:
    """Save actual responses and a synthetic DB for optional loopback browser inspection."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    path = output_dir / "history.sqlite3"
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
            completed = _seed(client, container)
            events = container.event_log.list_events()
            before = path.read_bytes()
            with browsing_guards(container) as guards:
                catalog = _get(client, "/explore?limit=2&title=graph&source=" + SOURCE)
                form = _get(client, SearchPage(catalog.text).links["saved-runs"])
                controls = SearchPage(form.text)
                submitted = {**controls.form_values, "limit": "2"}
                response = client.get(controls.form_action, params=submitted)
                first_url = response.url.raw_path.decode("ascii")
                first = _get(client, first_url)
                next_url = SearchPage(first.text).links["next-page"]
                second = _get(client, next_url)
                if _run_ids(first.text) != [INTERRUPTED, LEGACY] or _run_ids(second.text) != list(
                    reversed(completed)
                ):
                    raise RuntimeError("History pagination lost exact creation order.")
                if "Query preview truncated: only the first 300 characters" not in first.text:
                    raise RuntimeError("The synthetic long query did not expose its preview limit.")
                selected = client.get(controls.form_action, params={**submitted, "state": "DONE"})
                selected.raise_for_status()
                if _run_ids(selected.text) != [LEGACY, completed[1]]:
                    raise RuntimeError("The native exact recorded-state filter was not preserved.")
                legacy = _get(client, SearchPage(selected.text).links["inspect-run-1"], status=409)
                partial = _get(client, SearchPage(first.text).links["inspect-run-1"], status=409)
                if (
                    "no saved evidence snapshot" not in legacy.text
                    or "No terminal DONE" not in partial.text
                ):
                    raise RuntimeError("Legacy or interrupted fixtures were misrepresented.")
                filtered_next = _get(client, SearchPage(selected.text).links["next-page"])
                if _run_ids(filtered_next.text) != [completed[0]]:
                    raise RuntimeError("Filtered keyset continuation did not preserve DONE.")
                frozen_url = SearchPage(second.text).links["inspect-run-1"]
                # Inline browsing must use the bounded reader, not the unbounded event API.
                with patch.object(container.event_log, "list_events", side_effect=_no_work):
                    frozen = _get(client, frozen_url)
                reader = SearchPage(frozen.text)
                if reader.links["back-to-runs"] != next_url:
                    raise RuntimeError("The reader lost its exact history page.")
                if _get(client, reader.links["back-to-runs"]).content != second.content:
                    raise RuntimeError(
                        "Returning from frozen evidence changed the history selection."
                    )
                for fmt, extension in (("json", "json"), ("markdown", "md"), ("html", "html")):
                    artifacts["evidence." + extension] = _get(
                        client, reader.links["download-" + fmt]
                    ).content
                bundle = EvidenceBundle.model_validate_json(artifacts["evidence.json"])
                if (
                    bundle.run_id != completed[1]
                    or bundle.generation.provider != "fake"
                    or len(bundle.snapshot.sources) != 2
                    or SQLiteEvidenceReader(path).export(bundle.run_id) != bundle
                ):
                    raise RuntimeError(
                        "The saved evidence did not match the bounded Python reader."
                    )
                source = bundle.snapshot.sources[0]
                if (
                    f'id="source-{source.rank}"' not in frozen.text
                    or f'href="#source-{source.rank}"' not in frozen.text
                    or escape(source.chunk.text).replace("\r", "&#13;") not in frozen.text
                ):
                    raise RuntimeError("Frozen citation navigation lost its exact source passage.")
                history = _get(client, "/runs?limit=2")
                history_page = RunHistoryPage.model_validate_json(history.content)
                if history_page != SQLiteRunHistory(path).list_runs(limit=2) or [
                    run.run_id for run in history_page.runs
                ] != _run_ids(first.text):
                    raise RuntimeError("Browser, API and Python history disagree.")
                counts = {name: guard.call_count for name, guard in guards.items()}
            if path.read_bytes() != before or container.event_log.list_events() != events:
                raise RuntimeError("Browsing changed saved events or corpus bytes.")
        # Change/delete only the explicitly generated fixture, outside guarded page reads.
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE chunks SET text = 'CHANGED CURRENT CORPUS'")
        restarted = create_app(settings)
        with TestClient(restarted) as client, browsing_guards(restarted.state.container) as guards:
            if _get(client, frozen_url).content != frozen.content:
                raise RuntimeError("Current corpus edits changed frozen evidence.")
            counts.update({f"changed:{name}": guard.call_count for name, guard in guards.items()})
        with closing(sqlite3.connect(path)) as connection, connection:
            for table in ("entity_edges", "entity_mentions", "graph_chunks", "chunks", "documents"):
                connection.execute(f"DELETE FROM {table}")  # noqa: S608
            remaining = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        restarted = create_app(settings)
        before_restart_reads = path.read_bytes()
        with TestClient(restarted) as client, browsing_guards(restarted.state.container) as guards:
            after = _get(client, frozen_url)
            if after.content != frozen.content:
                raise RuntimeError("Corpus deletion or restart changed the saved reader.")
            for fmt, extension in (("json", "json"), ("markdown", "md"), ("html", "html")):
                if (
                    _get(client, reader.links["download-" + fmt]).content
                    != artifacts["evidence." + extension]
                ):
                    raise RuntimeError("Corpus deletion or restart changed a saved download.")
            if _get(client, first_url).content != first.content:
                raise RuntimeError("Restart changed history discovery.")
            counts.update({f"restart:{name}": guard.call_count for name, guard in guards.items()})
        if (
            any(counts.values())
            or any(guard.call_count for guard in network)
            or remaining != 0
            or path.read_bytes() != before_restart_reads
            or restarted.state.container.event_log.list_events() != events
        ):
            raise RuntimeError("Read-only demo guards or restart invariants failed.")
    for name, response in (
        ("catalog.html", catalog),
        ("history-first.html", first),
        ("history-next.html", second),
        ("filtered.html", selected),
        ("legacy.html", legacy),
        ("incomplete.html", partial),
        ("frozen.html", frozen),
        ("restarted.html", after),
        ("history.json", history),
    ):
        artifacts[name] = response.content
    urls = {
        "history-first": first_url,
        "history-next": next_url,
        "frozen-answer": frozen_url + "#answer",
        "frozen-source": frozen_url + f"#source-{source.rank}",
    }
    checks = {
        "fixture": FIXTURE,
        "history_pages": [_run_ids(first.text), _run_ids(second.text)],
        "recorded_done_page": _run_ids(selected.text),
        "legacy_status": legacy.status_code,
        "incomplete_status": partial.status_code,
        "query_preview_characters": len(history_page.runs[0].query_summary or ""),
        "query_truncated": history_page.runs[0].query_truncated,
        "frozen_run_id": bundle.run_id,
        "frozen_sources": len(bundle.snapshot.sources),
        "source_fragment": f"#source-{source.rank}",
        "api_python_browser_parity": True,
        "history_return_preserved": True,
        "corpus_change_delete_restart_identical": True,
        "all_downloads_identical": True,
        "database_bytes_unchanged_during_reads": True,
        "database_sha256": hashlib.sha256(before_restart_reads).hexdigest(),
        "events_before": len(events),
        "events_after": len(restarted.state.container.event_log.list_events()),
        "remaining_corpus_chunks": remaining,
        "forbidden_call_counts": counts,
        "socket_connection_calls": sum(guard.call_count for guard in network),
    }
    for filename, value in (("pages.json", urls), ("checks.json", checks)):
        artifacts[filename] = (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode()
    transcript = (
        "Saved run browser / original synthetic fixtures / offline\n"
        "Two real fake-adapter queries; one inserted legacy DONE and one incomplete trace.\n"
        f"History pages: {len(_run_ids(first.text))}, {len(_run_ids(second.text))}; "
        "exclusive cursor.\n"
        f"Query preview: {checks['query_preview_characters']} characters; truncation explicit.\n"
        f"Exact DONE filter includes legacy fixture; inspecting it returns {legacy.status_code}.\n"
        f"Interrupted trace returns {partial.status_code}; no worker-liveness assertion.\n"
        f"Validated reader: {len(bundle.snapshot.sources)} frozen passages; exact history return.\n"
        "API/Python/browser parity; JSON, Markdown and offline HTML downloads followed.\n"
        "Current corpus edited, then deleted; restarted reader and downloads byte-identical.\n"
        f"Remaining corpus chunks: {remaining}; "
        f"events before/after: {len(events)}/{checks['events_after']}.\n"
        "Forbidden generation/retrieval/reranking/writes/external HTTP/socket calls: 0.\n"
        "The synthetic event DB persists for loopback browser frames; no real papers or keys.\n"
        "Fake answers, recorded states and source associations are not scientific validation.\n"
    )
    artifacts["transcript.txt"] = transcript.encode()
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as output:
            output.write(content)
    return transcript


def serve_demo(output_dir: Path, port: int) -> None:
    """Serve only a generated synthetic fixture; never use ambient settings or credentials."""
    path = output_dir / "history.sqlite3"
    checks = json.loads((output_dir / "checks.json").read_text(encoding="utf-8"))
    if checks.get("fixture") != FIXTURE or path.is_symlink() or not path.is_file():
        raise ValueError("Expected an existing saved-run browser demo directory.")
    app = create_app(offline_settings(path))
    before = path.read_bytes()
    with browsing_guards(app.state.container):
        uvicorn.run(app, host="127.0.0.1", port=port, access_log=False)
    if path.read_bytes() != before:
        raise RuntimeError("The synthetic database changed during browser inspection.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", type=Path, help="Create new measured synthetic output.")
    action.add_argument("--serve", type=Path, help="Serve that existing fixture on loopback.")
    parser.add_argument("--port", type=int, default=8768)
    arguments = parser.parse_args()
    if arguments.serve is not None:
        serve_demo(arguments.serve, arguments.port)
    else:
        print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
