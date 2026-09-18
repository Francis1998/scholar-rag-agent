"""Discover persisted runs and follow evidence links using synthetic offline data."""

import argparse
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from agent.models import AgentRunResult, AgentState, StateTransition
from api.application import create_app
from api.dependencies import AppContainer
from scripts.demo_evidence_export import DEMO_QUERY, DEMO_TEXT, offline_settings
from storage.run_history import RunHistoryPage


def run_demo(output_dir: Path) -> str:
    """Exercise real API discovery after recreation, never overwriting saved artifacts."""
    names = (
        "history.sqlite3",
        "page-1.json",
        "page-2.json",
        "done.json",
        "events.json",
        "bundle.json",
        "bundle.md",
        "transcript.txt",
    )
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(
                f"Refusing to overwrite {output_dir / name}; choose a new directory."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "history.sqlite3"
    application = create_app(offline_settings(path))
    container: AppContainer = application.state.container
    with TestClient(application) as client:
        ingested = client.post(
            "/ingest/text",
            json={
                "title": "Synthetic history note",
                "source": "synthetic:history",
                "text": DEMO_TEXT,
            },
        )
        ingested.raise_for_status()
        completed_response = client.post("/query", json={"query": DEMO_QUERY})
        completed_response.raise_for_status()
        completed = AgentRunResult.model_validate(completed_response.json()["result"])
        if completed.state != AgentState.DONE:
            raise RuntimeError(f"Synthetic query failed: {completed.error}")
        before_restart = client.get(f"/runs/{completed.run_id}/export")
        before_restart.raise_for_status()

        with patch.object(
            container.llm,
            "generate",
            side_effect=RuntimeError("Deliberately injected synthetic offline failure"),
        ):
            failed_response = client.post(
                "/query", json={"query": "Demonstrate a synthetic generation failure"}
            )
        failed_response.raise_for_status()
        failed = AgentRunResult.model_validate(failed_response.json()["result"])
        if failed.state != AgentState.ERROR:
            raise RuntimeError("The injected offline failure did not produce a recorded ERROR.")

        # A deliberately partial fixture, not an active/background/resumable job.
        for source, target in (
            (AgentState.IDLE, AgentState.PLANNING),
            (AgentState.PLANNING, AgentState.RETRIEVING),
            (AgentState.RETRIEVING, AgentState.REASONING),
        ):
            container.event_log.append_transition(
                StateTransition(
                    agent_id="synthetic-fixture",
                    run_id="synthetic-interrupted-fixture",
                    from_state=source,
                    to_state=target,
                    payload={"query": "Inspect this synthetic interrupted trace"}
                    if target == AgentState.PLANNING
                    else {},
                )
            )

        application.state.container = AppContainer(offline_settings(path))
        first_response = client.get("/runs", params={"limit": 2})
        first_response.raise_for_status()
        first = RunHistoryPage.model_validate_json(first_response.content)
        if first.next_cursor is None:
            raise RuntimeError("The first discovery page did not provide its continuation.")
        second_response = client.get("/runs", params={"limit": 2, "cursor": first.next_cursor})
        second_response.raise_for_status()
        second = RunHistoryPage.model_validate_json(second_response.content)
        done_response = client.get("/runs", params={"state": "DONE"})
        done_response.raise_for_status()
        done = RunHistoryPage.model_validate_json(done_response.content)
        discovered = [*first.runs, *second.runs]
        if (
            [run.recorded_state for run in discovered]
            != [AgentState.REASONING, AgentState.ERROR, AgentState.DONE]
            or len({run.run_id for run in discovered}) != 3
            or second.next_cursor is not None
            or done.runs != second.runs
        ):
            raise RuntimeError("Discovery lost, duplicated, or misclassified synthetic runs.")
        recovered = second.runs[0]
        if recovered.events_url is None or recovered.export_url is None:
            raise RuntimeError("The discovered completed run has no navigation links.")
        events = client.get(recovered.events_url)
        exported = client.get(recovered.export_url)
        markdown = client.get(recovered.export_url, params={"format": "markdown"})
        events.raise_for_status()
        exported.raise_for_status()
        markdown.raise_for_status()
        bundle = EvidenceBundle.model_validate_json(exported.content)
        if (
            exported.content != before_restart.content
            or bundle.query != recovered.query_summary
            or bundle.generation.provider != "fake"
            or bundle.run_id != recovered.run_id
            or len(events.json()) != recovered.event_count
        ):
            raise RuntimeError("The discovered export did not retain its exact offline evidence.")

    for name, response in (
        ("page-1.json", first_response),
        ("page-2.json", second_response),
        ("done.json", done_response),
        ("events.json", events),
        ("bundle.json", exported),
        ("bundle.md", markdown),
    ):
        (output_dir / name).write_bytes(response.content)
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        "1. Produce synthetic offline run records",
                        f"POST /ingest/text -> {ingested.status_code}",
                        f"POST /query -> {completed_response.status_code}; state={completed.state}",
                        f"Injected offline failure -> {failed_response.status_code}; "
                        f"state={failed.state}",
                        "Append REASONING trace: fixture, not an active job.",
                        f"Saved completed provider: {bundle.generation.provider}",
                    ]
                ),
                "\n".join(
                    [
                        "2. Reopen SQLite and discover forgotten IDs",
                        "AppContainer recreated against the same history.sqlite3.",
                        f"GET /runs?limit=2 -> {first_response.status_code}",
                        "Recorded states: "
                        + ", ".join(str(run.recorded_state) for run in first.runs),
                        f"next_cursor={first.next_cursor}; an exclusive first-event ID",
                        f"Newest query: {first.runs[0].query_summary}",
                    ]
                ),
                "\n".join(
                    [
                        "3. Continue by creation order, then filter",
                        f"GET /runs?limit=2&cursor={first.next_cursor} -> "
                        f"{second_response.status_code}",
                        f"Second page: {recovered.recorded_state}; "
                        f"next_cursor={second.next_cursor}",
                        f"No duplicates: {len(discovered)} discovered runs",
                        f"GET /runs?state=DONE -> {done_response.status_code}; "
                        f"count={len(done.runs)}",
                        "Recorded state is not liveness; "
                        "pagination is not a frozen state snapshot.",
                    ]
                ),
                "\n".join(
                    [
                        "4. Follow the discovered evidence links",
                        f"GET /runs/<discovered-ID>/events -> {events.status_code}; "
                        f"events={recovered.event_count}",
                        f"GET /runs/<discovered-ID>/export -> {exported.status_code}",
                        f"Markdown export -> {markdown.status_code}; "
                        f"provider={bundle.generation.provider}",
                        "JSON export after restart: byte-identical",
                        "Saved: pages, filtered results, events, JSON/Markdown evidence, SQLite.",
                        "Synthetic/offline only: no research UI or scientific quality claim.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def main() -> None:
    """Run without a server, real paper, model credentials, or external calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("run-history-demo"))
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
