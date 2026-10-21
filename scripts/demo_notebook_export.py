"""Execute a fake-only notebook export, then remove its synthetic corpus and database."""

import argparse
import hashlib
import json
import socket
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from api.application import create_app
from scripts.demo_corpus_explorer import _no_work
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from scripts.demo_review_notebook import QUOTE, _seed, notebook_guards
from storage.review_notebook import ReviewNotebook

PANEL_TITLES = (
    "1. Save one answer and separate human opinions",
    "2. Download the whole notebook, not a page",
    "3. Keep the exact quoted occurrence",
    "4. Read the same artifact after deletion and restart",
)
OUTPUT_NAMES = (
    "notebook.json",
    "notebook.md",
    "reviews-first-page.json",
    "annotations-first-page.json",
    "checks.json",
    "transcript.txt",
)
RECORD_COUNT = 23


def _save(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)


def run_demo(output_dir: Path) -> str:
    """Persist actual downloads first; all setup writes target a new temporary fixture only."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    with TemporaryDirectory(prefix="scholar-notebook-export-") as temporary, ExitStack() as stack:
        network = [
            stack.enter_context(patch.object(target, method, side_effect=_no_work))
            for target, method in ((socket, "create_connection"), (socket.socket, "connect"))
        ]
        path = Path(temporary) / "synthetic.sqlite3"
        settings = offline_settings(path)
        app = create_app(settings)
        container = app.state.container
        with TestClient(app) as client:
            bundle = _seed(client, container)
            source = bundle.snapshot.sources[0]
            for number in range(3, RECORD_COUNT):
                response = client.post(
                    f"/runs/{bundle.run_id}/reviews",
                    json={
                        "review_id": str(uuid4()),
                        "decision": ("accepted", "needs_revision", "rejected")[number % 3],
                        "comment": f"Synthetic opinion {number + 1}: "
                        "repetition is not independent evidence.",
                        "cited_chunk_ids": [source.chunk.chunk_id],
                    },
                )
                if response.status_code != 201:
                    raise RuntimeError("The synthetic review setup failed.")
            for number in range(2, RECORD_COUNT):
                start = source.chunk.text.rindex(QUOTE)
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
                        "note": f"Synthetic note {number + 1}: "
                        "inspect the second occurrence, not a verdict.",
                    },
                )
                if response.status_code != 201:
                    raise RuntimeError("The synthetic annotation setup failed.")
            first_pages = {}
            for kind in ("reviews", "annotations"):
                page = client.get(f"/runs/{bundle.run_id}/{kind}")
                page.raise_for_status()
                first_pages[kind] = page.json()
                _save(output_dir / f"{kind}-first-page.json", page.content)
            events = container.event_log.list_events()
            before = path.read_bytes()
            downloads = {}
            with notebook_guards(container) as guards:
                page = client.get(
                    "/explore/run/notebook",
                    params={"run_id": bundle.run_id, "review_limit": 1, "annotation_limit": 1},
                )
                page.raise_for_status()
                links = SearchPage(page.text).links
                for format, extension in (("json", "json"), ("markdown", "md")):
                    response = client.get(links["download-notebook-" + format])
                    response.raise_for_status()
                    downloads[format] = response.content
                    _save(output_dir / f"notebook.{extension}", response.content)
                counts.update({name: guard.call_count for name, guard in guards.items()})
            notebook = ReviewNotebook.model_validate_json(downloads["json"])
            if (
                notebook.evidence != bundle
                or len(notebook.reviews) != RECORD_COUNT
                or len(notebook.annotations) != RECORD_COUNT
                or path.read_bytes() != before
                or container.event_log.list_events() != events
            ):
                raise RuntimeError("Complete frozen notebook or read-only invariants failed.")
        for stage in ("changed", "deleted"):
            with closing(sqlite3.connect(path)) as connection, connection:
                if stage == "changed":
                    connection.execute("UPDATE chunks SET text = 'REPLACED CURRENT CORPUS'")
                else:
                    for table in (
                        "entity_edges",
                        "entity_mentions",
                        "graph_chunks",
                        "chunks",
                        "documents",
                    ):
                        connection.execute(f"DELETE FROM {table}")  # noqa: S608
            restarted = create_app(settings)
            before = path.read_bytes()
            with (
                TestClient(restarted) as client,
                notebook_guards(restarted.state.container) as guards,
            ):
                for format, expected in downloads.items():
                    response = client.get(
                        "/explore/run/notebook/export",
                        params={"run_id": bundle.run_id, "format": format},
                    )
                    response.raise_for_status()
                    if response.content != expected:
                        raise RuntimeError("Notebook changed after corpus change/deletion/restart.")
                counts.update(
                    {f"{stage}:{name}": guard.call_count for name, guard in guards.items()}
                )
            events_after = restarted.state.container.event_log.list_events()
            if path.read_bytes() != before or events_after != events:
                raise RuntimeError("Restarted reads changed saved records.")
        remaining = len(restarted.state.container.document_store.list_chunks())
        socket_calls = sum(guard.call_count for guard in network)
        if remaining or socket_calls or any(counts.values()):
            raise RuntimeError("Offline/no-work notebook checks failed.")
    if path.exists():
        raise RuntimeError("The temporary synthetic database was not removed.")
    offline = ReviewNotebook.model_validate_json((output_dir / "notebook.json").read_bytes())
    if offline.to_markdown().encode("utf-8") != (output_dir / "notebook.md").read_bytes():
        raise RuntimeError("Portable JSON cannot reproduce the Markdown without the database.")
    selected = offline.annotations[0]
    text = offline.evidence.snapshot.sources[0].chunk.text
    if (
        selected.start != text.rindex(QUOTE)
        or selected.start == text.index(QUOTE)
        or text[selected.start : selected.end] != selected.quote
        or selected.source_text_sha256 != hashlib.sha256(text.encode("utf-8")).hexdigest()
    ):
        raise RuntimeError("The exact duplicate-quote selector was not preserved.")
    checks = {
        "synthetic_only": True,
        "provider": offline.evidence.generation.provider,
        "run_id": offline.evidence.run_id,
        "reviews": len(offline.reviews),
        "annotations": len(offline.annotations),
        "first_page_reviews": len(first_pages["reviews"]["reviews"]),
        "first_page_annotations": len(first_pages["annotations"]["annotations"]),
        "review_sequences": [item.sequence for item in offline.reviews],
        "annotation_sequences": [item.sequence for item in offline.annotations],
        "decisions": sorted({item.decision.value for item in offline.reviews}),
        "selected_annotation_id": str(selected.annotation_id),
        "selected_start": selected.start,
        "selected_end": selected.end,
        "first_occurrence_start": text.index(QUOTE),
        "source_text_sha256": selected.source_text_sha256,
        "events_before": len(events),
        "events_after": len(events_after),
        "corpus_change_delete_restart_identical": True,
        "database_removed": not path.exists(),
        "offline_json_reproduces_markdown": True,
        "remaining_corpus_chunks": remaining,
        "socket_connection_calls": socket_calls,
        "forbidden_call_counts": counts,
        "bytes": {format: len(content) for format, content in downloads.items()},
        "sha256": {
            format: hashlib.sha256(content).hexdigest() for format, content in downloads.items()
        },
    }
    panels = [
        "\n".join(
            [
                PANEL_TITLES[0],
                f"Setup: provider={checks['provider']}; state={offline.evidence.status}; "
                f"saved events={len(events)}.",
                f"Recorded human reviews={len(offline.reviews)}; "
                f"annotations={len(offline.annotations)}.",
                "Decisions: " + ", ".join(checks["decisions"]) + ".",
                "Human opinions are separate from model grounding and scientific truth.",
            ]
        ),
        "\n".join(
            [
                PANEL_TITLES[1],
                f"Default JSON pages: {checks['first_page_reviews']} reviews / "
                f"{checks['first_page_annotations']} annotations.",
                f"Complete export: {len(offline.reviews)} reviews / "
                f"{len(offline.annotations)} annotations.",
                f"Original sequences: {offline.reviews[0].sequence} "
                f"down to {offline.reviews[-1].sequence}.",
                f"JSON: {len(downloads['json'])} UTF-8 bytes; "
                f"Markdown: {len(downloads['markdown'])} UTF-8 bytes.",
                "One read-only SQLite snapshot; no page cursors or partial success.",
            ]
        ),
        "\n".join(
            [
                PANEL_TITLES[2],
                f"First occurrence starts at {text.index(QUOTE)}; selected span "
                f"[{selected.start}, {selected.end}).",
                "Exact quote: " + selected.quote,
                "Annotation UUID: " + str(selected.annotation_id),
                "Offsets are Unicode code points, not bytes or fuzzy text matches.",
                "The JSON retains source text, metadata and the full SHA-256.",
            ]
        ),
        "\n".join(
            [
                PANEL_TITLES[3],
                f"Current corpus chunks after deletion: {remaining}.",
                "JSON and Markdown: byte-identical after corpus changes and app restarts.",
                f"Saved events before/after: {checks['events_before']}/{checks['events_after']}.",
                f"Forbidden read-path calls: {sum(counts.values())}; socket calls: {socket_calls}.",
                "Temporary database removed; portable JSON still reproduces Markdown.",
                "Synthetic execution illustration, not a recording or scientific result.",
            ]
        ),
    ]
    transcript = "\n\n".join(panels) + "\n"
    _save(output_dir / "checks.json", (json.dumps(checks, indent=2) + "\n").encode("utf-8"))
    _save(output_dir / "transcript.txt", transcript.encode("utf-8"))
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
