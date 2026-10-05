"""Measure complete human-screening downloads using synthetic data and no model calls."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from agent.runner import AgentRunner
from api.application import create_app
from llm.fake import FakeLLMAdapter
from llm.providers import HTTPProviderAdapter
from retrieval.hybrid import HybridRetriever
from scripts.create_evidence_gif import render_panels
from scripts.demo_evidence_export import offline_settings
from storage.event_log import SQLiteEventLog
from storage.paper_screening import ScreeningExport


def _forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("This demo forbids network, generation, retrieval, and agent-event writes.")


def _download(client: TestClient, path: str, format: str) -> httpx.Response:
    response = client.get(path, params={"collection_revision": 2, "format": format})
    response.raise_for_status()
    return response


def run_demo(output_dir: Path) -> str:
    """Save actual download bytes and measured assertions, never a real user's corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "screening-results.json",
        "screening-results.csv",
        "measurements.json",
        "transcript.txt",
    ):
        target = output_dir / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}.")
    with (
        TemporaryDirectory(prefix="scholar-screening-export-") as temporary,
        patch.object(httpx.HTTPTransport, "handle_request", side_effect=_forbidden) as http,
        patch.object(
            httpx.AsyncHTTPTransport, "handle_async_request", side_effect=_forbidden
        ) as ahttp,
        patch.object(FakeLLMAdapter, "generate", side_effect=_forbidden) as fake,
        patch.object(HTTPProviderAdapter, "generate", side_effect=_forbidden) as live,
        patch.object(AgentRunner, "run", side_effect=_forbidden),
        patch.object(AgentRunner, "preview", side_effect=_forbidden) as preview,
        patch.object(HybridRetriever, "retrieve", side_effect=_forbidden) as retrieve,
        patch.object(SQLiteEventLog, "append_event", side_effect=_forbidden),
        patch.object(SQLiteEventLog, "append_transition", side_effect=_forbidden),
    ):
        database = Path(temporary) / "corpus.sqlite3"
        settings = offline_settings(database)
        app = create_app(settings)
        with TestClient(app) as client:
            ids = []
            for index in range(25):
                title = (
                    '=Synthetic methods, "quoted"'
                    if index == 0
                    else "Synthetic " + "L" * 310
                    if index == 1
                    else f"Synthetic note {index + 1} \u7814\u7a76"
                )
                source = (
                    "\t@synthetic:export"
                    if index == 0
                    else "synthetic:" + "s" * 600
                    if index == 1
                    else "synthetic:export"
                )
                response = client.post(
                    "/ingest/text",
                    json={
                        "title": title,
                        "source": source,
                        "text": f"Synthetic note {index + 1}, not a publication. GraphRAG example.",
                    },
                )
                response.raise_for_status()
                ids.append(response.json()["document_id"])
            created = client.post(
                "/collections", json={"name": "Synthetic screening export", "document_ids": ids}
            )
            created.raise_for_status()
            collection_path = f"/collections/{created.json()['collection_id']}"
            screening_path = f"{collection_path}/screening"
            client.put(
                f"{screening_path}/{ids[3]}",
                json={
                    "collection_revision": 1,
                    "expected_decision_revision": 0,
                    "decision": "include",
                    "reason": "Old synthetic human opinion.",
                },
            ).raise_for_status()
            client.put(
                collection_path,
                json={
                    "name": "Revised synthetic screening",
                    "document_ids": ids,
                    "expected_revision": 1,
                },
            ).raise_for_status()
            for index, decision in enumerate(("include", "exclude", "unsure")):
                client.put(
                    f"{screening_path}/{ids[index]}",
                    json={
                        "collection_revision": 2,
                        "expected_decision_revision": 0,
                        "decision": decision,
                        "reason": " \t=Synthetic human opinion\nNot scientific validation."
                        if index == 0
                        else f"Synthetic human {decision} opinion.",
                    },
                ).raise_for_status()
            before = database.read_bytes()
            queue = client.get(screening_path, params={"collection_revision": 2})
            queue.raise_for_status()
            path = f"{screening_path}/export"
            downloads = {format: _download(client, path, format) for format in ("json", "csv")}
            stale = client.get(path, params={"collection_revision": 1})
            unchanged = before == database.read_bytes()
        restarted = create_app(settings)
        with TestClient(restarted) as client:
            restart_equal = all(
                _download(client, path, format).content == response.content
                for format, response in downloads.items()
            )
            events = restarted.state.container.event_log.list_events()
        result = ScreeningExport.model_validate_json(downloads["json"].content)
        rows = list(csv.DictReader(io.StringIO(downloads["csv"].text, newline="")))
        included = [row["included_document_id"][1:] for row in rows if row["included_document_id"]]
        first = result.items[0]
        if first.review is None:
            raise RuntimeError("The synthetic human review was not retained.")
        round_trip = all(
            rows[0][field] == "'" + text
            for field, text in (
                ("title", first.title),
                ("source", first.source),
                ("review_reason", first.review.reason),
            )
        )
        counts = result.counts.model_dump()
        metrics = {
            "synthetic_only": True,
            "queue_page_items": len(queue.json()["items"]),
            "export_members": len(result.items),
            "csv_rows": len(rows),
            "counts": counts,
            "included_document_ids": included,
            "stale_include_excluded": ids[3] not in included,
            "title_truncations": sum(item.title_truncated for item in result.items),
            "source_truncations": sum(item.source_truncated for item in result.items),
            "csv_text_round_trip": round_trip,
            "stale_request_status": stale.status_code,
            "restart_bytes_equal": restart_equal,
            "export_database_bytes_unchanged": unchanged,
            "agent_event_count": len(events),
            "external_http_attempts": http.call_count + ahttp.call_count,
            "generation_attempts": fake.call_count + live.call_count,
            "retrieval_attempts": preview.call_count + retrieve.call_count,
            "downloads": {
                format: {
                    "bytes": len(response.content),
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                    "content_type": response.headers["content-type"],
                    "content_disposition": response.headers["content-disposition"],
                }
                for format, response in downloads.items()
            },
        }
        if (
            result.total_documents != 25
            or len(rows) != 25
            or len(queue.json()["items"]) != 20
            or counts != {"include": 1, "exclude": 1, "unsure": 1, "stale": 1, "unscreened": 21}
            or included != list(result.included_document_ids)
            or included != [ids[0]]
            or not all((round_trip, restart_equal, unchanged))
            or stale.status_code != 409
            or events
        ):
            raise RuntimeError(
                "Measured screening exports did not satisfy the complete-export contract."
            )
    panels = [
        "\n".join(
            [
                "1. Export the whole human screening collection",
                f"Default queue page: {metrics['queue_page_items']} members",
                f"JSON members: {metrics['export_members']}; CSV rows: {metrics['csv_rows']}",
                "Order is the saved collection order, not a relevance ranking.",
                "All observations come from synthetic, offline API responses.",
            ]
        ),
        "\n".join(
            [
                "2. Keep current decisions separate from old opinions",
                f"Include: {counts['include']}; exclude: {counts['exclude']}; "
                f"unsure: {counts['unsure']}",
                f"Stale: {counts['stale']}; unscreened: {counts['unscreened']}",
                f"Explicit included IDs: {len(included)}; "
                f"old include excluded: {metrics['stale_include_excluded']}",
                f"Outdated collection revision: HTTP {stale.status_code}",
            ]
        ),
        "\n".join(
            [
                "3. Save bounded JSON and spreadsheet-safe CSV",
                f"JSON bytes: {len(downloads['json'].content)}; "
                f"CSV bytes: {len(downloads['csv'].content)}",
                f"Title/source truncations: "
                f"{metrics['title_truncations']}/{metrics['source_truncations']}",
                f"CSV formula-like text round trip: {round_trip}",
                "Text cells have one reversible apostrophe prefix; JSON stays exact.",
            ]
        ),
        "\n".join(
            [
                "4. Reopen and reproduce without running a model",
                f"Identical download bytes after restart: {restart_equal}",
                f"Export leaves database bytes unchanged: {unchanged}",
                f"Agent events: {len(events)}; HTTP/model/retrieval attempts: "
                f"{metrics['external_http_attempts']}/{metrics['generation_attempts']}/{metrics['retrieval_attempts']}",
                "Not a live UI, active learning, frozen paper text, or scientific proof.",
            ]
        ),
    ]
    transcript = "\n\n".join(panels) + "\n"
    for format, response in downloads.items():
        (output_dir / f"screening-results.{format}").write_bytes(response.content)
    (output_dir / "measurements.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def create_gif(transcript: str, output_path: Path) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_path}.")
    render_panels(
        transcript.strip().split("\n\n"),
        output_path,
        banner="SCHOLAR RAG / SCREENING EXPORTS / MEASURED SYNTHETIC + OFFLINE",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif", type=Path)
    arguments = parser.parse_args()
    if arguments.gif is not None and (arguments.gif.exists() or arguments.gif.is_symlink()):
        raise FileExistsError(f"Refusing to overwrite {arguments.gif}.")
    transcript = run_demo(arguments.output)
    if arguments.gif is not None:
        create_gif(transcript, arguments.gif)
    print(transcript, end="")


if __name__ == "__main__":
    main()
