"""Execute a native retrieval workbench with original synthetic data and measured guards."""

import argparse
import asyncio
import json
import socket
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from agent.retrieval_preview import RetrievalPreview, RetrievalPreviewError
from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_browser_worksheet import WorksheetPage, worksheet_guards
from scripts.demo_browser_worksheet import _deny as _deny
from scripts.demo_evidence_export import offline_settings

SELECTED_ID = "paper/../synthetic-\u7814?draft#v1"
QUERY = "What are the local retrieval limitations?"
PANEL_TITLES = (
    "1. Open controls without running retrieval",
    "2. Inspect real prepared passages and policies",
    "3. Follow exact current sources, not saved runs",
    "4. Measure failures and the read-only boundary",
)


class PreviewPage(WorksheetPage):
    """Parse literal fields and native controls from the actual response, not a mock view."""

    def __init__(self, html: str) -> None:
        self.fields: dict[str, list[str]] = {}
        super().__init__(html)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._tag and self._field:
            self.fields.setdefault(self._field.split()[-1], []).append(self._text)
        super().handle_endtag(tag)


def run_demo(output_dir: Path) -> str:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    with TemporaryDirectory(prefix="native-preview-") as directory, ExitStack() as stack:
        network = [
            stack.enter_context(patch.object(target, method, side_effect=_deny))
            for target, method in (
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
                (socket, "create_connection"),
                (socket.socket, "connect"),
            )
        ]
        database = Path(directory) / "corpus.sqlite3"
        app = create_app(
            offline_settings(database).model_copy(update={"max_source_docs": 3, "max_hops": 1})
        )
        container: AppContainer = app.state.container
        chunks = [
            Chunk(
                document_id=identifier,
                chunk_id=identifier + "/chunk-0",
                title=title,
                source="synthetic:native-preview",
                text=text,
                metadata={"chunk_index": "0"},
            )
            for identifier, title, text in (
                (
                    SELECTED_ID,
                    "Synthetic lexical limitations",
                    "Original synthetic note: retrieval compares local terms, "
                    "not scientific truth. "
                    "Literal <script>not executable</script> & cafe\u0301 \u7814\u7a76.",
                ),
                (
                    "second-paper",
                    "Synthetic graph limitations",
                    "Original synthetic note: graph retrieval follows entity co-mentions. "
                    "Connections do not establish causality or scientific independence.",
                ),
                ("unselected", "Excluded synthetic note", "EXCLUDED_MARKER retrieval limitations."),
            )
        ]
        container.document_store.add_documents(
            [Document(**chunk.model_dump(exclude={"chunk_id"})) for chunk in chunks], chunks
        )
        container.hybrid_retriever.add_chunks(chunks)
        container.graph_builder.index_chunks(chunks)
        collection = container.paper_collections.create(
            name="Synthetic preview selection", document_ids=[SELECTED_ID, "second-paper"]
        )
        before = database.read_bytes()
        with TestClient(app) as client, worksheet_guards(container) as guards:
            with patch.object(container.runner, "preview", wraps=container.runner.preview) as calls:
                catalog = client.get("/explore?limit=2")
                catalog.raise_for_status()
                opened = client.get(PreviewPage(catalog.text).links["retrieval-preview"])
                opened.raise_for_status()
                form = PreviewPage(opened.text)
                opening_calls = calls.call_count
                fields = {
                    **form.form_values,
                    "query": QUERY,
                    "scope": "documents",
                    "document_ids": json.dumps([SELECTED_ID, "second-paper"]),
                    "max_chunks_per_document": "1",
                    "min_evidence_documents": "3",
                    "near_duplicate_threshold": "0.9",
                }
                html = client.post(
                    form.form_action, data=fields, headers={"Origin": "http://testserver"}
                )
                html.raise_for_status()
                submit_calls = calls.call_count - opening_calls
            result = PreviewPage(html.text)
            source_url = result.links["preview-source-1"]
            if not source_url.startswith("/explore/context?"):
                raise RuntimeError("The preview source left local exact-ID navigation.")
            context = client.get(source_url)
            context.raise_for_status()
            payload = {
                "query": QUERY,
                "document_ids": [SELECTED_ID, "second-paper"],
                "max_chunks_per_document": 1,
                "min_evidence_documents": 3,
                "near_duplicate_threshold": 0.9,
            }
            api = client.post("/retrieve", json=payload)
            api.raise_for_status()
            preview = RetrievalPreview.model_validate_json(api.content)
            python_preview = asyncio.run(
                container.runner.preview(
                    QUERY,
                    document_ids=[SELECTED_ID, "second-paper"],
                    max_chunks_per_document=1,
                    min_evidence_documents=3,
                    near_duplicate_threshold=0.9,
                )
            )
            collection_form = client.get(
                "/explore/retrieve", params={"collection_id": collection.collection_id}
            )
            collection_form.raise_for_status()
            collection_result = client.post(
                form.form_action,
                data={
                    **fields,
                    **PreviewPage(collection_form.text).form_values,
                    "query": QUERY,
                    "max_chunks_per_document": "1",
                    "min_evidence_documents": "3",
                    "near_duplicate_threshold": "0.9",
                },
                headers={"Origin": "http://testserver"},
            )
            collection_result.raise_for_status()
            invalid = client.post(
                form.form_action,
                data={**fields, "document_ids": ""},
                headers={"Origin": "http://testserver"},
            )
            empty = client.post(
                form.form_action,
                data={**fields, "document_ids": '["not-ingested"]'},
                headers={"Origin": "http://testserver"},
            )
            empty.raise_for_status()
            with patch.object(
                container.runner,
                "preview",
                new=AsyncMock(
                    side_effect=RetrievalPreviewError("retrieval_failed", "PRIVATE_DIAGNOSTIC")
                ),
            ):
                failed = client.post(
                    form.form_action, data=fields, headers={"Origin": "http://testserver"}
                )
            counts = {name: guard.call_count for name, guard in guards.items()}
        artifacts = {
            "form.html": opened.content,
            "preview.html": html.content,
            "source.html": context.content,
            "preview.json": api.content,
            "failed.html": failed.content,
        }
        checks = {
            "opening_preview_calls": opening_calls,
            "submission_preview_calls": submit_calls,
            "prepared_passages": len(preview.sources),
            "minimum_diagnostic": preview.evidence_assessment.model_dump()
            if preview.evidence_assessment
            else None,
            "api_python_browser_equal": (
                preview == python_preview
                and result.fields["prepared-context"] == [preview.context]
                and result.fields["context-sha256"] == [preview.context_sha256]
                and result.passages == [source.chunk.text for source in preview.sources]
            ),
            "collection_same_context": PreviewPage(collection_result.text).fields[
                "prepared-context"
            ]
            == [preview.context],
            "source_context_exact_id": PreviewPage(context.text).chunks
            == [preview.sources[0].chunk.chunk_id],
            "empty_selection_status": invalid.status_code,
            "unknown_ids_empty_not_widened": not PreviewPage(empty.text).passages,
            "failure_status": failed.status_code,
            "failure_is_not_empty_success": (
                failed.status_code == 500
                and not PreviewPage(failed.text).passages
                and "PRIVATE_DIAGNOSTIC" not in failed.text
            ),
            "database_bytes_unchanged": database.read_bytes() == before,
            "agent_events": len(container.event_log.list_events()),
            "forbidden_call_counts": counts,
            "external_connection_calls": sum(guard.call_count for guard in network),
            "output_bytes": {name: len(content) for name, content in artifacts.items()},
        }
        if (
            opening_calls != 0
            or submit_calls != 1
            or len(preview.sources) != 2
            or preview.evidence_assessment is None
            or preview.evidence_assessment.passed
            or invalid.status_code != 422
            or "EXCLUDED_MARKER" in html.text
            or any(counts.values())
            or checks["external_connection_calls"] != 0
            or checks["agent_events"] != 0
            or not all(value for value in checks.values() if isinstance(value, bool))
        ):
            raise RuntimeError("The measured retrieval workbench contract failed.")
    checks["temporary_database_removed"] = not database.exists()
    if not checks["temporary_database_removed"]:
        raise RuntimeError("Temporary corpus was not removed.")
    transcript = (
        "\n\n".join(
            (
                f"{PANEL_TITLES[0]}\n"
                f"Catalog -> native POST form; opening preview calls: {opening_calls}.\n"
                f"Explicit submission preview calls: {submit_calls}; no JavaScript or saved job.\n"
                "Question plus exact JSON IDs, collection or explicit whole-corpus scope.",
                f"{PANEL_TITLES[1]}\n"
                f"Actual ordered passages: {len(preview.sources)}; quota: 1 per document.\n"
                "Lexical collapse threshold: 0.9; required documents: 3; observed: 2.\n"
                "Minimum NOT met; full passages retained, not an empty success.\n"
                f"API / Python / browser preparation equal: {checks['api_python_browser_equal']}.",
                f"{PANEL_TITLES[2]}\n"
                f"Exact current source opened: {checks['source_context_exact_id']}.\n"
                f"Collection resolved the same context: {checks['collection_same_context']}.\n"
                "Complete text, rank, score, retriever, path, metadata and digests shown.\n"
                "Current source reads may change; a preview is not frozen saved evidence.",
                f"{PANEL_TITLES[3]}\n"
                f"Empty selection: HTTP {invalid.status_code}; "
                f"retrieval failure: HTTP {failed.status_code}.\n"
                f"Forbidden generation/write calls: {sum(counts.values())}; agent events: 0.\n"
                f"External connections: {checks['external_connection_calls']}; "
                f"database unchanged: {checks['database_bytes_unchanged']}.\n"
                "Temporary corpus removed. Synthetic output illustration, not browser footage.",
            )
        )
        + "\n"
    )
    artifacts["checks.json"] = (json.dumps(checks, indent=2, ensure_ascii=True) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode()
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as target:
            target.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(run_demo(parser.parse_args().output_dir), end="")


if __name__ == "__main__":
    main()
