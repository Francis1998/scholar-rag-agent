"""Measure the real literal-search API/Python workflow on disposable synthetic data."""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_evidence_export import offline_settings
from storage.literal_search import LiteralSearchPage, LiteralSearchRequest, SQLiteLiteralSearch

DEMO_QUERY = "exact phrase"
DEMO_LITERAL = "100%_proof\\path"
DEMO_TEXT = (
    "\u03b2\U0001f52c" + "Synthetic context. " * 60 + DEMO_QUERY + " beside " + DEMO_LITERAL + "."
)
PANEL_TITLES = (
    "1. Open a synthetic persisted corpus",
    "2. Find the phrase, not a ranked guess",
    "3. Continue in exact selected scope",
    "4. Check literal semantics and no side effects",
)
OUTPUT_NAMES = (
    "collection.json",
    "whole-corpus.json",
    "page-1.json",
    "page-2.json",
    "page-3.json",
    "literal.json",
    "empty.json",
    "mismatch.json",
    "python.json",
    "restarted.json",
    "checks.json",
    "transcript.txt",
)


def _no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Literal search attempted network, model, retrieval, or persistence work.")


def _block_search_work(stack: ExitStack, container: AppContainer) -> list[Mock]:
    return [
        stack.enter_context(patch.object(target, method, side_effect=_no_work))
        for target, method in (
            (container.document_store, "list_chunks"),
            (container.document_store, "add_documents"),
            (container.hybrid_retriever, "retrieve"),
            (container.hybrid_retriever, "add_chunks"),
            (container.llm, "generate"),
            (container.runner, "run"),
            (container.runner, "preview"),
            (container.event_log, "append_event"),
            (container.event_log, "list_events"),
            (container.paper_collections, "create"),
            (container.paper_collections, "resolve"),
        )
    ]


def run_demo(output_dir: Path) -> str:
    """Keep measured JSON/text only; never consult ambient credentials or retain the corpus."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    with TemporaryDirectory(prefix="scholar-literal-demo-") as temporary, ExitStack() as network:
        database_path = Path(temporary) / "corpus.sqlite3"
        network_guards = [
            network.enter_context(patch.object(target, method, side_effect=_no_work))
            for target, method in (
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
            )
        ]
        app = create_app(offline_settings(database_path))
        container: AppContainer = app.state.container
        documents = [
            Document(
                document_id=identifier,
                title=f"Synthetic {identifier}",
                source="synthetic:literal-search",
                text="Unexposed synthetic document body",
            )
            for identifier in ("a-outside", "paper-1", "paper-2")
        ]
        chunks = [
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title="Synthetic title " + ("t" * 310 if chunk_id == "chunk-1" else ""),
                source="synthetic:literal-search",
                text=text,
                metadata={"private": "UNEXPOSED_METADATA"},
            )
            for chunk_id, document_id, text in (
                ("outside", "a-outside", "The exact phrase also occurs outside the selection."),
                ("chunk-1", "paper-1", DEMO_TEXT),
                ("chunk-2", "paper-1", "A second exact phrase in the selected paper."),
                ("chunk-3", "paper-2", f"Another exact phrase, plus {DEMO_LITERAL}."),
            )
        ]
        container.document_store.add_documents(documents, chunks)
        collection = container.paper_collections.create(
            name="Synthetic selected papers", document_ids=["paper-1", "paper-2"]
        )
        # Reopen just the reader: the API's in-memory retriever deliberately has no fixture chunks.
        container.literal_search = SQLiteLiteralSearch(database_path)
        before = database_path.read_bytes()
        events_before = container.event_log.list_events()
        with TestClient(app) as client, ExitStack() as guards:
            blocked = _block_search_work(guards, container)
            whole_response = client.post("/research/search", json={"query": DEMO_QUERY})
            whole_response.raise_for_status()
            whole = LiteralSearchPage.model_validate_json(whole_response.content)
            pages = []
            responses = []
            cursor = None
            for _ in range(3):
                payload: dict[str, object] = {
                    "query": DEMO_QUERY,
                    "collection_id": collection.collection_id,
                    "limit": 1,
                }
                if cursor is not None:
                    payload["cursor"] = cursor
                response = client.post("/research/search", json=payload)
                response.raise_for_status()
                page = LiteralSearchPage.model_validate_json(response.content)
                pages.append(page)
                responses.append(response)
                cursor = page.next_cursor
            literal_response = client.post("/research/search", json={"query": DEMO_LITERAL})
            literal_response.raise_for_status()
            literal = LiteralSearchPage.model_validate_json(literal_response.content)
            empty_response = client.post("/research/search", json={"query": "EXACT PHRASE"})
            empty_response.raise_for_status()
            empty = LiteralSearchPage.model_validate_json(empty_response.content)
            mismatch_response = client.post(
                "/research/search",
                json={
                    "query": DEMO_QUERY + " changed",
                    "collection_id": collection.collection_id,
                    "cursor": pages[0].next_cursor,
                },
            )
            python_result = SQLiteLiteralSearch(database_path).search(
                LiteralSearchRequest(
                    query=DEMO_QUERY, collection_id=collection.collection_id, limit=1
                )
            )
            restarted = SQLiteLiteralSearch(database_path).search(
                LiteralSearchRequest(
                    query=DEMO_QUERY, collection_id=collection.collection_id, limit=1
                )
            )
            matches = [match for page in pages for match in page.matches]
            first = matches[0]
            if (
                len(whole.matches) != 4
                or [match.chunk_id for match in matches] != ["chunk-1", "chunk-2", "chunk-3"]
                or {match.document_id for match in matches} != {"paper-1", "paper-2"}
                or first.match_start != DEMO_TEXT.index(DEMO_QUERY)
                or first.match_start <= 800
                or first.excerpt != DEMO_TEXT[first.excerpt_start : first.excerpt_end]
                or DEMO_QUERY not in first.excerpt
                or not first.title_truncated
                or pages[0].next_cursor is None
                or pages[1].next_cursor is None
                or pages[2].next_cursor is not None
                or len(literal.matches) != 2
                or empty.matches
                or mismatch_response.status_code != 422
                or mismatch_response.json()["detail"]["code"] != "invalid_search_cursor"
                or python_result.to_json().encode() != responses[0].content
                or restarted != python_result
                or any(response.headers["cache-control"] != "no-store" for response in responses)
            ):
                raise RuntimeError("The measured literal search workflow violated its contracts.")
            work_calls = sum(guard.call_count for guard in blocked)
        events_after = container.event_log.list_events()
        after = database_path.read_bytes()
        network_calls = sum(guard.call_count for guard in network_guards)
        if before != after or events_before != events_after or work_calls or network_calls:
            raise RuntimeError("Literal search changed storage or attempted prohibited work.")

    checks = {
        "synthetic_only": True,
        "whole_corpus_matches": len(whole.matches),
        "selected_matches": len(matches),
        "selected_chunk_ids": [match.chunk_id for match in matches],
        "page_sizes": [len(page.matches) for page in pages],
        "match_start": first.match_start,
        "match_end": first.match_end,
        "excerpt_start": first.excerpt_start,
        "excerpt_end": first.excerpt_end,
        "title_characters": len(first.title),
        "title_truncated": first.title_truncated,
        "literal_matches": len(literal.matches),
        "empty_matches": len(empty.matches),
        "mismatch_status": mismatch_response.status_code,
        "mismatch_code": mismatch_response.json()["detail"]["code"],
        "first_page_bytes": len(responses[0].content),
        "python_http_identical": python_result.to_json().encode() == responses[0].content,
        "restart_identical": restarted == python_result,
        "network_calls": network_calls,
        "search_model_retrieval_or_write_calls": work_calls,
        "events_before": len(events_before),
        "events_after": len(events_after),
        "database_sha256": {
            "before": hashlib.sha256(before).hexdigest(),
            "after": hashlib.sha256(after).hexdigest(),
        },
        "temporary_database_removed": not database_path.exists(),
    }
    transcript = (
        "\n\n".join(
            (
                "\n".join(
                    (
                        PANEL_TITLES[0],
                        f"Saved {len(documents)} papers and {len(chunks)} chunks, "
                        "then reopened reader.",
                        f"Whole-corpus literal matches: {len(whole.matches)}",
                        f"Collection selection: {len(collection.document_ids)} papers; revision 1",
                        "The in-memory retriever was not populated with these fixtures.",
                        "Temporary SQLite only; settings ignore ambient keys and databases.",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[1],
                        f"POST /research/search -> {responses[0].status_code}; "
                        f'query="{DEMO_QUERY}"',
                        f"First hit: {first.document_id} / {first.chunk_id}",
                        f"Unicode match offsets: [{first.match_start}, {first.match_end})",
                        f"Excerpt offsets: [{first.excerpt_start}, {first.excerpt_end})",
                        f'Matched text: "{DEMO_TEXT[first.match_start : first.match_end]}"',
                        f"Title: {len(first.title)} characters; truncated={first.title_truncated}",
                        "Offsets refer to current full chunk text, not the displayed prefix.",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[2],
                        f"Page sizes: {', '.join(str(len(page.matches)) for page in pages)}",
                        f"Distinct selected chunks: {len(matches)}; final cursor=None",
                        "Order: document ID, then chunk ID; no rank or relevance score.",
                        f"Changed-query cursor reuse -> {mismatch_response.status_code}",
                        f"Python/HTTP identical: {checks['python_http_identical']}",
                        f"Reader restart identical: {checks['restart_identical']}",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[3],
                        f'Literal "{DEMO_LITERAL}" matches: {len(literal.matches)}',
                        f"Uppercase query matches: {len(empty.matches)} (case-sensitive)",
                        f"First-page UTF-8 JSON: {len(responses[0].content)} bytes",
                        f"Network/model/retrieval/write calls: {network_calls}/{work_calls}",
                        f"Events before/after: {len(events_before)}/{len(events_after)}",
                        "Database bytes unchanged; temporary database removed.",
                        "Exact wording is not scientific support or absence of evidence.",
                    )
                ),
            )
        )
        + "\n"
    )
    artifacts = {
        "collection.json": collection.model_dump_json().encode(),
        "whole-corpus.json": whole_response.content,
        **{f"page-{i}.json": response.content for i, response in enumerate(responses, 1)},
        "literal.json": literal_response.content,
        "empty.json": empty_response.content,
        "mismatch.json": mismatch_response.content,
        "python.json": python_result.to_json().encode(),
        "restarted.json": restarted.to_json().encode(),
        "checks.json": json.dumps(checks, indent=2, sort_keys=True).encode(),
        "transcript.txt": transcript.encode(),
    }
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
