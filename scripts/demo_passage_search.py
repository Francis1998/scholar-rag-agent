"""Measure native browser passage search against a temporary, synthetic offline corpus."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

import httpx
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_corpus_explorer import ExplorerPage, browsing_guards
from scripts.demo_corpus_explorer import _no_work as _no_work
from scripts.demo_evidence_export import offline_settings
from storage.literal_search import LiteralSearchPage, LiteralSearchRequest, SQLiteLiteralSearch
from storage.paper_collections import PaperCollection
from storage.source_context import SourceContext

SELECTED_ID = "paper/../\u7814?draft#v1"
SOURCE = "synthetic:passage-search"
PHRASE = "\U0001f52c cafe\u0301"
PANEL_TITLES = (
    "1. From the catalog to an exact phrase",
    "2. Inspect the first literal match",
    "3. Read its exact source and return",
    "4. Restart without generation or writes",
)


class SearchPage(ExplorerPage):
    """Read actual native controls, highlights and offsets without executing HTML."""

    def __init__(self, html: str) -> None:
        self.form_action = ""
        self.form_values: dict[str, str] = {}
        self.highlights: list[str] = []
        self.match_offsets: list[tuple[int, int]] = []
        self._form = False
        self._select = ""
        self._highlight = False
        super().__init__(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        attributes = dict(attrs)
        if tag == "form":
            self._form = True
            self.form_action = str(attributes["action"])
        if self._form:
            if tag == "input" and attributes.get("name"):
                self.form_values[str(attributes["name"])] = str(attributes.get("value", ""))
            if tag == "select":
                self._select = str(attributes["name"])
            if tag == "option" and "selected" in attributes:
                self.form_values[self._select] = str(attributes["value"])
        if tag == "mark":
            self._highlight = True
            self.highlights.append("")
        if attributes.get("class") == "match-offsets":
            self.match_offsets.append(
                (int(str(attributes["data-start"])), int(str(attributes["data-end"])))
            )

    def handle_data(self, data: str) -> None:
        super().handle_data(data)
        if self._highlight:
            self.highlights[-1] += data

    def handle_endtag(self, tag: str) -> None:
        super().handle_endtag(tag)
        if tag == "form":
            self._form = False
        if tag == "select":
            self._select = ""
        if tag == "mark":
            self._highlight = False


def seed_demo(container: AppContainer) -> PaperCollection:
    """Write only original synthetic fixtures before the guarded read-only measurements."""
    paper = Document(
        document_id=SELECTED_ID,
        title="Synthetic graph observations",
        source=SOURCE,
        text="UNEXPOSED_DOCUMENT_BODY",
    )
    excluded = Document(
        document_id="excluded", title="Another synthetic paper", source="other", text="Unexposed"
    )
    chunks = [
        Chunk(
            document_id=paper.document_id,
            chunk_id=identifier,
            title=paper.title,
            source=paper.source,
            text=text,
            metadata={"chunk_index": str(index), "private": "UNEXPOSED_METADATA"},
        )
        for index, identifier, text in (
            (0, "chunk-2", f"Synthetic preceding methods. {PHRASE} is fixture text, not a claim."),
            (
                1,
                "chunk-10",
                "Synthetic context. " * 60
                + PHRASE
                + " Repeated wording: "
                + PHRASE
                + '\r\nLiteral source markup: <script>alert("fixture")</script>',
            ),
            (2, "chunk-3", f"Synthetic limitations. {PHRASE} is not evidence of a real study."),
        )
    ]
    chunks.append(
        Chunk(
            document_id=excluded.document_id,
            chunk_id="excluded-chunk",
            title=excluded.title,
            source=excluded.source,
            text=PHRASE + " EXCLUDED_EVIDENCE",
            metadata={"chunk_index": "0"},
        )
    )
    container.document_store.add_documents([paper, excluded], chunks)
    return container.paper_collections.create(
        name="Synthetic phrase-reading selection", document_ids=[SELECTED_ID]
    )


def _get(client: TestClient, path: str, **params: str | int) -> httpx.Response:
    response = client.get(path, params=params) if params else client.get(path)
    response.raise_for_status()
    if (
        response.headers["cache-control"] != "no-store"
        or response.headers["x-content-type-options"] != "nosniff"
        or "UNEXPOSED_" in response.text
        or "EXCLUDED_EVIDENCE" in response.text
    ):
        raise RuntimeError("A browser read lost its privacy headers or exact selected scope.")
    return response


def run_demo(output_dir: Path) -> str:
    """Save measured HTML/JSON and assertions; never retain the temporary database."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    artifacts: dict[str, bytes] = {}
    with TemporaryDirectory(prefix="scholar-browser-search-") as temporary:
        path = Path(temporary) / "corpus.sqlite3"
        settings = offline_settings(path)
        app = create_app(settings)
        collection = seed_demo(app.state.container)
        before = path.read_bytes()
        events_before = app.state.container.event_log.list_events()
        with TestClient(app) as client, browsing_guards(app.state.container) as guards:
            catalog = _get(client, "/explore", source=SOURCE, title="graph", limit=2)
            form = _get(client, SearchPage(catalog.text).links["search-document-1"])
            controls = SearchPage(form.text)
            submitted = {**controls.form_values, "query": PHRASE, "limit": "1"}
            paper = _get(client, controls.form_action, **submitted)
            submitted.update(scope="collection", scope_id=collection.collection_id)
            first = _get(client, controls.form_action, **submitted)
            first_page = SearchPage(first.text)
            following = first_page.links["next-page"]
            second = _get(client, following)
            second_page = SearchPage(second.text)
            context = _get(client, second_page.links["source-context-1"])
            context_page = SearchPage(context.text)
            recentered = _get(client, context_page.links["center-context-2"])
            returned = _get(client, context_page.links["back-to-search"])
            whole = client.post("/research/search", json={"query": PHRASE})
            whole.raise_for_status()
            selected = SQLiteLiteralSearch(path).search(
                LiteralSearchRequest(query=PHRASE, document_ids=[SELECTED_ID])
            )
            request = LiteralSearchRequest(
                query=PHRASE, collection_id=collection.collection_id, limit=1
            )
            response = client.post("/research/search", json=request.model_dump(exclude_unset=True))
            response.raise_for_status()
            result = LiteralSearchPage.model_validate_json(response.content)
            python_result = SQLiteLiteralSearch(path).search(request)
            empty = _get(client, controls.form_action, **{**submitted, "query": PHRASE.upper()})
            next_params = {
                key: value[0] for key, value in parse_qs(urlsplit(following).query).items()
            }
            stale = client.get(
                controls.form_action, params={**next_params, "query": PHRASE.upper()}
            )
            source_response = _get(
                client, "/documents/context", document_id=SELECTED_ID, chunk_id="chunk-2"
            )
            source = SourceContext.model_validate_json(source_response.content)
            match = result.matches[0]
            if (
                controls.form_values["scope"] != "document"
                or SearchPage(paper.text).chunks != ["chunk-10"]
                or len(whole.json()["matches"]) != 4
                or len(selected.matches) != 3
                or any(item.document_id != SELECTED_ID for item in selected.matches)
                or first_page.chunks != ["chunk-10"]
                or first_page.documents != [SELECTED_ID]
                or first_page.highlights != [PHRASE]
                or first_page.match_offsets != [(match.match_start, match.match_end)]
                or first_page.passages != [match.excerpt]
                or match.match_start <= 800
                or second_page.chunks != ["chunk-2"]
                or context_page.chunks != [chunk.chunk_id for chunk in source.chunks]
                or source.anchor_chunk_id != "chunk-2"
                or [chunk.chunk_index for chunk in source.chunks] != [0, 1, 2]
                or SearchPage(recentered.text).links["back-to-search"] != following
                or returned.content != second.content
                or python_result.to_json().encode() != response.content
                or SearchPage(empty.text).chunks
                or stale.status_code != 422
                or SearchPage(stale.text).chunks
            ):
                raise RuntimeError("Native search, highlighting, scope, context or parity failed.")
            artifacts.update(
                {
                    "catalog.html": catalog.content,
                    "form.html": form.content,
                    "paper.html": paper.content,
                    "collection-first.html": first.content,
                    "collection-next.html": second.content,
                    "context.html": context.content,
                    "recentered.html": recentered.content,
                    "empty.html": empty.content,
                    "stale-cursor.html": stale.content,
                    "whole-corpus.json": whole.content,
                    "collection-first.json": response.content,
                    "context.json": source_response.content,
                    "paper.json": selected.to_json().encode(),
                }
            )
            counts = {name: guard.call_count for name, guard in guards.items()}
        restarted = create_app(settings)
        with TestClient(restarted) as client, browsing_guards(restarted.state.container) as guards:
            restart_identical = (
                _get(client, following).content == second.content
                and _get(client, second_page.links["source-context-1"]).content == context.content
            )
            for name, guard in guards.items():
                counts[name] += guard.call_count
        events_after = restarted.state.container.event_log.list_events()
        unchanged = path.read_bytes() == before
        if (
            not restart_identical
            or not unchanged
            or events_after != events_before
            or any(counts.values())
        ):
            raise RuntimeError("Reading/restart mutated evidence or attempted forbidden work.")
        checks = {
            "fixture": "scholar-browser-passage-search-synthetic-v1",
            "whole_corpus_matches": len(whole.json()["matches"]),
            "selected_paper_matches": len(selected.matches),
            "collection_revision": result.collection_revision,
            "first_chunk_id": match.chunk_id,
            "match_start": match.match_start,
            "match_end": match.match_end,
            "query_codepoints": len(PHRASE),
            "excerpt_start": match.excerpt_start,
            "excerpt_end": match.excerpt_end,
            "highlight_identical": first_page.highlights == [PHRASE],
            "next_chunk_id": second_page.chunks[0],
            "context_anchor": source.anchor_chunk_id,
            "source_indices": [chunk.chunk_index for chunk in source.chunks],
            "exact_search_return": returned.content == second.content,
            "empty_matches": len(SearchPage(empty.text).chunks),
            "changed_query_cursor_status": stale.status_code,
            "python_api_identical": python_result == result,
            "restart_identical": restart_identical,
            "database_bytes_unchanged": unchanged,
            "events_before": len(events_before),
            "events_after": len(events_after),
            "forbidden_call_counts": counts,
        }
    panels = [
        [
            PANEL_TITLES[0],
            f"Catalog -> native GET form -> paper search: HTTP {paper.status_code}",
            f"Whole corpus: {checks['whole_corpus_matches']} matches; selected paper: "
            f"{checks['selected_paper_matches']}.",
            f"Saved collection revision: {result.collection_revision}; no other paper returned.",
            f"Exact phrase (JSON): {json.dumps(PHRASE, ensure_ascii=True)}",
            "GET queries appear in URLs/history/logs; no scripts or external resources.",
        ],
        [
            PANEL_TITLES[1],
            f"First result: {match.chunk_id}; next page: {second_page.chunks[0]}.",
            f"First Unicode match: [{match.match_start}, {match.match_end}); "
            f"phrase length: {len(PHRASE)} code points.",
            f"Excerpt: [{match.excerpt_start}, {match.excerpt_end}); "
            f"characters: {len(match.excerpt)}.",
            f"HTML mark equals exact phrase: {checks['highlight_identical']}",
            "The later repeated occurrence is not highlighted; IDs are not relevance ranks.",
        ],
        [
            PANEL_TITLES[2],
            f"Exact context anchor: {source.anchor_chunk_id}; "
            f"source indices: {checks['source_indices']}",
            f"Native recenter -> original search page restored: {checks['exact_search_return']}",
            f"Case-changed phrase: {checks['empty_matches']} results, no widened scope.",
            f"Cursor with changed phrase: HTTP {stale.status_code}, no partial result.",
            "Current corpus, not frozen evidence; context requires valid source order.",
        ],
        [
            PANEL_TITLES[3],
            f"Python/API parity: {checks['python_api_identical']}; "
            f"restart identical: {restart_identical}",
            f"Database bytes unchanged: {unchanged}",
            f"Events before/after: {len(events_before)}/{len(events_after)}",
            f"Forbidden model/retrieval/network/write attempts: {sum(counts.values())}",
            "Original synthetic fixtures, not research findings or a quality benchmark.",
            "Actual HTML, JSON and checks saved; temporary corpus removed.",
        ],
    ]
    transcript = "\n\n".join("\n".join(panel) for panel in panels) + "\n"
    artifacts["checks.json"] = (json.dumps(checks, indent=2) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode()
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as destination:
            destination.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
