"""Offline HTML contracts exercised against real saved-run exports, not template mocks."""

import base64
import hashlib
import json
import re
import socket
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.evidence import EvidenceBundle
from agent.models import AgentAnswer, AgentState, Citation, Claim, StateTransition
from api.application import create_app
from api.dependencies import AppContainer
from api.evidence import render_markdown
from llm.schemas import LLMRequest, LLMResponse
from retrieval.models import Chunk, Document
from storage.evidence_export import EvidenceExportError

QUERY = "  How does GraphRAG connect synthetic evidence? \u7814\u7a76  "
HOSTILE = (
    '</code></pre><script>alert("fixture")</script><img src="https://example.invalid/pixel">'
    '<svg onload="alert(1)"></svg><base href="https://example.invalid/">'
    '<meta http-equiv="refresh" content="0;url=https://example.invalid/">'
    '<style>@import "https://example.invalid/x";</style><!-- comment -->'
    "[markdown](javascript:alert(1)) & \" ' \u202eRTL\u202c"
)
PASSAGE = (
    "\n\n  GraphRAG connects synthetic evidence.\r\n"
    "\tKeep exact spacing, cafe\u0301, \u7814\u7a76, \U0001f52c, "
    "\u05e9\u05dc\u05d5\u05dd and \u202eRTL\u202c. Literal &lt;b&gt; &amp; &#13;.\rEnd.  \n"
    + HOSTILE
)


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Node | str"] = field(default_factory=list)

    def walk(self) -> Iterator["Node"]:
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self) -> str:
        return "".join(child if isinstance(child, str) else child.text() for child in self.children)

    def by_id(self, identifier: str) -> "Node":
        matches = [node for node in self.walk() if node.attrs.get("id") == identifier]
        assert len(matches) == 1, (identifier, len(matches))
        return matches[0]

    def by_tag(self, tag: str) -> list["Node"]:
        return [node for node in self.walk() if node.tag == tag]

    def by_class(self, name: str) -> list["Node"]:
        return [node for node in self.walk() if name in node.attrs.get("class", "").split()]


class ReaderDOM(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]
        self.declarations: list[str] = []
        self.feed(html)
        self.close()
        assert self.stack == [self.root], "Unclosed HTML element"

    def handle_decl(self, decl: str) -> None:
        self.declarations.append(decl.lower())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        assert len({key for key, _ in attrs}) == len(attrs), "Duplicate attribute"
        node = Node(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in {"meta", "br", "hr"}:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        assert self.stack[-1].tag == tag, (tag, self.stack[-1].tag)
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)

    def handle_comment(self, data: str) -> None:
        pytest.fail(f"Dynamic HTML comment escaped its literal container: {data}")


@dataclass
class ReaderAPI:
    client: TestClient
    container: AppContainer
    database_path: Path
    network: list[Mock]

    def path(self, run_id: str) -> str:
        return f"/runs/{quote(run_id, safe='')}/export"


@pytest.fixture
def reader_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ReaderAPI]:
    network = []
    for component, method in (
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
    ):
        guard = Mock(side_effect=AssertionError("The reader tests forbid live network traffic"))
        monkeypatch.setattr(component, method, guard)
        network.append(guard)
    database_path = tmp_path / "reader.sqlite3"
    application = create_app(offline_settings(database_path))
    with TestClient(application) as client:
        yield ReaderAPI(client, application.state.container, database_path, network)
    assert all(guard.call_count == 0 for guard in network)


def complete(
    api: ReaderAPI,
    chunks: list[Chunk] | None = None,
    query: str = QUERY,
    **options: object,
) -> EvidenceBundle:
    if chunks is None:
        chunks = [
            Chunk(
                chunk_id=f"chunk-{number}",
                document_id=f"paper-{number}",
                title=f"Synthetic note {number}",
                text=PASSAGE if number == 1 else "GraphRAG connects a second synthetic passage.",
                source=f"synthetic:note-{number}",
                metadata={"fixture": "synthetic-only", "url": "https://example.invalid/not-a-link"},
            )
            for number in (1, 2)
        ]
    api.container.document_store.add_documents(
        [Document(**chunk.model_dump(exclude={"chunk_id"})) for chunk in chunks], chunks
    )
    api.container.hybrid_retriever.add_chunks(chunks)
    result = api.client.post("/query", json={"query": query, **options})
    assert result.status_code == 200
    saved = result.json()["result"]
    assert saved["state"] == "DONE", saved
    return api.container.evidence_exporter.export(saved["run_id"])


def html_response(api: ReaderAPI, bundle: EvidenceBundle) -> httpx.Response:
    response = api.client.get(api.path(bundle.run_id), params={"format": "html"})
    assert response.status_code == 200, response.text[:1000]
    return response


def assert_links_resolve(root: Node) -> None:
    identifiers = [node.attrs["id"] for node in root.walk() if "id" in node.attrs]
    assert len(identifiers) == len(set(identifiers))
    for identifier in identifiers:
        assert re.fullmatch(r"[a-z]+(?:-[a-z]+|-[1-9][0-9]*)*", identifier)
    for link in root.by_tag("a"):
        href = link.attrs["href"]
        assert href.startswith("#") and href[1:] in identifiers


def assert_references(node: Node, chunk_ids: list[str], rank_by_id: dict[str, int]) -> None:
    rows = node.by_tag("li")
    assert len(rows) == len(chunk_ids)
    for row, chunk_id in zip(rows, chunk_ids, strict=True):
        assert row.by_class("literal")[0].text() == chunk_id
        if chunk_id in rank_by_id:
            assert [link.attrs["href"] for link in row.by_tag("a")] == [
                f"#source-{rank_by_id[chunk_id]}"
            ]
        else:
            assert not row.by_tag("a")
            assert "Unresolved reference" in row.text()


def test_real_route_is_a_secure_self_contained_semantic_attachment(reader_api: ReaderAPI) -> None:
    bundle = complete(reader_api)
    response = html_response(reader_api, bundle)
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    digest = hashlib.sha256(bundle.run_id.encode("utf-8")).hexdigest()[:16]
    assert (
        response.headers["content-disposition"] == f'attachment; filename="evidence-{digest}.html"'
    )
    parsed = ReaderDOM(response.text)
    assert parsed.declarations == ["doctype html"]
    root = parsed.root
    assert len(root.by_tag("main")) == 1 and len(root.by_tag("h1")) == 1
    assert len(root.by_tag("nav")) == 1
    assert root.by_tag("html")[0].attrs["lang"] == "en"
    assert root.by_tag("meta")[0].attrs == {"charset": "utf-8"}
    for identifier in (
        "summary",
        "question",
        "answer",
        "claims",
        "citations",
        "sources",
        "configuration",
        "plan",
        "request",
        "generation",
        "events",
    ):
        root.by_id(identifier)
    for identifier in ("plan", "request", "events"):
        assert root.by_id(identifier).tag == "details"
        assert root.by_id(identifier).children[0].tag == "summary"
    assert_links_resolve(root)
    style = root.by_tag("style")
    assert len(style) == 1 and not style[0].attrs
    style_hash = base64.b64encode(hashlib.sha256(style[0].text().encode()).digest()).decode()
    metas = [
        node
        for node in root.by_tag("meta")
        if node.attrs.get("http-equiv", "").lower() == "content-security-policy"
    ]
    assert len(metas) == 1
    csp = metas[0].attrs["content"]
    directives = {
        parts[0]: parts[1:] for directive in csp.split(";") if (parts := directive.split())
    }
    assert directives["default-src"] == ["'none'"]
    assert directives["style-src"] == [f"'sha256-{style_hash}'"]
    assert directives["base-uri"] == ["'none'"] and directives["form-action"] == ["'none'"]
    assert response.headers["content-security-policy"] == csp + "; frame-ancestors 'none'"
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "@media print" in style[0].text()
    assert "unicode-bidi: plaintext" in style[0].text()
    assert "overflow-wrap: anywhere" in style[0].text()
    assert not re.search(r"url\s*\(|@import", style[0].text(), re.IGNORECASE)


def test_keyboard_skip_link_precedes_navigation_and_targets_focusable_main(
    reader_api: ReaderAPI,
) -> None:
    bundle = complete(reader_api)
    root = ReaderDOM(html_response(reader_api, bundle).text).root
    first_link = root.by_tag("a")[0]
    assert first_link.text() == "Skip to main content"
    assert first_link.attrs["href"] == "#main"
    main = root.by_id("main")
    assert main.tag == "main" and main.attrs["tabindex"] == "-1"
    assert main is root.by_tag("main")[0]
    assert ".skip-link:focus" in root.by_tag("style")[0].text()
    assert_links_resolve(root)


def test_documented_readonly_python_example_executes_against_saved_evidence(
    reader_api: ReaderAPI,
    tmp_path: Path,
) -> None:
    bundle = complete(reader_api)
    guide = Path(__file__).resolve().parents[1] / "docs/guides/OFFLINE_EVIDENCE_READER_GUIDE.md"
    script = guide.read_text(encoding="utf-8").split("```python\n", 1)[1].split("```", 1)[0]
    destination = tmp_path / "documented-example.html"
    script = (
        script.replace(
            '"/absolute/path/existing-runs.sqlite3"', repr(str(reader_api.database_path))
        )
        .replace('"/absolute/path/new-evidence.html"', repr(str(destination)))
        .replace('"replace-with-a-completed-run-uuid"', repr(bundle.run_id))
    )
    begin = '    connection.execute("BEGIN")\n'
    assert begin in script
    script = script.replace(
        begin,
        '    connection.execute("BEGIN")\n'
        "    try:\n"
        "        connection.execute('DELETE FROM agent_events')\n"
        "    except sqlite3.OperationalError as exc:\n"
        "        if 'readonly' not in str(exc).lower():\n"
        "            raise\n"
        "    else:\n"
        "        raise AssertionError('The documented connection must reject writes')\n",
    )
    before = reader_api.database_path.read_bytes()
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert destination.read_bytes() == html_response(reader_api, bundle).content
    assert reader_api.database_path.read_bytes() == before


def test_exact_text_source_metadata_and_recorded_provenance_are_preserved(
    reader_api: ReaderAPI,
) -> None:
    bundle = complete(
        reader_api,
        max_chunks_per_document=1,
        min_evidence_documents=2,
        near_duplicate_threshold=1,
        document_ids=["paper-1", "paper-2"],
    )
    root = ReaderDOM(html_response(reader_api, bundle).text).root
    assert root.by_id("question").by_class("literal")[0].text() == bundle.query
    assert root.by_id("answer").by_class("answer-text")[0].text() == bundle.answer.answer
    assert bundle.run_id in root.by_id("summary").text()
    assert bundle.completed_at in root.by_id("summary").text()
    assert bundle.snapshot.context_sha256 in root.by_id("summary").text()
    assert "fake" in root.by_id("summary").text()
    assert "Not recorded" in root.by_id("summary").text()
    for name, expected in (
        ("plan", bundle.plan.model_dump(mode="json")),
        ("request", bundle.snapshot.request.model_dump(mode="json")),
        ("generation", bundle.generation.model_dump(mode="json")),
        ("document-scope", ["paper-1", "paper-2"]),
        ("runtime-configuration", bundle.configuration.model_dump(mode="json")),
        ("evidence-policy", bundle.plan.observation.evidence_policy.model_dump(mode="json")),
    ):
        assert json.loads(root.by_id(name).by_class("literal")[0].text()) == expected
    for source in bundle.snapshot.sources:
        node = root.by_id(f"source-{source.rank}")
        passage = node.by_class("passage")[0]
        assert passage.tag == "pre" and passage.attrs["dir"] == "auto"
        assert passage.children[0].tag == "code", "A direct pre newline is stripped by browsers"
        assert passage.text() == source.chunk.text
        assert hashlib.sha256(passage.text().encode("utf-8")).hexdigest() == source.text_sha256
        expected = source.model_dump(mode="json")
        expected["chunk"].pop("text")
        assert json.loads(node.by_class("source-metadata")[0].text()) == expected
    assert [json.loads(node.text()) for node in root.by_id("events").by_class("event-record")] == [
        event.model_dump(mode="json") for event in bundle.events
    ]
    for warning in [*bundle.warnings, *bundle.answer.warnings]:
        assert warning in root.text()


def test_all_claim_and_citation_links_keep_proposed_accepted_and_missing_distinct(
    reader_api: ReaderAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = 'missing-"><script>fixture</script>'

    async def generate(request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="GraphRAG connects evidence. Unsupported hypothesis.",
            parsed_claims=["GraphRAG connects evidence.", "Unsupported hypothesis."],
            citation_chunk_ids=[*reversed(request.citation_chunk_ids), missing],
            raw_provider="synthetic",
            model_name="recorded-model-not-todays-default",
        )

    def ground(answer_text: str, claims: list[Claim], retrieved_chunks: list[Chunk]) -> AgentAnswer:
        return AgentAnswer(
            answer=answer_text,
            claims=[
                Claim(text=claims[0].text, chunk_ids=[retrieved_chunks[0].chunk_id], grounded=True),
                Claim(text=claims[1].text, chunk_ids=[missing], grounded=False),
            ],
            citations=[
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    snippet=chunk.text[:240],
                )
                for chunk in reversed(retrieved_chunks)
            ]
            + [Citation(chunk_id=missing, document_id="missing", title="?", snippet="?")],
            ungrounded=True,
            warnings=["Synthetic unsupported hypothesis; do not infer support from a link."],
        )

    monkeypatch.setattr(reader_api.container.llm, "generate", generate)
    monkeypatch.setattr(reader_api.container.runner._executor._grounder, "ground", ground)
    bundle = complete(reader_api)
    root = ReaderDOM(html_response(reader_api, bundle).text).root
    ranks = {source.chunk.chunk_id: source.rank for source in bundle.snapshot.sources}
    for index, (claim, association) in enumerate(
        zip(bundle.answer.claims, bundle.claim_evidence, strict=True),
        start=1,
    ):
        node = root.by_id(f"claim-{index}")
        assert node.by_class("claim-text")[0].text() == claim.text
        assert_references(node.by_class("proposed")[0], association.proposed_chunk_ids, ranks)
        assert_references(node.by_class("accepted")[0], association.grounded_chunk_ids, ranks)
        assert "Proposed references" in node.text()
        assert "Accepted by recorded grounding" in node.text()
        assert str(claim.grounded).lower() in node.by_class("grounded-flag")[0].text()
    for index, citation in enumerate(bundle.answer.citations, start=1):
        node = root.by_id(f"citation-{index}")
        assert_references(node.by_class("citation-reference")[0], [citation.chunk_id], ranks)
        assert json.loads(node.by_class("citation-record")[0].text()) == citation.model_dump()
    assert not root.by_id("claim-2").by_class("accepted")[0].by_tag("a")
    assert "token overlap" in root.text() and "not scientific validation" in root.text()
    assert "recorded-model-not-todays-default" in root.by_id("summary").text()
    assert_links_resolve(root)


def test_hostile_strings_are_literal_in_every_surface_and_private_events_stay_omitted(
    reader_api: ReaderAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = Chunk(
        chunk_id=HOSTILE,
        document_id="doc-" + HOSTILE,
        title="title-" + HOSTILE,
        text=PASSAGE,
        source="javascript:" + HOSTILE,
        metadata={HOSTILE: HOSTILE},
    )
    run_id = 'run"\r\nX-Injected: fixture-\u7814\u7a76'
    monkeypatch.setattr("agent.runner.uuid4", lambda: run_id)

    async def generate(request: LLMRequest) -> LLMResponse:
        reader_api.container.event_log.append_event(
            agent_id="local-agent",
            run_id=run_id,
            event_type=HOSTILE,
            payload={"private": "UNEXPORTED-PRIVATE-DIAGNOSTIC"},
        )
        return LLMResponse(
            text=HOSTILE,
            parsed_claims=[HOSTILE],
            citation_chunk_ids=request.citation_chunk_ids,
            raw_provider=HOSTILE,
            model_name=HOSTILE,
        )

    def ground(answer_text: str, claims: list[Claim], retrieved_chunks: list[Chunk]) -> AgentAnswer:
        return AgentAnswer(
            answer=answer_text,
            claims=claims,
            citations=[
                Citation(
                    chunk_id=retrieved_chunks[0].chunk_id,
                    document_id=chunk.document_id,
                    title=HOSTILE,
                    snippet=HOSTILE,
                )
            ],
            warnings=[HOSTILE],
            ungrounded=True,
        )

    monkeypatch.setattr(reader_api.container.llm, "generate", generate)
    monkeypatch.setattr(reader_api.container.runner._executor._grounder, "ground", ground)
    bundle = complete(reader_api, [chunk], query=HOSTILE)
    with closing(sqlite3.connect(reader_api.database_path)) as connection, connection:
        connection.execute(
            "UPDATE agent_events SET agent_id = ? WHERE run_id = ?", (HOSTILE, run_id)
        )
    response = html_response(reader_api, bundle)
    assert "x-injected" not in response.headers
    assert "UNEXPORTED-PRIVATE-DIAGNOSTIC" not in response.text
    root = ReaderDOM(response.text).root
    assert root.by_id("question").by_class("literal")[0].text() == HOSTILE
    assert root.by_id("answer").by_class("answer-text")[0].text() == HOSTILE
    assert root.by_id("source-1").by_class("passage")[0].text() == PASSAGE
    assert HOSTILE in root.by_id("summary").text()
    assert "payload_omitted" in root.by_id("events").text()
    allowed = {
        "document",
        "html",
        "head",
        "meta",
        "title",
        "style",
        "body",
        "header",
        "main",
        "footer",
        "nav",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "a",
        "dl",
        "dt",
        "dd",
        "ul",
        "ol",
        "li",
        "strong",
        "pre",
        "code",
        "details",
        "summary",
        "bdi",
    }
    assert {node.tag for node in root.walk()} <= allowed
    assert len(root.by_tag("style")) == 1
    assert all(
        node.attrs.get("http-equiv", "").lower() != "refresh" for node in root.by_tag("meta")
    )
    for node in root.walk():
        assert not any(name.startswith("on") for name in node.attrs)
        assert not ({"src", "srcset", "style", "action", "data", "formaction"} & node.attrs.keys())
        assert all(HOSTILE not in value for value in node.attrs.values())
    assert_links_resolve(root)


def test_empty_recorded_evidence_is_explicit_not_a_missing_snapshot(reader_api: ReaderAPI) -> None:
    bundle = complete(reader_api, [])
    response = html_response(reader_api, bundle)
    root = ReaderDOM(response.text).root
    assert "no source passages" in root.by_id("sources").text()
    assert not root.by_id("sources").by_class("passage")
    assert "No final citations" in root.by_id("citations").text()
    assert "ungrounded" in root.by_id("answer").text().lower()
    assert_links_resolve(root)


def test_renderer_matches_route_without_mutating_bundle(reader_api: ReaderAPI) -> None:
    from agent.evidence_html import render_html

    bundle = complete(reader_api)
    before = bundle.model_dump_json()
    assert render_html(bundle).encode("utf-8") == html_response(reader_api, bundle).content
    assert bundle.model_dump_json() == before


def test_empty_recorded_model_identity_is_not_replaced_with_missing_label(
    reader_api: ReaderAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def generate(request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="GraphRAG connects evidence.",
            parsed_claims=["GraphRAG connects evidence."],
            citation_chunk_ids=request.citation_chunk_ids,
            raw_provider="synthetic",
            model_name="",
        )

    monkeypatch.setattr(reader_api.container.llm, "generate", generate)
    bundle = complete(reader_api)
    summary = ReaderDOM(html_response(reader_api, bundle).text).root.by_id("summary")
    fields = dict(
        zip(
            (node.text() for node in summary.by_tag("dt")),
            (node.text() for node in summary.by_tag("dd")),
            strict=True,
        )
    )
    assert fields["Recorded model"] == bundle.generation.model_name == ""


@pytest.mark.parametrize("character", ["&", "\u7814", "\r"])
def test_exact_rendered_utf8_byte_cap_includes_escaping_and_never_truncates(
    reader_api: ReaderAPI,
    character: str,
) -> None:
    from agent.evidence_html import MAX_HTML_BYTES, render_html

    assert MAX_HTML_BYTES == 4_194_304
    bundle = complete(reader_api)
    bundle.answer.answer = ""
    baseline = len(render_html(bundle).encode("utf-8"))
    cost = {"&": 5, "\u7814": 3, "\r": 5}[character]
    count, remainder = divmod(MAX_HTML_BYTES - baseline, cost)
    bundle.answer.answer = character * count + "x" * remainder
    assert len(render_html(bundle).encode("utf-8")) == MAX_HTML_BYTES
    bundle.answer.answer += "x"
    with pytest.raises(EvidenceExportError) as error:
        render_html(bundle)
    assert error.value.code == "html_export_too_large" and error.value.status_code == 413


def test_oversized_saved_answer_fails_html_only_with_explicit_error(reader_api: ReaderAPI) -> None:
    bundle = complete(reader_api)
    with closing(sqlite3.connect(reader_api.database_path)) as connection, connection:
        answer_event = connection.execute(
            "SELECT id, payload FROM agent_events WHERE run_id = ? AND "
            "json_extract(payload, '$.to_state') = 'ANSWERING'",
            (bundle.run_id,),
        ).fetchone()
        payload = json.loads(answer_event[1])
        payload["payload"]["answer"] = "&" * 900_000
        connection.execute(
            "UPDATE agent_events SET payload = ? WHERE id = ?",
            (json.dumps(payload), answer_event[0]),
        )
    response = reader_api.client.get(reader_api.path(bundle.run_id), params={"format": "html"})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "html_export_too_large"
    assert "4194304" in response.json()["detail"]["message"]
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-disposition" not in response.headers
    assert reader_api.client.get(reader_api.path(bundle.run_id)).status_code == 200
    assert (
        reader_api.client.get(
            reader_api.path(bundle.run_id),
            params={"format": "markdown"},
        ).status_code
        == 200
    )


def test_unrepresentable_nul_is_an_explicit_html_error_not_silent_text_corruption(
    reader_api: ReaderAPI,
) -> None:
    bundle = complete(
        reader_api,
        [
            Chunk(
                chunk_id="nul",
                document_id="nul-doc",
                title="Synthetic NUL fixture",
                text="GraphRAG\0 preserves this in JSON, but HTML cannot.",
                source="synthetic:nul",
            )
        ],
    )
    response = reader_api.client.get(reader_api.path(bundle.run_id), params={"format": "html"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "html_text_not_representable"
    assert (
        reader_api.client.get(reader_api.path(bundle.run_id)).json()["snapshot"]["sources"][0][
            "chunk"
        ]["text"]
        == bundle.snapshot.sources[0].chunk.text
    )


def test_unpaired_surrogate_is_rejected_explicitly_by_python_renderer(
    reader_api: ReaderAPI,
) -> None:
    from agent.evidence_html import render_html

    bundle = complete(reader_api)
    bundle.answer.answer = "\ud800"
    with pytest.raises(EvidenceExportError) as error:
        render_html(bundle)
    assert error.value.code == "html_text_not_representable"
    assert "Inspect the saved record" in str(error.value)


def test_old_valid_snapshots_without_optional_scope_or_policy_still_render(
    reader_api: ReaderAPI,
) -> None:
    bundle = complete(reader_api)
    with closing(sqlite3.connect(reader_api.database_path)) as connection, connection:
        rows = connection.execute(
            "SELECT id, payload FROM agent_events WHERE run_id = ?", (bundle.run_id,)
        ).fetchall()
        for event_id, raw in rows:
            payload = json.loads(raw)
            candidates = [payload]
            if isinstance(payload.get("payload"), dict):
                candidates.append(payload["payload"])
            for candidate in list(candidates):
                if isinstance(candidate.get("observation"), dict):
                    candidates.append(candidate["observation"])
            for candidate in candidates:
                candidate.pop("document_ids", None)
                candidate.pop("evidence_policy", None)
            connection.execute(
                "UPDATE agent_events SET payload = ? WHERE id = ?", (json.dumps(payload), event_id)
            )
    root = ReaderDOM(html_response(reader_api, bundle).text).root
    assert json.loads(root.by_id("document-scope").by_class("literal")[0].text()) is None
    assert json.loads(root.by_id("evidence-policy").by_class("literal")[0].text()) is None
    assert len(root.by_id("sources").by_class("passage")) == len(bundle.snapshot.sources)


@pytest.mark.parametrize(
    ("state", "code", "status"),
    [
        (None, "run_not_found", 404),
        (AgentState.REASONING, "run_incomplete", 409),
        (AgentState.ERROR, "run_failed", 409),
        (AgentState.DONE, "snapshot_unavailable", 409),
    ],
)
def test_html_reuses_missing_incomplete_failed_and_legacy_export_errors(
    reader_api: ReaderAPI,
    state: AgentState | None,
    code: str,
    status: int,
) -> None:
    if state is not None:
        reader_api.container.event_log.append_transition(
            StateTransition(
                agent_id="legacy",
                run_id="unexportable",
                from_state=AgentState.IDLE,
                to_state=state,
            )
        )
    path = reader_api.path("unexportable")
    existing = reader_api.client.get(path)
    response = reader_api.client.get(path, params={"format": "html"})
    assert response.status_code == existing.status_code == status
    assert response.json() == existing.json()
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("damage", ["json", "version", "digest", "rank", "generation"])
def test_html_never_renders_corrupt_or_unsupported_saved_records(
    reader_api: ReaderAPI,
    damage: str,
) -> None:
    bundle = complete(reader_api)
    with closing(sqlite3.connect(reader_api.database_path)) as connection, connection:
        row = connection.execute(
            "SELECT id, payload FROM agent_events WHERE run_id = ? AND event_type = ?",
            (bundle.run_id, "evidence_snapshot"),
        ).fetchone()
        payload = json.loads(row[1])
        if damage == "version":
            payload["schema_version"] = "999"
        elif damage == "digest":
            payload["sources"][0]["text_sha256"] = "0" * 64
        elif damage == "rank":
            payload["sources"][0]["rank"] = 50
        if damage == "generation":
            connection.execute(
                "DELETE FROM agent_events WHERE run_id = ? AND event_type = ?",
                (bundle.run_id, "generation_record"),
            )
        else:
            connection.execute(
                "UPDATE agent_events SET payload = ? WHERE id = ?",
                ("{invalid" if damage == "json" else json.dumps(payload), row[0]),
            )
    response = reader_api.client.get(reader_api.path(bundle.run_id), params={"format": "html"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_run_record"
    assert response.json() == reader_api.client.get(reader_api.path(bundle.run_id)).json()


def test_html_and_existing_formats_survive_corpus_removal_and_restart_without_work(
    reader_api: ReaderAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = complete(reader_api)
    path = reader_api.path(bundle.run_id)
    before = {
        format: reader_api.client.get(path, params={"format": format})
        for format in ("json", "markdown", "html")
    }
    assert all(response.status_code == 200 for response in before.values())
    assert before["json"].content == (
        json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    assert before["markdown"].content == render_markdown(bundle).encode("utf-8")
    assert reader_api.client.get(path).content == before["json"].content
    events_before = reader_api.container.event_log.list_events()
    with closing(sqlite3.connect(reader_api.database_path)) as connection, connection:
        connection.execute("DELETE FROM entity_edges")
        connection.execute("DELETE FROM entity_mentions")
        connection.execute("DELETE FROM graph_chunks")
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM documents")
    application = create_app(offline_settings(reader_api.database_path))
    container: AppContainer = application.state.container
    assert container.document_store.list_chunks() == []
    guards = []
    for component, method in (
        (container.runner, "run"),
        (container.runner, "preview"),
        (container.llm, "generate"),
        (container.hybrid_retriever, "retrieve"),
        (container.document_store, "list_chunks"),
        (container.document_store, "add_documents"),
        (container.graph_store, "chunks_for_entities"),
        (container.event_log, "append_event"),
        (container.event_log, "append_transition"),
    ):
        guard = Mock(
            side_effect=AssertionError("Export attempted generation, retrieval or a write")
        )
        monkeypatch.setattr(component, method, guard)
        guards.append(guard)
    saved_bytes = reader_api.database_path.read_bytes()
    with TestClient(application) as client:
        for format, original in before.items():
            response = client.get(path, params={"format": format})
            assert response.status_code == 200
            assert response.content == original.content
            assert (
                response.headers["content-disposition"] == original.headers["content-disposition"]
            )
        assert client.get(path).content == before["json"].content
    assert all(guard.call_count == 0 for guard in guards)
    assert container.event_log.list_events() == events_before
    assert reader_api.database_path.read_bytes() == saved_bytes
    process = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from storage.event_log import SQLiteEventLog\n"
            "from storage.evidence_export import EvidenceExporter\n"
            "from agent.evidence_html import render_html\n"
            "bundle = EvidenceExporter(SQLiteEventLog(sys.argv[1])).export(sys.argv[2])\n"
            "sys.stdout.buffer.write(render_html(bundle).encode('utf-8'))\n",
            str(reader_api.database_path),
            bundle.run_id,
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8")
    assert process.stdout == before["html"].content


def test_later_events_do_not_rewrite_completed_html(reader_api: ReaderAPI) -> None:
    bundle = complete(reader_api)
    before = html_response(reader_api, bundle)
    reader_api.container.event_log.append_event(
        agent_id="local-agent",
        run_id=bundle.run_id,
        event_type="later_note",
        payload={"note": "Not part of this frozen reader"},
    )
    assert html_response(reader_api, bundle).content == before.content


def test_openapi_adds_html_without_changing_default_or_json_schema(reader_api: ReaderAPI) -> None:
    operation = reader_api.client.get("/openapi.json").json()["paths"]["/runs/{run_id}/export"][
        "get"
    ]
    format = next(
        parameter for parameter in operation["parameters"] if parameter["name"] == "format"
    )
    assert format["schema"]["enum"] == ["json", "markdown", "html"]
    assert format["schema"]["default"] == "json"
    content = operation["responses"]["200"]["content"]
    assert content["application/json"]["schema"]["$ref"].endswith("/EvidenceBundle")
    assert content["text/html"]["schema"] == {"type": "string"}
    assert content["text/markdown"]["schema"] == {"type": "string"}
    assert {"404", "409", "413", "422"} <= operation["responses"].keys()
