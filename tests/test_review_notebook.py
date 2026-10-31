"""Native, read-only human opinions beside exact frozen saved-run evidence."""

import logging
import sqlite3
from contextlib import closing
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from agent.evidence import EvidenceBundle, EvidenceSource
from agent.models import AgentState
from agent.review_models import AnswerReviewError, ReviewDecision, ReviewSubmission
from api import explorer
from api.application import create_app
from storage.evidence_annotations import (
    AnnotationError,
    AnnotationSubmission,
    SavedEvidenceAnnotation,
)
from storage.evidence_export import MAX_EVENT_BYTES, MAX_RUN_EVENTS, EvidenceExportError
from storage.run_history import MAX_EVENT_ID
from tests.test_corpus_explorer import (
    HOSTILE,
    ExplorerAPI,
    ExplorerDOM,
    assert_headers,
    href,
    identities,
    no_work,
    values,
)
from tests.test_corpus_explorer import api as api
from tests.test_run_comparison import ComparisonAPI, _save, _source
from tests.test_run_explorer import complete, reader_headers, record

NOTEBOOK = "/explore/run/notebook"
ANNOTATION = "/explore/run/annotation"
COMMENT = "\n  Human opinion, not proof. <script>inert</script> & cafe\u0301.\r\n"
NOTE = "  Inspect this exact quote. <b>Literal note</b>.\r\n"
TEXT = "\nCaf\u00e9 \U0001f52c e\u0301.\r\nRepeated quote. Repeated quote.\r" + HOSTILE
PRIVATE = "PRIVATE_DIAGNOSTIC"


def fixture_run(
    api: ExplorerAPI,
    *,
    run_id: str = "saved",
    chunk_id: str = "chunk/one",
    document_id: str = "paper/one",
    text: str = TEXT,
) -> EvidenceBundle:
    _save(
        ComparisonAPI(api.app, api.client, api.container, api.path),
        run_id,
        [_source(chunk_id, document_id=document_id, text=text, source="javascript:alert(1)")],
    )
    return api.container.saved_run_evidence.export(run_id)


def opinion(api: ExplorerAPI, bundle: EvidenceBundle, comment: str = COMMENT) -> None:
    api.container.answer_reviews.create(
        bundle.run_id,
        ReviewSubmission(
            review_id=uuid4(),
            decision=ReviewDecision.NEEDS_REVISION,
            comment=comment,
            cited_chunk_ids=(bundle.snapshot.sources[0].chunk.chunk_id,),
        ),
    )


def annotate(
    api: ExplorerAPI,
    bundle: EvidenceBundle,
    quote: str = "Repeated quote",
    *,
    start: int | None = None,
    note: str = NOTE,
) -> SavedEvidenceAnnotation:
    source = bundle.snapshot.sources[0]
    start = source.chunk.text.rindex(quote) if start is None else start
    saved, _ = api.container.evidence_annotations.create(
        bundle.run_id,
        AnnotationSubmission(
            annotation_id=uuid4(),
            document_id=source.chunk.document_id,
            chunk_id=source.chunk.chunk_id,
            source_text_sha256=source.text_sha256,
            start=start,
            end=start + len(quote),
            quote=quote,
            note=note,
        ),
    )
    return saved


def mutate(api: ExplorerAPI, sql: str, params: tuple[object, ...] = ()) -> None:
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(sql, params)


def test_completed_run_exposes_exact_human_review_notebook(api: ExplorerAPI) -> None:
    bundle = complete(api)
    source = bundle.snapshot.sources[0]
    for decision in ReviewDecision:
        api.container.answer_reviews.create(
            bundle.run_id,
            ReviewSubmission(
                review_id=uuid4(),
                decision=decision,
                comment=COMMENT,
                cited_chunk_ids=(source.chunk.chunk_id,),
            ),
        )
    quote = "Literal <script>not executable</script>"
    start = source.chunk.text.index(quote)
    saved, _ = api.container.evidence_annotations.create(
        bundle.run_id,
        AnnotationSubmission(
            annotation_id=uuid4(),
            document_id=source.chunk.document_id,
            chunk_id=source.chunk.chunk_id,
            source_text_sha256=source.text_sha256,
            start=start,
            end=start + len(quote),
            quote=quote,
            note=NOTE,
        ),
    )

    notebook = api.page(NOTEBOOK, run_id=bundle.run_id)
    assert values(notebook, "review-decision") == [item.value for item in reversed(ReviewDecision)]
    assert values(notebook, "review-comment") == [COMMENT] * 3
    assert values(notebook, "annotation-note") == [NOTE]
    assert values(notebook, "annotation-quote") == [quote]
    assert identities(notebook, "run-id") == [bundle.run_id]
    assert not notebook.by_tag("script")
    source_page = api.page(href(notebook, f"annotation-source-{saved.sequence}"))
    assert values(source_page, "passage-text") == [source.chunk.text]
    assert source_page.by_id("annotation-quote").text() == quote
    assert values(source_page, "annotation-note") == [NOTE]


def test_empty_notebook_is_discoverable_from_saved_run_browser(api: ExplorerAPI) -> None:
    bundle = complete(api)
    notebook = api.page(NOTEBOOK, run_id=bundle.run_id)
    assert "No answer reviews recorded" in notebook.text()
    assert "No evidence annotations recorded" in notebook.text()
    history = api.page("/explore/runs", state="DONE", limit=1)
    discovered = api.page(href(history, "review-notebook-1"))
    assert identities(discovered, "run-id") == [bundle.run_id]
    assert href(discovered, "back-to-runs") == "/explore/runs?limit=1&state=DONE&catalog_limit=20"
    reader = reader_headers(api.client.get(href(history, "inspect-run-1")))
    assert href(reader, "review-notebook") == href(history, "review-notebook-1")


def test_independent_paging_retains_both_histories_and_exact_saved_run_return(
    api: ExplorerAPI,
) -> None:
    bundle = fixture_run(api)
    for number in range(3):
        opinion(api, bundle, f"Review {number}")
        annotate(api, bundle, note=f"Note {number}")
    params = {
        "run_id": bundle.run_id,
        "review_limit": 1,
        "annotation_limit": 1,
        "limit": 2,
        "cursor": MAX_EVENT_ID,
        "state": "DONE",
        "question": " literal %_\U0001f52c ",
        "catalog_limit": 3,
        "source": "literal ?#/&",
        "title": "Exact title",
        "catalog_cursor": "paper/../\u7814?x",
    }
    first = api.page(NOTEBOOK, **params)
    assert values(first, "review-comment") == ["Review 2"]
    assert values(first, "annotation-note") == ["Note 2"]
    older_reviews = api.page(href(first, "reviews-next-page"))
    assert values(older_reviews, "review-comment") == ["Review 1"]
    assert values(older_reviews, "annotation-note") == ["Note 2"]
    both_url = href(older_reviews, "annotations-next-page")
    both = api.page(both_url)
    assert values(both, "review-comment") == ["Review 1"]
    assert values(both, "annotation-note") == ["Note 1"]
    both_params = parse_qs(urlsplit(both_url).query)
    annotation_sequence = values(both, "annotation-sequence")[0]
    selected_url = href(both, f"annotation-source-{annotation_sequence}")
    opinion(api, bundle, "Appended review")
    annotate(api, bundle, note="Appended annotation")
    selected = api.page(selected_url)
    assert values(selected, "annotation-note") == ["Note 1"]
    assert href(selected, "back-to-notebook") == both_url
    assert values(api.page(both_url), "review-comment") == ["Review 1"]
    assert values(api.page(href(both, "reviews-first-page")), "review-comment") == [
        "Appended review"
    ]
    assert values(api.page(href(both, "reviews-first-page")), "annotation-note") == ["Note 1"]
    assert values(api.page(href(both, "annotations-first-page")), "annotation-note") == [
        "Appended annotation"
    ]
    for root in (first, both, selected):
        returned = parse_qs(urlsplit(href(root, "back-to-runs")).query)
        assert returned == {
            key: [str(value)]
            for key, value in params.items()
            if key not in {"run_id", "review_limit", "annotation_limit"}
        }
        assert "not one shared snapshot" in both.text()
    for form in both.by_tag("form"):
        fields = {node.attrs["name"]: node.attrs["value"] for node in form.by_tag("input")}
        kind = "review" if "review_cursor" not in fields else "annotation"
        assert fields[f"{kind}_limit"] == "1"
        assert f"{kind}_cursor" not in fields
        other = "annotation" if kind == "review" else "review"
        assert fields[f"{other}_cursor"] == both_params[f"{other}_cursor"][0]
        assert form.attrs["method"] == "get"
        fields[f"{kind}_limit"] = "2"
        submitted = api.page(form.attrs["action"], **fields)
        assert len(values(submitted, f"{kind}-sequence")) == 2


@pytest.mark.parametrize("quote", ["Repeated quote", "\U0001f52c", "\u0301", "\r\n", HOSTILE])
def test_exact_highlight_preserves_unicode_cr_and_only_the_selected_occurrence(
    api: ExplorerAPI, quote: str
) -> None:
    bundle = fixture_run(api)
    saved = annotate(api, bundle, quote)
    root = api.page(ANNOTATION, run_id=bundle.run_id, annotation_id=str(saved.annotation_id))
    passage = root.by_class("passage-text")[0]
    assert passage.text() == TEXT
    assert len(passage.by_tag("mark")) == 1
    assert passage.by_id("annotation-quote").text() == quote
    code = passage.by_tag("code")[0]
    mark_index = code.children.index(passage.by_id("annotation-quote"))
    assert (
        "".join(child for child in code.children[:mark_index] if isinstance(child, str))
        == TEXT[: saved.start]
    )
    assert (
        "".join(child for child in code.children[mark_index + 1 :] if isinstance(child, str))
        == TEXT[saved.end :]
    )
    assert root.by_id("annotation-quote").attrs["tabindex"] == "-1"
    assert values(root, "quote-span") == [f"[{saved.start}, {saved.end})"]
    for tag in ("script", "img", "base", "iframe", "object", "svg"):
        assert not root.by_tag(tag)
    assert len({node.attrs["id"] for node in root.walk() if "id" in node.attrs}) == len(
        [node for node in root.walk() if "id" in node.attrs]
    )
    for node in root.walk():
        assert not any(name.startswith("on") or name == "style" for name in node.attrs)
    for anchor in root.by_tag("a"):
        path = urlsplit(anchor.attrs["href"]).path
        assert not path or path.startswith("/explore")
    full = reader_headers(api.client.get(href(root, "full-frozen-source")))
    assert full.by_id("source-1").by_class("passage")[0].text() == TEXT


@pytest.mark.parametrize(
    "identifier",
    [
        ".",
        "..",
        "paper/../\u7814?draft#v1",
        "//example.invalid/a",
        "'\"?%#&+\\",
        "cafe\u0301/\u202e",
    ],
)
def test_literal_run_and_source_ids_round_trip_without_path_interpretation(
    api: ExplorerAPI, identifier: str
) -> None:
    bundle = fixture_run(api, run_id=identifier, chunk_id=identifier, document_id=identifier)
    opinion(api, bundle)
    saved = annotate(api, bundle)
    history = api.page("/explore/runs")
    notebook = api.page(href(history, "review-notebook-1"))
    assert identities(notebook, "run-id") == [identifier]
    assert identities(notebook, "chunk-id") == [identifier, identifier]
    assert identities(notebook, "document-id") == [identifier, identifier]
    source_url = href(notebook, f"annotation-source-{saved.sequence}")
    assert urlsplit(source_url).path == ANNOTATION
    assert parse_qs(urlsplit(source_url).query)["run_id"] == [identifier]
    assert values(api.page(source_url), "passage-text") == [TEXT]


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_mount_and_root_path_cover_links_forms_and_sanitized_errors(
    api: ExplorerAPI, prefix: str, mounted: bool
) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    saved = annotate(api, bundle)
    application = api.app
    root_path = prefix
    if mounted:
        application = FastAPI()
        application.mount("/library", api.app)
        root_path = "/prefix"
    with TestClient(application, root_path=root_path) as client:
        notebook_response = client.get(prefix + NOTEBOOK, params={"run_id": bundle.run_id})
        assert notebook_response.status_code == 200
        root = ExplorerDOM(notebook_response.text).root
        source_response = client.get(href(root, f"annotation-source-{saved.sequence}"))
        assert source_response.status_code == 200
        error = client.get(
            prefix + ANNOTATION, params={"run_id": bundle.run_id, "annotation_id": str(uuid4())}
        )
        invalid = client.get(prefix + NOTEBOOK + "?review_limit=0")
        assert error.status_code == 404 and invalid.status_code == 422
        for response in (notebook_response, source_response, error, invalid):
            assert_headers(response)
            page = ExplorerDOM(response.text).root
            for anchor in page.by_tag("a"):
                path = urlsplit(anchor.attrs["href"]).path
                assert not path or path.startswith(prefix + "/")
            for form in page.by_tag("form"):
                assert form.attrs["action"].startswith(prefix + "/")


def test_reads_survive_corpus_changes_deletion_restart_without_any_runtime_work(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = complete(api)
    opinion(api, bundle)
    saved = annotate(api, bundle, "Synthetic graph evidence")
    notebook_url = NOTEBOOK + "?" + urlencode({"run_id": bundle.run_id})
    notebook_response = api.client.get(notebook_url)
    assert notebook_response.status_code == 200
    root = ExplorerDOM(notebook_response.text).root
    source_url = href(root, f"annotation-source-{saved.sequence}")
    source_response = api.client.get(source_url)
    assert source_response.status_code == 200
    events = api.container.event_log.list_events()
    for change in ("unchanged", "changed", "deleted"):
        if change == "changed":
            mutate(api, "UPDATE chunks SET text = 'CHANGED_CURRENT_CORPUS'")
        if change == "deleted":
            for table in ("entity_edges", "entity_mentions", "graph_chunks", "chunks", "documents"):
                mutate(api, f"DELETE FROM {table}")  # noqa: S608
        application = api.app if change == "unchanged" else create_app(offline_settings(api.path))
        container = application.state.container
        before = api.path.read_bytes()
        with (
            TestClient(application) as client,
            browsing_guards(container) as guards,
            monkeypatch.context() as patch,
        ):
            for target, method in (
                (container.event_log, "list_events"),
                (container.evidence_exporter, "export"),
                (container.document_store, "list_chunks"),
                (container.answer_reviews, "create"),
                (container.notebook_reviews, "create"),
                (container.evidence_annotations, "create"),
            ):
                patch.setattr(target, method, no_work)
            assert client.get(notebook_url).content == notebook_response.content
            assert client.get(source_url).content == source_response.content
            assert not any(guard.call_count for guard in guards.values())
        assert api.path.read_bytes() == before
        assert api.container.event_log.list_events() == events


@pytest.mark.parametrize(
    "suffix",
    [
        "review_limit=0",
        "review_limit=101",
        "review_limit=",
        "review_limit=1.0",
        "review_limit=true",
        "review_limit=%EF%BC%91",
        "review_cursor=",
        "review_cursor=0",
        "review_cursor=-1",
        f"review_cursor={MAX_EVENT_ID + 1}",
        "annotation_limit=0",
        "annotation_limit=101",
        "annotation_cursor=",
        "annotation_cursor=1e1",
        "annotation_cursor=%2B1",
        "review_limit=1&review_limit=2",
        "annotation_cursor=1&annotation_cursor=2",
        "unknown=PRIVATE_DIAGNOSTIC",
        "state=done",
        "limit=0",
        "cursor=0",
        "catalog_limit=101",
        "source=%FF",
        "title=%ED%A0%80",
        "query=" + "x" * 16_384,
    ],
)
@pytest.mark.parametrize("path", [NOTEBOOK, ANNOTATION])
def test_bad_queries_fail_before_reads_without_echoing_user_text(
    api: ExplorerAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    path: str,
    suffix: str,
) -> None:
    for target, method in (
        (api.container.saved_run_evidence, "export"),
        (api.container.notebook_reviews, "list_reviews"),
        (api.container.evidence_annotations, "list_annotations"),
        (api.container.evidence_annotations, "get_annotation"),
    ):
        monkeypatch.setattr(target, method, no_work)
    query = "run_id=PRIVATE_DIAGNOSTIC"
    if path == ANNOTATION:
        query += "&annotation_id=00000000-0000-4000-8000-000000000001"
    response = api.client.get(path + "?" + query + "&" + suffix)
    assert response.status_code == 422
    assert_headers(response)
    assert PRIVATE not in response.text + caplog.text
    assert "invalid_notebook_request" in caplog.text


@pytest.mark.parametrize(
    "url",
    [
        NOTEBOOK,
        NOTEBOOK + "?run_id=",
        NOTEBOOK + "?run_id=" + "x" * 257,
        NOTEBOOK + "?run_id=a&run_id=b",
        NOTEBOOK + "?run_id=%FF",
        NOTEBOOK + "?run_id=a&annotation_id=00000000-0000-4000-8000-000000000001",
        ANNOTATION + "?run_id=a",
        ANNOTATION + "?run_id=a&annotation_id=",
        ANNOTATION + "?run_id=a&annotation_id=PRIVATE_DIAGNOSTIC",
    ],
)
def test_invalid_id_selection_is_a_sanitized_422(api: ExplorerAPI, url: str) -> None:
    response = api.client.get(url)
    assert response.status_code == 422
    assert_headers(response)
    assert PRIVATE not in response.text


@pytest.mark.parametrize("path", [NOTEBOOK, ANNOTATION])
@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_native_notebook_has_no_write_methods(api: ExplorerAPI, path: str, method: str) -> None:
    before = api.path.read_bytes()
    response = api.client.request(method, path, json={"comment": PRIVATE})
    assert response.status_code == 405 and response.headers["allow"] == "GET"
    assert response.headers["cache-control"] == "no-store"
    assert PRIVATE not in response.text
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("path", [NOTEBOOK, ANNOTATION])
@pytest.mark.parametrize(
    "state,code",
    [
        (None, "run_not_found"),
        (AgentState.REASONING, "run_incomplete"),
        (AgentState.ERROR, "run_failed"),
        (AgentState.DONE, "snapshot_unavailable"),
    ],
)
def test_nonexportable_run_is_not_an_empty_notebook(
    api: ExplorerAPI,
    caplog: pytest.LogCaptureFixture,
    path: str,
    state: AgentState | None,
    code: str,
) -> None:
    if state is not None:
        record(api, "unavailable", state)
    params = {"run_id": "unavailable"}
    if path == ANNOTATION:
        params["annotation_id"] = str(uuid4())
    response = api.client.get(path, params=params)
    assert response.status_code == (404 if state is None else 409)
    assert_headers(response)
    assert code in caplog.text
    assert not ExplorerDOM(response.text).root.by_tag("article")


def test_annotation_links_are_run_scoped_stable_uuid_lookups(api: ExplorerAPI) -> None:
    first = fixture_run(api)
    second = fixture_run(api, run_id="another")
    saved = annotate(api, first)
    for run_id, annotation_id in ((second.run_id, saved.annotation_id), (first.run_id, uuid4())):
        response = api.client.get(
            ANNOTATION, params={"run_id": run_id, "annotation_id": str(annotation_id)}
        )
        assert response.status_code == 404
        assert "No annotation with this exact ID" in response.text
        assert NOTE not in response.text
    record, source = api.container.evidence_annotations.get_annotation(
        first.run_id, saved.annotation_id
    )
    assert record == saved and source == first.snapshot.sources[0]


@pytest.mark.parametrize(
    "sql,params,code",
    [
        ("UPDATE answer_reviews SET created_at = ?", (PRIVATE,), "invalid_review_record"),
        (
            "UPDATE answer_reviews SET cited_chunk_ids = '[\"missing\"]'",
            (),
            "invalid_review_record",
        ),
        ("UPDATE evidence_annotations SET quote = ?", (PRIVATE,), "invalid_annotation_record"),
        ("UPDATE evidence_annotations SET start = end", (), "invalid_annotation_record"),
        (
            "UPDATE evidence_annotations SET source_text_sha256 = ?",
            ("0" * 64,),
            "invalid_annotation_record",
        ),
        (
            "UPDATE evidence_annotations SET document_id = ?",
            (PRIVATE,),
            "invalid_annotation_record",
        ),
        ("UPDATE evidence_annotations SET created_at = ?", (PRIVATE,), "invalid_annotation_record"),
        (
            "UPDATE agent_events SET payload = '{}' WHERE event_type = 'evidence_snapshot'",
            (),
            "invalid_run_record",
        ),
    ],
)
def test_corrupt_opinions_or_frozen_evidence_fail_without_partial_html(
    api: ExplorerAPI,
    caplog: pytest.LogCaptureFixture,
    sql: str,
    params: tuple[object, ...],
    code: str,
) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    saved = annotate(api, bundle)
    mutate(api, sql, params)
    response = api.client.get(NOTEBOOK, params={"run_id": bundle.run_id})
    assert response.status_code == 409
    assert_headers(response)
    assert code in caplog.text
    assert PRIVATE not in response.text + caplog.text
    assert not ExplorerDOM(response.text).root.by_tag("article")
    if "answer_reviews" not in sql:
        response = api.client.get(
            ANNOTATION, params={"run_id": bundle.run_id, "annotation_id": str(saved.annotation_id)}
        )
        assert response.status_code == 409
        assert_headers(response)
        assert PRIVATE not in response.text + caplog.text


@pytest.mark.parametrize("kind", ["review", "annotation"])
def test_corrupt_lookahead_is_not_silently_hidden_by_pagination(
    api: ExplorerAPI,
    kind: str,
) -> None:
    bundle = fixture_run(api)
    for _ in range(2):
        opinion(api, bundle)
        annotate(api, bundle)
    if kind == "review":
        mutate(api, "UPDATE answer_reviews SET created_at = 'invalid' WHERE id = 1")
    else:
        mutate(api, "UPDATE evidence_annotations SET quote = 'invalid' WHERE sequence = 1")
    response = api.client.get(
        NOTEBOOK, params={"run_id": bundle.run_id, "review_limit": 1, "annotation_limit": 1}
    )
    assert response.status_code == 409
    assert not ExplorerDOM(response.text).root.by_tag("article")


@pytest.mark.parametrize("path", [NOTEBOOK, ANNOTATION])
@pytest.mark.parametrize("overflow", ["events", "payload"])
def test_opinion_reads_use_existing_frozen_event_byte_and_count_bounds(
    api: ExplorerAPI,
    caplog: pytest.LogCaptureFixture,
    path: str,
    overflow: str,
) -> None:
    bundle = fixture_run(api)
    saved = annotate(api, bundle)
    if overflow == "events":
        for _ in range(MAX_RUN_EVENTS):
            api.container.event_log.append_event("synthetic-agent", bundle.run_id, "later", {})
    else:
        api.container.event_log.append_event(
            "synthetic-agent", bundle.run_id, "later", {"text": "x" * MAX_EVENT_BYTES}
        )
    params = {"run_id": bundle.run_id}
    if path == ANNOTATION:
        params["annotation_id"] = str(saved.annotation_id)
    response = api.client.get(path, params=params)
    assert response.status_code == 409
    assert "evidence_read_limit_exceeded" in caplog.text
    assert_headers(response)


def test_notebook_limits_are_default20_max100_and_exclusive(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    for number in range(101):
        opinion(api, bundle, f"Review {number}")
        annotate(api, bundle, note=f"Note {number}")
    default = api.page(NOTEBOOK, run_id=bundle.run_id)
    assert len(values(default, "review-sequence")) == 20
    assert len(values(default, "annotation-sequence")) == 20
    maximum = api.page(NOTEBOOK, run_id=bundle.run_id, review_limit=100, annotation_limit=100)
    assert len(values(maximum, "review-sequence")) == 100
    assert len(values(maximum, "annotation-sequence")) == 100
    assert values(api.page(href(maximum, "reviews-next-page")), "review-comment") == ["Review 0"]
    assert values(api.page(href(maximum, "annotations-next-page")), "annotation-note") == ["Note 0"]
    empty = api.page(NOTEBOOK, run_id=bundle.run_id, review_cursor=1, annotation_cursor=1)
    assert "No answer reviews remain" in empty.text()
    assert "No evidence annotations remain" in empty.text()
    assert not empty.by_class("next-page")


def test_escaped_html_overflow_does_not_trim_or_drop_review_text(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    for _ in range(60):
        opinion(api, bundle, "&" * 4000)
    response = api.client.get(NOTEBOOK, params={"run_id": bundle.run_id, "review_limit": 100})
    assert response.status_code == 413
    assert_headers(response)
    assert "Nothing was truncated or skipped" in response.text
    assert not ExplorerDOM(response.text).root.by_tag("article")
    smaller = api.page(NOTEBOOK, run_id=bundle.run_id, review_limit=20)
    assert values(smaller, "review-comment") == ["&" * 4000] * 20


@pytest.mark.parametrize("path", [NOTEBOOK, ANNOTATION])
def test_exact_utf8_html_byte_ceiling_includes_navigation_and_markup(
    api: ExplorerAPI,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    saved = annotate(api, bundle)
    params = {"run_id": bundle.run_id}
    if path == ANNOTATION:
        params["annotation_id"] = str(saved.annotation_id)
    response = api.client.get(path, params=params)
    assert response.status_code == 200 and len(response.content) > len(response.text)
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content))
    assert api.client.get(path, params=params).content == response.content
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content) - 1)
    assert api.client.get(path, params=params).status_code == 413


def test_annotation_json_response_bound_remains_in_force(
    api: ExplorerAPI,
    caplog: pytest.LogCaptureFixture,
) -> None:
    quote = "\U0001f52c" * 1000
    bundle = fixture_run(api, text=quote)
    for _ in range(40):
        annotate(api, bundle, quote, note="\U0001f52c" * 1000)
    response = api.client.get(NOTEBOOK, params={"run_id": bundle.run_id, "annotation_limit": 100})
    assert response.status_code == 413
    assert "annotation_response_too_large" in caplog.text
    assert not ExplorerDOM(response.text).root.by_tag("article")
    assert len(values(api.page(NOTEBOOK, run_id=bundle.run_id), "annotation-note")) == 20


@pytest.mark.parametrize("path", [NOTEBOOK, ANNOTATION])
def test_nul_quote_cannot_be_silently_replaced_by_html_parsing(
    api: ExplorerAPI,
    caplog: pytest.LogCaptureFixture,
    path: str,
) -> None:
    bundle = fixture_run(api, text="before\0after")
    saved = annotate(api, bundle, "\0")
    params = {"run_id": bundle.run_id}
    if path == ANNOTATION:
        params["annotation_id"] = str(saved.annotation_id)
    response = api.client.get(path, params=params)
    assert response.status_code == 409
    assert "html_text_not_representable" in caplog.text
    assert not ExplorerDOM(response.text).root.by_tag("article")


@pytest.mark.parametrize("failure", ["review", "annotation", "sqlite", "unknown"])
def test_storage_failures_are_sanitized_and_logged(
    api: ExplorerAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    bundle = fixture_run(api)

    def fail(*args: object, **kwargs: object) -> None:
        if failure == "sqlite":
            raise sqlite3.OperationalError(PRIVATE)
        if failure == "review":
            raise AnswerReviewError("review_storage_unavailable", PRIVATE, 503)
        raise EvidenceExportError(
            "annotation_storage_unavailable" if failure == "annotation" else PRIVATE, PRIVATE, 503
        )

    target = (
        api.container.notebook_reviews
        if failure == "review"
        else api.container.evidence_annotations
    )
    monkeypatch.setattr(target, "list_reviews" if failure == "review" else "list_annotations", fail)
    with caplog.at_level(logging.WARNING):
        response = api.client.get(NOTEBOOK, params={"run_id": bundle.run_id})
    assert response.status_code == (409 if failure == "unknown" else 503)
    assert_headers(response)
    assert PRIVATE not in response.text + caplog.text
    assert caplog.records


def test_control_bearing_return_filters_keep_links_without_lossy_form_values(
    api: ExplorerAPI,
) -> None:
    bundle = fixture_run(api)
    saved = annotate(api, bundle)
    options = {"run_id": bundle.run_id, "source": "source\0\r\n", "title": "title\r\n"}
    root = api.page(NOTEBOOK, **options)
    assert not root.by_tag("form")
    assert "controls that native inputs cannot preserve" in root.text()
    selected = api.page(href(root, f"annotation-source-{saved.sequence}"))
    for page in (root, selected):
        assert parse_qs(urlsplit(href(page, "back-to-catalog")).query) == {
            "limit": ["20"],
            "source": [options["source"]],
            "title": [options["title"]],
        }


def test_corrupt_review_numeric_json_is_an_explicit_logged_error(
    api: ExplorerAPI, caplog: pytest.LogCaptureFixture
) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    mutate(api, "UPDATE answer_reviews SET cited_chunk_ids = ?", ("[" + "9" * 5000 + "]",))
    response = api.client.get(NOTEBOOK, params={"run_id": bundle.run_id})
    assert response.status_code == 409
    assert_headers(response)
    assert "invalid_review_record" in caplog.text
    assert not ExplorerDOM(response.text).root.by_tag("article")


def test_selected_annotation_and_source_share_one_read_snapshot(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = fixture_run(api)
    saved = annotate(api, bundle)
    mutate(api, "PRAGMA journal_mode=WAL")
    original = api.container.evidence_annotations._sources

    def concurrent_edit(
        connection: sqlite3.Connection, encoding: str, run_id: str
    ) -> dict[str, EvidenceSource]:
        sources = original(connection, encoding, run_id)
        mutate(api, "UPDATE evidence_annotations SET quote = 'changed outside the service'")
        return sources

    monkeypatch.setattr(api.container.evidence_annotations, "_sources", concurrent_edit)
    selected = api.page(ANNOTATION, run_id=bundle.run_id, annotation_id=str(saved.annotation_id))
    assert selected.by_id("annotation-quote").text() == saved.quote
    assert values(selected, "passage-text") == [TEXT]
    monkeypatch.setattr(api.container.evidence_annotations, "_sources", original)
    response = api.client.get(
        ANNOTATION, params={"run_id": bundle.run_id, "annotation_id": str(saved.annotation_id)}
    )
    assert response.status_code == 409


def test_single_annotation_python_read_rejects_invalid_id_and_missing_storage(
    api: ExplorerAPI,
) -> None:
    with pytest.raises(AnnotationError) as invalid:
        api.container.evidence_annotations.get_annotation("", uuid4())
    assert invalid.value.code == "invalid_annotation_request"
    bundle = fixture_run(api)
    saved = annotate(api, bundle)
    api.path.unlink()
    response = api.client.get(
        ANNOTATION, params={"run_id": bundle.run_id, "annotation_id": str(saved.annotation_id)}
    )
    assert response.status_code == 503
    assert_headers(response)
    assert not api.path.exists()


def test_utf16_storage_decodes_exact_annotation_and_source(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    saved = annotate(api, bundle)
    with closing(sqlite3.connect(api.path)) as original:
        statements = "\n".join(original.iterdump())
    alternate = api.path.with_name("utf16.sqlite3")
    with closing(sqlite3.connect(alternate)) as connection:
        connection.execute("PRAGMA encoding='UTF-16'")
        connection.executescript(statements)
    app = create_app(offline_settings(alternate))
    with TestClient(app) as client:
        notebook = client.get(NOTEBOOK, params={"run_id": bundle.run_id})
        assert notebook.status_code == 200
        assert values(ExplorerDOM(notebook.text).root, "review-comment") == [COMMENT]
        selected = client.get(
            ANNOTATION, params={"run_id": bundle.run_id, "annotation_id": str(saved.annotation_id)}
        )
        assert selected.status_code == 200
        assert values(ExplorerDOM(selected.text).root, "passage-text") == [TEXT]


def test_notebook_openapi_describes_get_only_routes_and_existing_bounds(api: ExplorerAPI) -> None:
    paths = api.client.get("/openapi.json").json()["paths"]
    for path in (NOTEBOOK, ANNOTATION):
        assert set(paths[path]) == {"get"}
        endpoint = paths[path]["get"]
        assert "text/html" in endpoint["responses"]["200"]["content"]
        fields = {field["name"]: field for field in endpoint["parameters"]}
        for name in ("review_limit", "annotation_limit"):
            assert fields[name]["schema"]["minimum"] == 1
            assert fields[name]["schema"]["maximum"] == 100
            assert fields[name]["schema"]["default"] == 20
        assert {"404", "405", "409", "413", "422", "503"} <= set(endpoint["responses"])


def test_existing_review_apis_keep_first_terminal_semantics_while_notebook_is_bounded(
    api: ExplorerAPI,
) -> None:
    bundle = complete(api)
    for _ in range(MAX_RUN_EVENTS):
        api.container.event_log.append_event("synthetic-agent", bundle.run_id, "ignored-later", {})
    url = f"/runs/{bundle.run_id}/reviews"
    assert api.client.get(url).status_code == 200
    response = api.client.post(
        url,
        json={
            "review_id": str(uuid4()),
            "decision": "accepted",
            "comment": "Human opinion about the first completed capture.",
        },
    )
    assert response.status_code == 201
    assert api.client.get(url).json()["reviews"] == [response.json()]
    assert api.client.get(NOTEBOOK, params={"run_id": bundle.run_id}).status_code == 409


def test_overlong_json_integer_is_sanitized_by_the_review_service(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    opinion(api, bundle)
    mutate(api, "UPDATE answer_reviews SET cited_chunk_ids = ?", ("[" + "9" * 5000 + "]",))
    with pytest.raises(AnswerReviewError) as failure:
        api.container.answer_reviews.list_reviews(bundle.run_id)
    assert failure.value.code == "invalid_review_record"
    assert failure.value.status_code == 409
