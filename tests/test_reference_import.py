"""Reference import: deduplication, idempotency, no-overwrite, provenance, API and HTML."""

import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

from api import reference_import as import_api
from ingestion.reference_import import (
    ImportTarget,
    ReferenceImportPreview,
    SQLiteReferenceImports,
    UploadedFile,
)
from retrieval.models import Document
from storage.paper_collections import CollectionEdit, CollectionError, PaperCollection
from tests.test_corpus_explorer import HOSTILE, ExplorerAPI, ExplorerDOM, assert_headers, values
from tests.test_corpus_explorer import api as api

FIXTURES = Path(__file__).parent / "fixtures" / "references"
ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}


def fixture(name: str, file_name: str | None = None) -> UploadedFile:
    return UploadedFile(file_name or name, (FIXTURES / name).read_bytes())


def ris(*records: dict[str, str]) -> bytes:
    lines = []
    for record in records:
        lines.append("TY  - JOUR")
        lines.extend(f"{tag}  - {value}" for tag, value in record.items())
        lines.append("ER  - ")
    return "\n".join(lines).encode()


def service(api: ExplorerAPI) -> SQLiteReferenceImports:
    return api.container.reference_imports


def new(name: str) -> ImportTarget:
    return ImportTarget(new_collection_name=name)


def into(collection: PaperCollection) -> ImportTarget:
    return ImportTarget(
        collection_id=collection.collection_id, collection_revision=collection.revision
    )


def seed(api: ExplorerAPI, document_id: str, title: str, **metadata: str) -> None:
    api.container.document_store.add_documents(
        [Document(document_id=document_id, title=title, text="", source="s", metadata=metadata)],
        [],
    )


def rows(api: ExplorerAPI, sql: str, *params: str) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(api.path)) as connection:
        return connection.execute(sql, params).fetchall()


def statuses(preview: ReferenceImportPreview) -> list[tuple[int, int, str, str | None]]:
    return [
        (item.file_index, item.record_index, item.status, item.match_reason)
        for item in preview.candidates
    ]


def test_collection_duplicates_match_doi_forms_pmid_then_title_with_year(api: ExplorerAPI) -> None:
    seed(api, "m-doi", "A member found by DOI", doi="https://doi.org/10.5555/ABC")
    seed(api, "m-pmid", "A member found by PMID", pmid="0777")
    seed(api, "m-title", "Effects of light on circadian phase in humans", year="2019")
    collection = api.container.paper_collections.create(
        name="Existing", document_ids=["m-doi", "m-pmid", "m-title"]
    )
    content = ris(
        {"TI": "Completely different first title", "DO": "HTTP://DX.DOI.ORG/10.5555/abc."},
        {"TI": "Completely different second title", "AN": "777", "DB": "PubMed"},
        {"TI": "Effects of Light on Circadian Phase in Humans.", "PY": "2019"},
        {"TI": "Effects of light on circadian phase in humans", "PY": "2020"},
        {"TI": "Effects of light on circadian phase in humans"},
    )
    preview = service(api).preview([UploadedFile("s.ris", content)], into(collection))
    assert statuses(preview) == [
        (1, 1, "duplicate_in_collection", "doi"),
        (1, 2, "duplicate_in_collection", "pmid"),
        (1, 3, "duplicate_in_collection", "title_year"),
        (1, 4, "new", None),
        (1, 5, "new", None),
    ]
    assert [item.matched_document_id for item in preview.candidates[:3]] == [
        "m-doi",
        "m-pmid",
        "m-title",
    ]
    assert preview.can_commit and preview.current_members == 3


def test_duplicates_within_and_across_files_are_reported_not_added(api: ExplorerAPI) -> None:
    files = [fixture("search.ris"), fixture("library.json"), fixture("pubmed.nbib")]
    preview = service(api).preview(files, new("Sleep review"))
    assert statuses(preview) == [
        (1, 1, "new", None),
        (1, 3, "new", None),
        (2, 1, "duplicate_in_import", "doi"),
        (2, 2, "new", None),
        (3, 1, "duplicate_in_import", "doi"),
    ]
    assert preview.candidates[2].matched_record == (1, 1)
    totals = preview.totals
    assert (totals.records_read, totals.records_skipped, totals.duplicates_in_import) == (11, 6, 2)
    assert (totals.papers_to_add, totals.labels.included, totals.labels.excluded) == (3, 1, 1)
    assert [file.papers_to_add for file in preview.files] == [2, 1, 0]
    assert preview.files[0].skipped_reasons == {"missing_title": 1, "unterminated_record": 1}
    assert rows(api, "SELECT COUNT(*) FROM paper_collections") == [(0,)]
    assert rows(api, "SELECT COUNT(*) FROM documents") == [(0,)]


def test_reimporting_the_same_file_reports_duplicates_and_never_doubles(api: ExplorerAPI) -> None:
    first = service(api).preview([fixture("search.ris")], new("Sleep"))
    result = service(api).commit(first.draft_id)
    assert len(result.collection.document_ids) == 2 and result.collection.revision == 1
    again = service(api).preview([fixture("search.ris")], into(result.collection))
    assert again.blocking_reason == "nothing_new" and not again.can_commit
    assert again.files[0].previously_imported == (result.record.import_id,)
    assert statuses(again) == [
        (1, 1, "duplicate_in_collection", "doi"),
        (1, 3, "duplicate_in_collection", "document_id"),
    ]
    for draft, code in (
        (again.draft_id, "reference_import_nothing_new"),
        (first.draft_id, "reference_import_already_committed"),
    ):
        with pytest.raises(CollectionError) as raised:
            service(api).commit(draft)
        assert (raised.value.code, raised.value.status_code) == (code, 409)
    assert api.container.paper_collections.get(result.collection.collection_id) == result.collection
    assert rows(api, "SELECT COUNT(*) FROM reference_imports") == [(1,)]


def test_existing_documents_and_members_are_never_overwritten_or_reordered(
    api: ExplorerAPI,
) -> None:
    first = service(api).commit(service(api).preview([fixture("search.ris")], new("A")).draft_id)
    stored = rows(api, "SELECT document_id, title, text, source, metadata FROM documents")
    seed(api, "kept-1", "A member that stays first")
    seed(api, "kept-2", "A member that stays second")
    target = api.container.paper_collections.create(name="B", document_ids=["kept-2", "kept-1"])
    ingested: list[list[str]] = []
    original = service(api)._ingest
    service(api)._ingest = lambda documents: (
        ingested.append([document.document_id for document in documents]),
        original(documents),
    )
    preview = service(api).preview([fixture("search.ris", "renamed.ris")], into(target))
    assert [item.existing_document for item in preview.candidates] == [True, True]
    assert preview.totals.existing_documents_reused == 2
    result = service(api).commit(preview.draft_id)
    assert ingested == []
    assert result.collection.document_ids == ("kept-2", "kept-1", *first.collection.document_ids)
    assert result.collection.revision == 2
    assert [paper.created for paper in result.record.papers] == [False, False]
    current = rows(api, "SELECT document_id, title, text, source, metadata FROM documents")
    assert [row for row in current if not str(row[0]).startswith("kept-")] == stored


def test_capacity_and_stale_revisions_block_commit_without_writes(api: ExplorerAPI) -> None:
    for index in range(99):
        seed(api, f"member-{index:02d}", f"Existing member number {index}")
    full = api.container.paper_collections.create(
        name="Nearly full", document_ids=[f"member-{index:02d}" for index in range(99)]
    )
    preview = service(api).preview([fixture("search.ris")], into(full))
    assert preview.blocking_reason == "capacity_exceeded" and not preview.can_commit
    with pytest.raises(CollectionError) as raised:
        service(api).commit(preview.draft_id)
    assert (raised.value.code, raised.value.status_code) == (
        "reference_import_capacity_exceeded",
        422,
    )
    small = api.container.paper_collections.create(name="Small", document_ids=["member-00"])
    stale = service(api).preview([fixture("search.ris")], into(small))
    api.container.paper_collections.edit(
        small.collection_id, changes=CollectionEdit(name="Renamed", expected_revision=1)
    )
    with pytest.raises(CollectionError) as raised:
        service(api).commit(stale.draft_id)
    assert (raised.value.code, raised.value.status_code) == ("collection_revision_conflict", 409)
    assert rows(api, "SELECT COUNT(*) FROM reference_imports") == [(0,)]
    assert rows(api, "SELECT COUNT(*) FROM reference_import_staging WHERE import_id IS NULL") == [
        (2,)
    ]


def test_imported_labels_never_become_screening_decisions(api: ExplorerAPI) -> None:
    preview = service(api).preview([fixture("asreview.csv")], new("Labelled"))
    assert preview.totals.labels.model_dump() == {"included": 1, "excluded": 1, "unlabelled": 0}
    result = service(api).commit(preview.draft_id)
    queue = api.container.paper_screening.list_queue(
        result.collection.collection_id, collection_revision=result.collection.revision, limit=100
    )
    assert queue.counts.unscreened == 2 and queue.included_document_ids == ()
    assert {item.status for item in queue.items} == {"unscreened"}
    for (metadata,) in rows(api, "SELECT metadata FROM documents"):
        assert "label" not in str(metadata) and "included" not in str(metadata)


def test_each_paper_and_the_collection_record_provenance(api: ExplorerAPI) -> None:
    files = [fixture("search.ris"), fixture("library.json")]
    result = service(api).commit(service(api).preview(files, new("Provenance")).draft_id)
    record = result.record
    assert [(paper.file_index, paper.record_index) for paper in record.papers] == [
        (1, 1),
        (1, 3),
        (2, 2),
    ]
    assert [file.file_name for file in record.files] == ["search.ris", "library.json"]
    assert [file.skipped for file in record.files] == [(), ()]
    assert record.files[1].skipped_reasons == {"missing_title": 1, "not_an_object": 1}
    assert record.totals.duplicates_in_import == 1
    ((metadata,),) = rows(
        api, "SELECT metadata FROM documents WHERE document_id = ?", record.papers[0].document_id
    )
    stored = json.loads(str(metadata))
    assert {key: stored[key] for key in ("import_file", "import_format", "import_line")} == {
        "import_file": "search.ris",
        "import_format": "ris",
        "import_line": "4",
    }
    assert stored["import_record_index"] == "1" and stored["doi"] == "10.5555/sleep.2021.001"
    assert stored["import_file_sha256"] == record.files[0].sha256
    assert ["DO", "https://doi.org/10.5555/SLEEP.2021.001"] in json.loads(
        stored["import_original_ids"]
    )
    report = api.container.collection_duplicates.inspect(
        result.collection.collection_id, collection_revision=1
    )
    assert report.inspected_count == 3
    history = service(api).history(result.collection.collection_id)
    assert [item.import_id for item in history.imports] == [record.import_id]


def test_expired_unknown_and_malformed_drafts_are_rejected(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = service(api).preview([fixture("wos.tsv")], new("Later"))
    clock = service(api)._clock
    monkeypatch.setattr(service(api), "_clock", lambda: clock() + timedelta(hours=25))
    for draft, code, status in (
        (preview.draft_id, "reference_import_draft_not_found", 404),
        ("imd_" + "0" * 32, "reference_import_draft_not_found", 404),
        ("../draft", "invalid_reference_import", 422),
    ):
        with pytest.raises(CollectionError) as raised:
            service(api).commit(draft)
        assert (raised.value.code, raised.value.status_code) == (code, status)
    assert rows(api, "SELECT COUNT(*) FROM paper_collections") == [(0,)]


def drafts(api: ExplorerAPI) -> int:
    return int(rows(api, "SELECT COUNT(*) FROM reference_import_staging")[0][0])


def test_multipart_preview_commit_and_history_api(api: ExplorerAPI) -> None:
    uploads = [
        ("files", ("search.ris", (FIXTURES / "search.ris").read_bytes())),
        ("files", ("library.json", (FIXTURES / "library.json").read_bytes())),
    ]
    preview = api.client.post(
        "/collections/imports/preview", files=uploads, data={"new_collection_name": "Sleep"}
    )
    assert preview.status_code == 200, preview.text
    assert preview.headers["cache-control"] == "no-store"
    assert preview.headers["x-content-type-options"] == "nosniff"
    body = preview.json()
    assert body["totals"]["papers_to_add"] == 3 and body["can_commit"]
    committed = api.client.post("/collections/imports", json={"draft_id": body["draft_id"]})
    assert committed.status_code == 201, committed.text
    collection = committed.json()["collection"]
    assert committed.headers["location"] == f"/collections/{collection['collection_id']}"
    assert api.client.get(committed.headers["location"]).json() == collection
    history = api.client.get(f"/collections/{collection['collection_id']}/imports").json()
    assert [item["import_id"] for item in history["imports"]] == [
        committed.json()["record"]["import_id"]
    ]
    assert history["imports"][0]["files"][0]["papers_to_add"] == 2


def test_json_preview_accepts_text_content_for_an_existing_collection(api: ExplorerAPI) -> None:
    seed(api, "member", "Light exposure before bed and melatonin suppression", year="2017")
    collection = api.container.paper_collections.create(name="Light", document_ids=["member"])
    response = api.client.post(
        "/collections/imports/preview",
        json={
            "files": [
                {
                    "file_name": "asreview.csv",
                    "content": (FIXTURES / "asreview.csv").read_text(encoding="utf-8"),
                    "format": "csv",
                }
            ],
            "collection_id": collection.collection_id,
            "collection_revision": 1,
        },
    )
    assert response.status_code == 200, response.text
    candidates = response.json()["candidates"]
    assert [(item["status"], item["match_reason"]) for item in candidates] == [
        ("new", None),
        ("duplicate_in_collection", "title_year"),
    ]
    assert response.json()["files"][0]["format_detected"] is False


def upload(name: str, content: bytes, field: str = "files", **data: str) -> dict[str, object]:
    return {"files": [(field, (name, content))], "data": {"new_collection_name": "x", **data}}


def as_json(**file: str) -> dict[str, object]:
    return {
        "json": {
            "files": [{"file_name": "a.ris", "content": "x", **file}],
            "new_collection_name": "x",
        }
    }


INVALID, MEDIA = "invalid_reference_import", "reference_import_media_type"
INVALID_PREVIEWS = [
    ({"content": b"x", "headers": {"content-type": "text/plain"}}, 415, MEDIA),
    ({"json": {"files": [], "new_collection_name": "x"}}, 422, INVALID),
    ({"json": {"files": [{"file_name": "a.ris", "content": "x"}]}}, 422, INVALID),
    (as_json(format="xlsx"), 422, INVALID),
    (upload("a.ris", b"x", format="pdf"), 422, INVALID),
    (upload("a.ris", b"x", field="other"), 422, INVALID),
    (upload("big.ris", b"x" * 2_097_153), 413, "reference_file_too_large"),
    (upload("a.txt", b"prose" + HOSTILE.encode()), 422, "reference_format_unknown"),
]  # fmt: skip


@pytest.mark.parametrize(("kwargs", "status", "code"), INVALID_PREVIEWS)
def test_invalid_previews_use_the_error_envelope_and_store_nothing(
    api: ExplorerAPI, kwargs: dict[str, object], status: int, code: str
) -> None:
    response = api.client.post("/collections/imports/preview", **kwargs)  # type: ignore[arg-type]
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"
    assert "script" not in response.text and drafts(api) == 0


def test_request_body_bound_is_enforced_before_parsing(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(import_api, "MAX_MULTIPART_BYTES", 1024)
    response = api.client.post(
        "/collections/imports/preview",
        files=[("files", ("a.ris", b"TY  - JOUR\n" * 200))],
        data={"new_collection_name": "x"},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "reference_import_too_large"


def test_commit_and_history_errors(api: ExplorerAPI) -> None:
    unknown = api.client.post("/collections/imports", json={"draft_id": "imd_" + "a" * 32})
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["code"] == "reference_import_draft_not_found"
    assert api.client.post("/collections/imports", json={"draft_id": "nope"}).status_code == 422
    missing = api.client.get("/collections/col_" + "0" * 32 + "/imports")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "collection_not_found"


def test_browser_form_is_explicit_multipart_and_linked_from_collections(api: ExplorerAPI) -> None:
    opened = api.client.get("/explore/collections/import")
    assert opened.status_code == 200
    assert_headers(opened, referrer_policy="same-origin")
    root = ExplorerDOM(opened.text).root
    form = root.by_id("reference-import")
    assert (form.attrs["method"], form.attrs["enctype"]) == ("post", "multipart/form-data")
    assert root.by_id("reference-files").attrs["name"] == "files"
    assert "multiple" in root.by_id("reference-files").attrs
    assert root.by_id("new-collection-name").attrs["name"] == "new_collection_name"
    assert "EXPLICIT HUMAN WRITE / REFERENCE IMPORT" in opened.text
    listing = ExplorerDOM(api.client.get("/explore/collections").text).root
    assert listing.by_id("import-references").attrs["href"] == "/explore/collections/import"
    seed(api, "member", "A saved member")
    collection = api.container.paper_collections.create(name="Saved", document_ids=["member"])
    detail = ExplorerDOM(
        api.client.get(
            "/explore/collection",
            params={"collection_id": collection.collection_id, "collection_revision": 1},
        ).text
    ).root
    target = detail.by_id("import-references").attrs["href"]
    existing = ExplorerDOM(api.client.get(target).text).root
    hidden = {
        node.attrs["name"]: node.attrs["value"]
        for node in existing.by_tag("input")
        if node.attrs.get("type") == "hidden"
    }
    assert hidden == {"collection_id": collection.collection_id, "collection_revision": "1"}
    stale = api.client.get(
        "/explore/collections/import",
        params={"collection_id": collection.collection_id, "collection_revision": 2},
    )
    assert stale.status_code == 409 and not ExplorerDOM(stale.text).root.by_tag("form")


def test_browser_preview_escapes_file_contents_names_and_commits_explicitly(
    api: ExplorerAPI,
) -> None:
    content = ris({"TI": HOSTILE + " sleep and memory", "PY": "2020", "DO": "10.5555/x1"})
    preview = api.client.post(
        "/explore/collections/import",
        files=[("files", (HOSTILE + ".ris", content))],
        data={"new_collection_name": "Review " + HOSTILE[:40]},
        headers=ORIGIN,
    )
    assert preview.status_code == 200, preview.text
    assert_headers(preview, referrer_policy="same-origin")
    root = ExplorerDOM(preview.text).root
    assert not root.by_tag("script") and not root.by_tag("img") and not root.by_tag("base")
    assert values(root, "candidate-title") == [HOSTILE + " sleep and memory"]
    # Clients percent-encode quotes in multipart file names; directories are dropped.
    assert values(root, "file-name") == ["%22> & [link](javascript:alert(1)).ris"]
    assert root.by_id("papers-to-add").text() == "1"
    form = root.by_id("reference-import-commit")
    assert form.attrs["action"] == "/explore/collections/import/commit"
    assert rows(api, "SELECT COUNT(*) FROM documents") == [(0,)]
    draft = form.by_tag("input")[0].attrs["value"]
    committed = api.client.post(
        "/explore/collections/import/commit",
        data={"draft_id": draft},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert committed.status_code == 303
    page = api.client.get(committed.headers["location"])
    assert page.status_code == 200 and "sleep and memory" in page.text
    assert "<script>" not in page.text


@pytest.mark.parametrize(
    ("path", "headers", "status"),
    [
        ("/explore/collections/import", {}, 403),
        ("/explore/collections/import", {"origin": "http://evil.example"}, 403),
        ("/explore/collections/import/commit", {}, 403),
    ],
)
def test_browser_writes_require_same_origin_and_store_nothing(
    api: ExplorerAPI, path: str, headers: dict[str, str], status: int
) -> None:
    response = api.client.post(
        path,
        files=[("files", ("a.ris", (FIXTURES / "search.ris").read_bytes()))],
        data={"new_collection_name": "x"},
        headers=headers,
    )
    assert response.status_code == status
    assert drafts(api) == 0 and rows(api, "SELECT COUNT(*) FROM documents") == [(0,)]


def test_browser_method_and_encoding_errors_are_html(api: ExplorerAPI) -> None:
    response = api.client.get("/explore/collections/import/commit")
    assert response.status_code == 405 and response.headers["allow"] == "POST"
    assert_headers(response, referrer_policy="same-origin")
    encoded = api.client.post(
        "/explore/collections/import/commit", json={"draft_id": "x"}, headers=ORIGIN
    )
    assert encoded.status_code == 415
    upload = api.client.post(
        "/explore/collections/import", data={"new_collection_name": "x"}, headers=ORIGIN
    )
    assert upload.status_code == 415 and drafts(api) == 0
