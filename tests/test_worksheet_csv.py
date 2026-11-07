"""Lossless, spreadsheet-safe worksheet rows across Python, API and native forms."""

import csv
import io
import json
from unittest.mock import AsyncMock

import pytest
from scripts.demo_browser_worksheet import worksheet_guards

from agent.research_worksheet import (
    ResearchWorksheet,
    WorksheetError,
    WorksheetRequest,
    WorksheetRow,
)
from retrieval.models import SearchResult
from tests.test_corpus_explorer import ExplorerDOM
from tests.test_research_worksheet import WorksheetAPI, denied
from tests.test_research_worksheet import worksheet_api as worksheet_api
from tests.test_worksheet_explorer import controls, failure

API = "/research/worksheet"
BROWSER = "/explore/worksheet"


def rows(content: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(content, newline="")))


async def test_csv_retains_every_field_and_question_paper_passage_order(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    api.container.runner._safety_limits.max_source_docs = 3
    original = api.add("b", "retrieval " + "\U0001f52c" * 801)
    original.title = "t" * 301
    original.source = "s" * 513
    sources = [
        SearchResult(
            chunk=original.model_copy(update={"chunk_id": f"chunk-{index}"}),
            score=0.0,
            retriever="synthetic",
            path=["exact", "path"],
        )
        for index in range(3)
    ]
    monkeypatch.setattr(
        api.container.runner._executor,
        "retrieve",
        AsyncMock(side_effect=[sources, [], sources, []]),
    )

    async def unchanged(query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results

    monkeypatch.setattr(api.container.runner._executor._reranker, "rerank", unchanged)
    collection = api.container.paper_collections.create(name="Selected", document_ids=["b", "a"])
    worksheet = await api.container.worksheets.build(
        WorksheetRequest(
            questions=["  retrieval  ", "retrieval"], collection_id=collection.collection_id
        )
    )
    content = worksheet.to_csv()
    parsed = rows(content)
    assert len(parsed) == 6
    assert [(r["question_number"], r["document_number"], r["passage_number"]) for r in parsed] == [
        ("1", "1", "1"),
        ("1", "1", "2"),
        ("1", "2", ""),
        ("2", "1", "1"),
        ("2", "1", "2"),
        ("2", "2", ""),
    ]
    for row in parsed:
        question = worksheet.rows[int(row["question_number"]) - 1]
        document = worksheet.documents[int(row["document_number"]) - 1]
        cell = question.cells[int(row["document_number"]) - 1]
        assert row["schema_version"] == "'1.0"
        assert row["csv_text_encoding"] == "'apostrophe-prefix-v1"
        assert row["kind"] == "'retrieval_passages"
        assert row["collection_id"] == "'" + collection.collection_id
        assert row["passages_per_cell"] == "2"
        assert json.loads(row["limits_json"][1:]) == worksheet.limits.model_dump(mode="json")
        assert json.loads(row["warnings_json"][1:]) == list(worksheet.warnings)
        assert row["question"] == "'" + question.question
        assert row["document_id"] == "'" + document.document_id
        assert row["inspection_url"] == "'" + str(document.inspection_url)
        assert row["status"] == "'" + cell.status
        assert row["preview_source_count"] == str(cell.preview_source_count)
        assert row["passages_omitted"] == str(cell.passages_omitted)
        assert json.loads(row["configuration_json"][1:]) == cell.configuration.model_dump(
            mode="json"
        )
        if cell.passages:
            passage = cell.passages[int(row["passage_number"]) - 1]
            assert row["passage_document_id"] == "'" + passage.document_id
            for name in ("chunk_id", "retriever", "title", "source", "excerpt", "text_sha256"):
                assert row[name] == "'" + getattr(passage, name)
            for name in ("rank", "score", "text_characters"):
                assert row[name] == str(getattr(passage, name))
            for name in ("title_truncated", "source_truncated", "excerpt_truncated"):
                assert row[name] == "true"
            assert json.loads(row["path_json"][1:]) == list(passage.path)
            assert row["passages_omitted"] == "1"
        else:
            assert row["status"] == "'no_passages"
            assert row["preview_source_count"] == row["passages_omitted"] == "0"
            assert all(value == "" for key, value in row.items() if key in PASSAGE_FIELDS)
    assert parsed[0]["score"] == "0.0"
    assert content == worksheet.to_csv()
    assert len(content.encode()) <= 262144


PASSAGE_FIELDS = {
    "passage_number",
    "passage_document_id",
    "chunk_id",
    "rank",
    "score",
    "retriever",
    "path_json",
    "title",
    "title_truncated",
    "source",
    "source_truncated",
    "excerpt",
    "excerpt_truncated",
    "text_characters",
    "text_sha256",
}


@pytest.mark.parametrize(
    "text",
    [
        "=1+1",
        "+1",
        "-1",
        "@SUM(1,1)",
        "\t=1+1",
        "\r\n=1+1",
        " \t+1",
        "'=literal",
        'Comma, quote " and \u7814\u7a76\r\nnext line',
        "\x00=1+1",
        "",
    ],
)
async def test_csv_strings_are_reversibly_prefixed_even_empty_control_or_formula_led(
    worksheet_api: WorksheetAPI, text: str
) -> None:
    worksheet = await worksheet_api.container.worksheets.build(
        WorksheetRequest(questions=["retrieval"], document_ids=["a"])
    )
    cell = worksheet.rows[0].cells[0]
    passage = cell.passages[0].model_copy(
        update={"title": text, "source": text, "excerpt": text, "path": (text,)}
    )
    changed = worksheet.model_copy(
        update={
            "rows": (
                worksheet.rows[0].model_copy(
                    update={"cells": (cell.model_copy(update={"passages": (passage,)}),)}
                ),
            )
        }
    )
    parsed = rows(changed.to_csv())[0]
    for name in ("title", "source", "excerpt"):
        assert parsed[name] == "'" + text
        assert parsed[name][1:] == text
    assert json.loads(parsed["path_json"][1:]) == [text]
    assert parsed["collection_id"] == ""
    assert json.loads(changed.to_json())["rows"][0]["cells"][0]["passages"][0]["excerpt"] == text


async def test_csv_exact_utf8_limit_counts_quoting_prefixes_and_crlf(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.research_worksheet as module

    worksheet = await worksheet_api.container.worksheets.build(
        WorksheetRequest(questions=['\u7814 "retrieval",\r\n=1'], document_ids=["a"])
    )
    content = worksheet.to_csv()
    assert content.startswith('"schema_version","csv_text_encoding",')
    assert content.endswith("\r\n") and not content.startswith("\ufeff")
    size = len(content.encode("utf-8"))
    assert size > len(content)
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", size)
    assert worksheet.to_csv() == content
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", size - 1)
    with pytest.raises(WorksheetError) as error:
        worksheet.to_csv()
    assert error.value.code == "worksheet_too_large" and error.value.status_code == 413


async def test_python_api_native_csv_parity_and_no_writes_or_generation(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    payload = WorksheetRequest(questions=["retrieval", "retrieval"], document_ids=["b", "a"])
    before = api.database.read_bytes()
    with worksheet_guards(api.container) as guards:
        worksheet = await api.container.worksheets.build(payload)
        expected = worksheet.to_csv()
        build = AsyncMock(wraps=api.container.worksheets.build)
        monkeypatch.setattr(api.container.worksheets, "build", build)
        opened = ExplorerDOM(api.client.get(BROWSER).text).root
        options = {node.attrs["value"]: node.text() for node in opened.by_tag("option")}
        assert options["csv"] == "CSV download"
        assert "apostrophe" in opened.text() and "rerun retrieval" in opened.text()
        native = api.client.post(
            opened.by_id("worksheet-form").attrs["action"],
            data={
                **controls(opened),
                "questions": "\n".join(payload.questions),
                "document_ids": json.dumps(payload.document_ids),
                "format": "csv",
            },
        )
        response = api.client.post(
            API, params={"format": "csv"}, json=payload.model_dump(mode="json", exclude_none=True)
        )
        for download in (response, native):
            assert download.status_code == 200, download.text
            assert download.text == expected
            assert download.headers["content-type"] == "text/csv; charset=utf-8"
            assert download.headers["content-disposition"] == 'attachment; filename="worksheet.csv"'
            assert download.headers["cache-control"] == "no-store"
            assert download.headers["x-content-type-options"] == "nosniff"
            assert int(download.headers["content-length"]) == len(download.content)
        assert native.headers["referrer-policy"] == "no-referrer"
        assert "default-src 'none'" in native.headers["content-security-policy"]
        assert build.await_count == 2  # Each export builds again, never a saved result.
        assert not any(guard.call_count for guard in guards.values())
    assert before == api.database.read_bytes()


def test_csv_openapi_and_invalid_format_before_build(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    operation = api.client.get("/openapi.json").json()["paths"][API]["post"]
    assert "text/csv" in operation["responses"]["200"]["content"]
    assert next(p for p in operation["parameters"] if p["name"] == "format")["schema"]["enum"] == [
        "json",
        "markdown",
        "csv",
    ]
    monkeypatch.setattr(api.container.worksheets, "build", denied)
    assert (
        api.client.post(
            API + "?format=CSV", json={"questions": ["retrieval"], "document_ids": ["a"]}
        ).status_code
        == 422
    )


def test_csv_limit_failure_is_sanitized_and_does_not_change_existing_exports(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    payload = {"questions": ["retrieval"], "document_ids": ["a"]}
    original = {
        fmt: api.client.post(API, params={"format": fmt}, json=payload).content
        for fmt in ("json", "markdown")
    }

    def too_large(self: ResearchWorksheet) -> str:
        raise WorksheetError("worksheet_too_large", "No partial worksheet was returned.", 413)

    monkeypatch.setattr(ResearchWorksheet, "to_csv", too_large)
    for fmt in ("json", "markdown"):
        assert api.client.post(API, params={"format": fmt}, json=payload).content == original[fmt]
    response = api.client.post(API, params={"format": "csv"}, json=payload)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "worksheet_too_large"
    assert "content-disposition" not in response.headers
    failure(
        api.client.post(
            BROWSER, data={"questions": "retrieval", "document_ids": '["a"]', "format": "csv"}
        ),
        413,
    )


def test_csv_all_fifty_no_passage_cells_in_order(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = worksheet_api
    identifiers = [f"paper-{i}" for i in range(10)]
    for identifier in identifiers:
        api.add(identifier)
    monkeypatch.setattr(api.container.runner._executor, "retrieve", AsyncMock(return_value=[]))
    response = api.client.post(
        API,
        params={"format": "csv"},
        json={"questions": ["same question"] * 5, "document_ids": identifiers},
    )
    assert response.status_code == 200, response.text
    parsed = rows(response.text)
    assert len(parsed) == 50
    assert [(r["question_number"], r["document_id"]) for r in parsed] == [
        (str(question), "'" + identifier) for question in range(1, 6) for identifier in identifiers
    ]
    assert all(r["status"] == "'no_passages" and r["chunk_id"] == "" for r in parsed)


async def test_csv_real_cap_never_trims_rows_or_breaks_json_markdown(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    worksheet = await worksheet_api.container.worksheets.build(
        WorksheetRequest(questions=["retrieval"], document_ids=["a"])
    )
    cell = worksheet.rows[0].cells[0]
    documents = tuple(
        worksheet.documents[0].model_copy(update={"document_id": f"paper-{i}"}) for i in range(10)
    )
    cells = tuple(
        cell.model_copy(
            update={
                "document_id": document.document_id,
                "preview_source_count": 3,
                "passages": tuple(
                    cell.passages[0].model_copy(
                        update={
                            "document_id": document.document_id,
                            "rank": rank,
                            "chunk_id": f"{document.document_id}-{rank}",
                        }
                    )
                    for rank in range(1, 4)
                ),
            }
        )
        for document in documents
    )
    large = worksheet.model_copy(
        update={
            "documents": documents,
            "passages_per_cell": 3,
            "rows": tuple(WorksheetRow(question="retrieval " * 50, cells=cells) for _ in range(5)),
        }
    )
    # Validate actual field bounds, not an oversized strings-only proxy.
    large = ResearchWorksheet.model_validate(large.model_dump())
    assert len(large.to_json().encode()) <= 262144
    assert len(large.to_markdown().encode()) <= 262144
    with pytest.raises(WorksheetError, match="No partial worksheet") as error:
        large.to_csv()
    assert error.value.status_code == 413
    monkeypatch.setattr(worksheet_api.container.worksheets, "build", AsyncMock(return_value=large))
    for fmt, status in (("json", 200), ("markdown", 200), ("csv", 413)):
        response = worksheet_api.client.post(
            API, params={"format": fmt}, json={"questions": ["retrieval"], "document_ids": ["a"]}
        )
        assert response.status_code == status
    failure(
        worksheet_api.client.post(
            BROWSER, data={"questions": "retrieval", "document_ids": '["a"]', "format": "csv"}
        ),
        413,
    )


@pytest.mark.parametrize("identifier", ["=1+1", "'already", "nul\0id", ".."])
async def test_csv_exact_question_and_identity_roundtrip(
    worksheet_api: WorksheetAPI, identifier: str
) -> None:
    api = worksheet_api
    api.add(identifier)
    question = "\t=1+1,\r\n\u7814"
    worksheet = await api.container.worksheets.build(
        WorksheetRequest(questions=[question], document_ids=[identifier])
    )
    row = rows(worksheet.to_csv())[0]
    assert row["question"][1:] == question
    assert row["document_id"][1:] == row["passage_document_id"][1:] == identifier
    assert row["chunk_id"][1:] == f"chunk-{identifier}"
    url = worksheet.documents[0].inspection_url
    assert row["inspection_url"] == ("" if url is None else "'" + url)


@pytest.mark.parametrize("format", ["csv", "json", "markdown"])
def test_export_failure_after_first_cell_is_not_partial(
    worksheet_api: WorksheetAPI, monkeypatch: pytest.MonkeyPatch, format: str
) -> None:
    api = worksheet_api
    original = api.container.runner.preview
    calls = 0

    async def fail_second(query: str, *, document_ids: tuple[str, ...]) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("PRIVATE_DIAGNOSTIC")
        return await original(query, document_ids=document_ids)

    monkeypatch.setattr(api.container.runner, "preview", fail_second)
    response = api.client.post(
        API,
        params={"format": format},
        json={"questions": ["retrieval"], "document_ids": ["a", "b"]},
    )
    assert response.status_code == 500
    assert "PRIVATE_DIAGNOSTIC" not in response.text and "chunk-a" not in response.text
    assert "content-disposition" not in response.headers
