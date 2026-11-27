"""Reference-library parsers: every supported export, malformed input and fixed bounds."""

import json
from pathlib import Path

import pytest

from ingestion import reference_files
from ingestion.reference_files import (
    ReferenceFileError,
    detect_format,
    normalize_doi,
    normalize_pmid,
    parse_reference_file,
    title_key,
)

FIXTURES = Path(__file__).parent / "fixtures" / "references"
TITLE = "Caffeine timing and sleep onset in adolescents: a randomised crossover trial"
DOI = "10.5555/sleep.2021.001"

# file, format, record (index, line) list, skipped (index, line, reason) list
CASES = [
    ("search.ris", "ris", [(1, 4), (3, 30)],
     [(2, 26, "missing_title"), (4, 39, "unterminated_record")]),
    ("pubmed.nbib", "medline", [(1, 1)],
     [(2, 23, "malformed_line"), (3, 29, "missing_title")]),
    ("library.json", "csl_json", [(1, None), (2, None)],
     [(3, None, "not_an_object"), (4, None, "missing_title")]),
    ("asreview.csv", "csv", [(1, 2), (2, 4)],
     [(3, 5, "missing_title"), (4, 6, "column_count_mismatch")]),
    ("wos.tsv", "tsv", [(1, 2)], []),
]  # fmt: skip


def parse(
    name: str, content: bytes | None = None, **options: str
) -> reference_files.ParsedReferenceFile:
    data = (FIXTURES / name).read_bytes() if content is None else content
    return parse_reference_file(data, file_name=name, **options)  # type: ignore[arg-type]


@pytest.mark.parametrize(("name", "kind", "records", "skipped"), CASES)
def test_each_export_format_parses_records_and_reports_skips_with_lines(
    name: str,
    kind: str,
    records: list[tuple[int, int | None]],
    skipped: list[tuple[int, int | None, str]],
) -> None:
    parsed = parse(name)
    assert parsed.format == kind and parsed.format_detected
    assert [(record.record_index, record.line) for record in parsed.records] == records
    assert [(item.record_index, item.line, item.reason) for item in parsed.skipped] == skipped
    first = parsed.records[0]
    assert first.title.rstrip(".") == TITLE
    assert (first.doi, first.year) == (DOI, "2021")
    assert len(first.authors) == 2 and first.authors[0].startswith("Garcia, Mar")
    assert parsed.byte_count == len((FIXTURES / name).read_bytes())


@pytest.mark.parametrize(("name", "kind"), [(case[0], case[1]) for case in CASES])
def test_bom_and_crlf_exports_parse_identically(name: str, kind: str) -> None:
    original = (FIXTURES / name).read_bytes()
    windows = b"\xef\xbb\xbf" + original.replace(b"\n", b"\r\n")
    baseline, variant = parse(name), parse(name, windows)
    assert variant.format == kind
    assert variant.records == baseline.records and variant.skipped == baseline.skipped
    assert variant.sha256 != baseline.sha256


def test_excel_unicode_text_utf16_is_decoded() -> None:
    content = (FIXTURES / "wos.tsv").read_text(encoding="utf-8").encode("utf-16")
    assert parse("wos.tsv", content).records == parse("wos.tsv").records


def test_ris_maps_repeated_tags_continuations_unicode_and_identifiers() -> None:
    first, chapter = parse("search.ris").records
    assert first.title == TITLE
    assert first.authors == ("Garcia, María José", "Østergaard, Søren")
    assert (
        first.abstract
        == "Background: evening caffeine delays sleep.\nMethods: 40 adolescents, crossover."
    )
    assert first.keywords == ("caffeine", "sleep onset")
    assert (first.journal, first.volume, first.issue, first.pages) == (
        "Journal of Synthetic Sleep Research",
        "12",
        "3",
        "101-109",
    )
    assert first.pmid == "33445566" and first.publication_type == "JOUR"
    assert ("DO", "https://doi.org/10.5555/SLEEP.2021.001") in first.original_ids
    assert ("AN", "33445566") in first.original_ids
    assert chapter.title == "睡眠と記憶の統合に関する総説" and chapter.authors == ("山田, 太郎",)
    assert chapter.url == "https://example.org/chapter/7" and chapter.year == "2019"
    assert (first.label, chapter.label) == ("included", "excluded")


def test_ris_new_type_tag_before_end_reports_the_open_record() -> None:
    content = b"TY  - JOUR\nTI  - First title here\nTY  - JOUR\nTI  - Second title here\nER  -\n"
    parsed = parse("x.ris", content)
    assert [(item.record_index, item.line, item.reason) for item in parsed.skipped] == [
        (1, 1, "unterminated_record")
    ]
    assert [(record.record_index, record.title) for record in parsed.records] == [
        (2, "Second title here")
    ]


def test_medline_tags_without_a_value_do_not_break_the_record() -> None:
    parsed = parse("x.nbib", b"PMID- 7\nTI  - A title before an empty tag\nAB  -\nDP  - 2001\n")
    assert not parsed.skipped and parsed.records[0].year == "2001"


def test_medline_uses_full_authors_doi_lid_terms_and_types() -> None:
    record = parse("pubmed.nbib").records[0]
    assert record.title.endswith("crossover trial.")
    assert (
        record.abstract
        == "Background: evening caffeine delays sleep. Methods: 40 adolescents, crossover."
    )
    assert record.authors == ("Garcia, Maria Jose", "Ostergaard, Soren")
    assert record.keywords == ("Caffeine/*administration & dosage", "sleep onset")
    assert record.publication_type == "Journal Article; Randomized Controlled Trial"
    assert record.journal == "Journal of Synthetic Sleep Research"
    assert ("AID", "S1234-5678(21)00001-2 [pii]") in record.original_ids


@pytest.mark.parametrize(
    ("data", "titles"),
    [
        ({"items": [{"title": "Items wrapper title"}]}, ["Items wrapper title"]),
        ({"ITEM-1": {"title": "Keyed object title"}}, ["Keyed object title"]),
        ({"title": "Single item title", "type": "book"}, ["Single item title"]),
    ],
)
def test_csl_json_accepts_array_items_keyed_and_single_objects(
    data: object, titles: list[str]
) -> None:
    parsed = parse("x.json", json.dumps(data).encode())
    assert [record.title for record in parsed.records] == titles


def test_csl_json_people_dates_numbers_and_keywords() -> None:
    first, report = parse("library.json").records
    assert first.volume == "12" and first.pages == "101-109"
    assert ("id", "http://zotero.org/users/1/items/ABCD1234") in first.original_ids
    assert report.authors == ("Synthetic Sleep Consortium",) and report.year == "2018"
    assert report.keywords == ("sleep hygiene", "shift work") and report.url.endswith("/2")


@pytest.mark.parametrize(
    ("header", "row"),
    [
        (
            "primary_title;Abstract Note;Author Names;publication_year\n",
            "Sleep and mood in nurses;Text;Roe, R;2015\n",
        ),
        ("TITLE,ABSTRACT,AUTHORS,PY\n", 'Sleep and mood in nurses,Text,"Roe, R",2015\n'),
    ],
)
def test_csv_headers_are_case_insensitive_aliases_with_detected_delimiter(
    header: str, row: str
) -> None:
    record = parse("x.csv", (header + row).encode()).records[0]
    assert (record.title, record.abstract, record.authors, record.year) == (
        "Sleep and mood in nurses",
        "Text",
        ("Roe, R",),
        "2015",
    )


def test_csv_reports_label_columns_and_multiline_cells() -> None:
    first, second = parse("asreview.csv").records
    assert "\n" not in first.title and first.abstract.count("\n") == 1
    assert (first.label, second.label) == ("included", "excluded")


@pytest.mark.parametrize(
    ("name", "content", "chosen", "code"),
    [
        ("x.csv", b"x,y\n1,2\n", "auto", "reference_columns_invalid"),
        ("x.csv", b"title,Title\na,b\n", "auto", "reference_columns_invalid"),
        ("x.csv", b'title,abstract\n"unclosed,quote\n', "auto", "invalid_reference_file"),
        ("x.json", b"[1, 2", "auto", "invalid_reference_file"),
        ("x.json", b'[{"volume": ' + b"9" * 5000 + b"}]", "auto", "invalid_reference_file"),
        ("x.tsv", b"\xff\xfeA\x00\x00\xd8", "auto", "reference_file_not_utf8"),
        ("x.csv", b"caf\xe9", "auto", "reference_file_not_utf8"),
        ("x.ris", b"no tagged lines here\n", "ris", "reference_file_empty"),
        ("x.txt", b"just some prose", "auto", "reference_format_unknown"),
        ("x.csv", b"title\n", "auto", "reference_file_empty"),
    ],
)
def test_unreadable_files_fail_as_a_whole_without_echoing_content(
    name: str, content: bytes, chosen: str, code: str
) -> None:
    with pytest.raises(ReferenceFileError) as raised:
        parse(name, content, format=chosen)
    assert raised.value.code == code and raised.value.status_code == 422
    assert "unclosed" not in str(raised.value) and "prose" not in str(raised.value)


def test_files_and_record_counts_above_bounds_are_413(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ReferenceFileError) as raised:
        parse("big.ris", b"x" * (reference_files.MAX_FILE_BYTES + 1))
    assert (raised.value.code, raised.value.status_code) == ("reference_file_too_large", 413)
    monkeypatch.setattr(reference_files, "MAX_RECORDS_PER_FILE", 2)
    content = b"".join(b"TY  - JOUR\nTI  - Title number %d here\nER  -\n" % n for n in range(3))
    with pytest.raises(ReferenceFileError) as raised:
        parse("many.ris", content)
    assert (raised.value.code, raised.value.status_code) == ("reference_file_too_many_records", 413)


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        ([b"TI  - " + b"t" * 513], "field_too_long"),
        ([b"TI  - Fine title here", b"AB  - " + b"a" * 20_001], "field_too_long"),
        ([b"TI  - Fine title"] + [b"AU  - Author %d" % n for n in range(101)], "too_many_values"),
        ([b"TI  - Fine title here", b"AU  - " + b"n" * 201], "field_too_long"),
        ([b"TI  - Bell \x07 in the title"], "control_characters"),
        ([b"TI  - Null \x00 byte"], "control_characters"),
    ],
)  # fmt: skip
def test_oversized_or_unsafe_records_are_skipped_never_clipped(
    lines: list[bytes], reason: str
) -> None:
    content = b"\n".join(
        [b"TY  - JOUR", *lines, b"ER  -", b"TY  - JOUR", b"TI  - Kept title", b"ER  -"]
    )
    parsed = parse("x.ris", content)
    assert [(item.record_index, item.reason) for item in parsed.skipped] == [(1, reason)]
    assert [record.title for record in parsed.records] == ["Kept title"]


def test_csl_json_lone_surrogates_are_skipped() -> None:
    parsed = parse("x.json", b'[{"title": "Broken \\ud800 title"}, {"title": "Fine title"}]')
    assert [item.reason for item in parsed.skipped] == ["control_characters"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.1000/ABC", "10.1000/abc"),
        ("https://doi.org/10.1000/ABC", "10.1000/abc"),
        ("http://dx.doi.org/10.1000/abc.", "10.1000/abc"),
        ("DOI: 10.1000/abc", "10.1000/abc"),
        ("doi 10.1000/abc", "10.1000/abc"),
        ("  HTTPS://DOI.ORG/10.1000/abc  ", "10.1000/abc"),
        ("10.1/too-short-prefix", ""),
        ("not a doi", ""),
        ("", ""),
    ],
)
def test_doi_normalization(value: str, expected: str) -> None:
    assert normalize_doi(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("33445566", "33445566"),
        ("PMID: 00123", "123"),
        ("pubmed 42", "42"),
        ("0", ""),
        ("12a", ""),
        ("1234567890", ""),
    ],
)
def test_pmid_normalization(value: str, expected: str) -> None:
    assert normalize_pmid(value) == expected


def test_title_key_ignores_case_accents_punctuation_but_not_short_titles() -> None:
    assert title_key("Café  intake, and SLEEP: a trial.") == title_key(
        "cafe intake and sleep a trial"
    )
    assert title_key("Editorial") == title_key("Erratum: x") == ""


def test_format_detection_prefers_content_then_extension() -> None:
    assert detect_format("\n\nTY  - JOUR\n", "export.txt") == "ris"
    assert detect_format("PMID- 1\nTI  - x\n", "pubmed.txt") == "medline"
    assert detect_format(" [ ]", "x.csv") == "csl_json"
    assert detect_format("title\tabstract\n", "x.txt") == "tsv"
    assert detect_format("title\n", "x.csv") == "csv"
    explicit = parse("x.txt", b"title,abstract\nA title,\n", format="csv")
    assert not explicit.format_detected and explicit.records[0].title == "A title"


@pytest.mark.parametrize("name", ["", "a" * 256, "bad\nname.ris"])
def test_file_names_must_be_bounded_labels(name: str) -> None:
    with pytest.raises(ReferenceFileError) as raised:
        parse_reference_file(b"TY  - JOUR\nTI  - x\nER  -\n", file_name=name)
    assert raised.value.code == "invalid_reference_file_name"
