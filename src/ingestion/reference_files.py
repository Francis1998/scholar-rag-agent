"""Dependency-free, defensive parsers for exported reference libraries.

Reads the files a systematic review starts from: RIS (PubMed, Embase, Scopus, Web of
Science, Zotero, Mendeley, EndNote), PubMed MEDLINE/NBIB, CSL-JSON (Zotero) and
ASReview-style CSV/TSV. Parsing is pure and offline: no network, models or storage.
Malformed or untitled records are reported with their record index and line number and
skipped; only a file that cannot be read at all fails as a whole.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from retrieval.duplicate_paper_cluster import _normalize_doi, _normalize_title

ReferenceFormat = Literal["ris", "medline", "csl_json", "csv", "tsv"]
FormatChoice = Literal["auto", "ris", "medline", "csl_json", "csv", "tsv"]
ImportedLabel = Literal["included", "excluded"]
REFERENCE_FORMATS: tuple[ReferenceFormat, ...] = ("ris", "medline", "csl_json", "csv", "tsv")

MAX_FILE_BYTES = 2_097_152
MAX_RECORDS_PER_FILE = 1_000
MAX_FILE_NAME_CHARACTERS = 255
MAX_TITLE_CHARACTERS = 512
MAX_ABSTRACT_CHARACTERS = 20_000
MAX_AUTHORS = 100
MAX_KEYWORDS = 50
MAX_NAME_CHARACTERS = 200
MAX_FIELD_CHARACTERS = 512
MAX_URL_CHARACTERS = 2_048
MAX_SHORT_CHARACTERS = 64

_LINES = re.compile(r"\r\n|\r|\n")
_SPACES = re.compile(r"[ \t\n\r\x0b\x0c]+")
_UNSAFE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f\ud800-\udfff]")
_YEAR = re.compile(r"(?<![0-9])[0-9]{4}(?![0-9])")
_DOI_SHAPE = re.compile(r"10\.[0-9]{4,9}/\S+")
_DOI_PREFIXES = (
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "dx.doi.org/",
    "doi.org/",
    "doi ",
)
_PMID_PREFIX = re.compile(r"^(?:pmid|pubmed(?:\s*id)?)\s*[:\-]?\s*", re.IGNORECASE)
_PMID_IN_TEXT = re.compile(r"\bPMID\s*[:\-]\s*([0-9]{1,9})\b", re.IGNORECASE)
_PUBMED_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/([0-9]{1,9})(?:[/?#]|$)", re.IGNORECASE)
_RIS_TAG = re.compile(r"^([A-Z][A-Z0-9])[ ]{1,2}-(?:[ ]+(.*?))?\s*$")
_MEDLINE_TAG = re.compile(r"^([A-Z][A-Z0-9]{0,3})[ ]*- (.*?)\s*$")


class ReferenceFileError(ValueError):
    """A whole file cannot be read; carries a stable public code and HTTP status."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class ReferenceRecord:
    """One importable reference with normalized identifiers and its original ones."""

    record_index: int
    line: int | None
    title: str
    abstract: str = ""
    authors: tuple[str, ...] = ()
    year: str = ""
    doi: str = ""
    pmid: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    keywords: tuple[str, ...] = ()
    url: str = ""
    publication_type: str = ""
    original_ids: tuple[tuple[str, str], ...] = ()
    label: ImportedLabel | None = None


@dataclass(frozen=True)
class SkippedRecord:
    """A reported, never imported record: 1-based index, first line and a stable reason."""

    record_index: int
    line: int | None
    reason: str
    message: str


@dataclass(frozen=True)
class ParsedReferenceFile:
    """Complete parse of one uploaded file; records and skips keep file order."""

    file_name: str
    format: ReferenceFormat
    format_detected: bool
    sha256: str
    byte_count: int
    records: tuple[ReferenceRecord, ...]
    skipped: tuple[SkippedRecord, ...]


@dataclass
class _Raw:
    """Format-neutral fields of one source record before validation."""

    record_index: int
    line: int | None
    fields: dict[str, list[str]] = field(default_factory=dict)
    ids: list[tuple[str, str]] = field(default_factory=list)
    label: ImportedLabel | None = None
    problem: tuple[str, str] | None = None

    def add(self, name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return
        text = str(value).strip()
        if text:
            self.fields.setdefault(name, []).append(text)

    def first(self, name: str) -> str:
        return next(iter(self.fields.get(name, ())), "")


def normalize_doi(value: str) -> str:
    """Return a lowercase bare DOI (``10.x/...``) or "" when the value is not DOI-shaped.

    Starts from the duplicate-candidate clusterer's normalization (lowercase; strips
    ``https://doi.org/``, ``http://doi.org/`` and ``doi:``) and also strips ``dx.doi.org``
    URLs, so imported, existing and connector DOIs compare equal. No DOI is resolved.
    """
    candidate = _normalize_doi(value)
    for prefix in _DOI_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :].strip().strip(".")
            break
    return candidate if _DOI_SHAPE.fullmatch(candidate) else ""


def normalize_pmid(value: str) -> str:
    """Return a PubMed ID as plain decimal digits without leading zeros, or ""."""
    candidate = _PMID_PREFIX.sub("", value.strip()).strip()
    if candidate.isascii() and candidate.isdecimal() and len(candidate) <= 9:
        number = int(candidate)
        return str(number) if number else ""
    return ""


def title_key(title: str) -> str:
    """Conservative title identity: casefolded words without accents or punctuation.

    Builds on the clusterer's title normalization. Titles shorter than four words return
    "" so that generic titles such as "Editorial" never match on title alone.
    """
    decomposed = unicodedata.normalize("NFKD", _normalize_title(title).casefold())
    words = re.findall(r"\w+", "".join(c for c in decomposed if not unicodedata.combining(c)))
    return " ".join(words) if len(words) >= 4 else ""


def _line(value: str) -> str:
    return _SPACES.sub(" ", value).strip()


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _skip(raw: _Raw, reason: str, message: str) -> SkippedRecord:
    return SkippedRecord(raw.record_index, raw.line, reason, message)


def _year(raw: _Raw) -> str:
    for value in raw.fields.get("year", ()):
        match = _YEAR.search(value)
        if match:
            return match.group(0)
    return ""


def _pages(raw: _Raw) -> str:
    pages = _line(raw.first("pages"))
    if pages:
        return pages
    start, end = _line(raw.first("start_page")), _line(raw.first("end_page"))
    return f"{start}-{end}" if start and end and start != end else start or end


def _url(raw: _Raw) -> str:
    for value in raw.fields.get("url", ()):
        candidate = _line(value)
        if candidate.lower().startswith(("https://", "http://")):
            return candidate
    return ""


def _limits(record: ReferenceRecord) -> tuple[str, str] | None:
    singles = {
        "title": (record.title, MAX_TITLE_CHARACTERS),
        "abstract": (record.abstract, MAX_ABSTRACT_CHARACTERS),
        "journal": (record.journal, MAX_FIELD_CHARACTERS),
        "publication_type": (record.publication_type, MAX_FIELD_CHARACTERS),
        "url": (record.url, MAX_URL_CHARACTERS),
        "doi": (record.doi, MAX_FIELD_CHARACTERS),
        "year": (record.year, MAX_SHORT_CHARACTERS),
        "volume": (record.volume, MAX_SHORT_CHARACTERS),
        "issue": (record.issue, MAX_SHORT_CHARACTERS),
        "pages": (record.pages, MAX_SHORT_CHARACTERS),
    }
    for name, (value, limit) in singles.items():
        if len(value) > limit:
            return "field_too_long", f"The {name} exceeds {limit} characters; nothing was clipped."
    for name, values, count in (
        ("authors", record.authors, MAX_AUTHORS),
        ("keywords", record.keywords, MAX_KEYWORDS),
    ):
        if len(values) > count:
            return "too_many_values", f"More than {count} {name}; nothing was clipped."
        if any(len(value) > MAX_NAME_CHARACTERS for value in values):
            return "field_too_long", f"One of the {name} exceeds {MAX_NAME_CHARACTERS} characters."
    if any(len(value) > MAX_FIELD_CHARACTERS for _, value in record.original_ids):
        return (
            "field_too_long",
            f"An original identifier exceeds {MAX_FIELD_CHARACTERS} characters.",
        )
    return None


def _build(raw: _Raw) -> ReferenceRecord | SkippedRecord:
    """Validate one raw record; every rejection is explicit and nothing is clipped."""
    if raw.problem is not None:
        return _skip(raw, *raw.problem)
    texts = [value for values in raw.fields.values() for value in values]
    texts += [value for pair in raw.ids for value in pair]
    if any(_UNSAFE.search(value) for value in texts):
        return _skip(
            raw,
            "control_characters",
            "The record contains control characters or invalid Unicode; it was not imported.",
        )
    title = _line(raw.first("title"))
    if not title:
        return _skip(raw, "missing_title", "The record has no title, so it cannot be screened.")
    dois = raw.fields.get("doi", [])
    pmids = raw.fields.get("pmid", [])
    record = ReferenceRecord(
        record_index=raw.record_index,
        line=raw.line,
        title=title,
        abstract="\n".join(_line(value) for value in raw.fields.get("abstract", ())),
        authors=_unique(_line(value) for value in raw.fields.get("authors", ())),
        year=_year(raw),
        doi=next((doi for doi in map(normalize_doi, dois) if doi), ""),
        pmid=next((pmid for pmid in map(normalize_pmid, pmids) if pmid), ""),
        journal=_line(raw.first("journal")),
        volume=_line(raw.first("volume")),
        issue=_line(raw.first("issue")),
        pages=_pages(raw),
        keywords=_unique(_line(value) for value in raw.fields.get("keywords", ())),
        url=_url(raw),
        publication_type="; ".join(_unique(_line(v) for v in raw.fields.get("type", ()))),
        original_ids=tuple(dict.fromkeys((name, _line(value)) for name, value in raw.ids)),
        label=raw.label,
    )
    problem = _limits(record)
    return record if problem is None else _skip(raw, *problem)


def _decode(content: bytes) -> str:
    """Decode UTF-8 (an Excel/Windows BOM is removed) or BOM-marked UTF-16."""
    try:
        if content.startswith((b"\xff\xfe", b"\xfe\xff")):
            return content.decode("utf-16")
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ReferenceFileError(
            "reference_file_not_utf8",
            f"The file is not valid UTF-8 (first invalid byte at offset {exc.start}). "
            "Export or re-save it as UTF-8; nothing was imported.",
        ) from exc


def detect_format(text: str, file_name: str) -> ReferenceFormat:
    """Detect a format from content first, then from the file extension."""
    stripped = text.lstrip()
    if stripped[:1] in ("[", "{"):
        return "csl_json"
    lines = [line for line in _LINES.split(text[:65_536]) if line.strip()][:50]
    if any(line.startswith("TY") and _RIS_TAG.match(line) for line in lines):
        return "ris"
    if any(line.startswith("PMID") and _MEDLINE_TAG.match(line) for line in lines):
        return "medline"
    name = file_name.lower()
    if name.endswith((".tsv", ".tab")) or (lines and "\t" in lines[0]):
        return "tsv"
    if name.endswith(".csv") or (lines and ("," in lines[0] or ";" in lines[0])):
        return "csv"
    raise ReferenceFileError(
        "reference_format_unknown",
        "The format could not be detected. Choose ris, medline, csl_json, csv or tsv "
        "explicitly; nothing was imported.",
    )


_Mapping = tuple[tuple[str, tuple[str, ...], bool], ...]
# (field, source tags in priority order, use only the first tag that has values)
_RIS_MAP: _Mapping = (
    ("title", ("TI", "T1", "CT", "BT"), True),
    ("abstract", ("AB", "N2"), True),
    ("authors", ("AU", "A1"), False),
    ("year", ("PY", "Y1", "DA"), False),
    ("doi", ("DO", "DI"), False),
    ("journal", ("JO", "JF", "T2", "JA", "J2"), False),
    ("volume", ("VL",), False),
    ("issue", ("IS",), False),
    ("start_page", ("SP",), False),
    ("end_page", ("EP",), False),
    ("keywords", ("KW",), False),
    ("url", ("UR",), False),
    ("type", ("TY",), True),
)
_MEDLINE_MAP: _Mapping = (
    ("pmid", ("PMID",), True),
    ("title", ("TI", "BTI"), True),
    ("abstract", ("AB",), True),
    ("authors", ("FAU", "AU", "CN"), True),
    ("year", ("DP", "DEP", "DA"), False),
    ("journal", ("JT", "TA"), False),
    ("volume", ("VI",), False),
    ("issue", ("IP",), False),
    ("pages", ("PG",), False),
    ("keywords", ("MH", "OT"), False),
    ("type", ("PT",), False),
)


def _map_tags(raw: _Raw, tags: dict[str, list[str]], mapping: _Mapping) -> None:
    for name, sources, first_only in mapping:
        for tag in sources:
            values = tags.get(tag, [])
            for value in values:
                raw.add(name, value)
            if values and first_only:
                break


def _ris_record(index: int, line: int, tags: dict[str, list[str]]) -> _Raw:
    raw = _Raw(index, line)
    _map_tags(raw, tags, _RIS_MAP)
    for tag in ("ID", "AN", "DO", "DI"):
        raw.ids.extend((tag, value) for value in tags.get(tag, []) if value)
    database = " ".join(tags.get("DB", []) + tags.get("DP", [])).lower()
    candidates = [
        match.group(1)
        for tag in ("AN", "N1", "M1", "M2")
        for value in tags.get(tag, [])
        for match in _PMID_IN_TEXT.finditer(value)
    ]
    candidates += [
        match.group(1)
        for value in tags.get("UR", [])
        for match in [_PUBMED_URL.search(value)]
        if match
    ]
    if any(name in database for name in ("pubmed", "medline", "nlm")):
        candidates += [value for value in tags.get("AN", []) if value.strip().isdecimal()]
    for candidate in candidates:
        raw.add("pmid", candidate)
    notes = " ".join(tags.get("N1", []))
    if "ASReview_relevant" in notes:
        raw.label = "included"
    elif "ASReview_irrelevant" in notes:
        raw.label = "excluded"
    return raw


def _ris(text: str) -> Iterator[_Raw]:
    """TY starts and ER ends a record; untagged lines continue the previous tag's value."""
    index = 0
    start: int | None = None
    tags: dict[str, list[str]] = {}
    last: str | None = None
    for number, line in enumerate(_LINES.split(text), 1):
        if not line.strip():
            continue
        match = _RIS_TAG.match(line)
        if match is None:
            if start is not None and last is not None:
                tags[last][-1] = (tags[last][-1] + " " + line.strip()).strip()
            continue
        tag, value = match.group(1), (match.group(2) or "").strip()
        if tag == "TY":
            if start is not None:
                index += 1
                yield _Raw(index, start, problem=("unterminated_record", _UNTERMINATED))
            start, tags, last = number, {"TY": [value]}, "TY"
        elif start is not None and tag == "ER":
            index += 1
            yield _ris_record(index, start, tags)
            start, tags, last = None, {}, None
        elif start is not None:
            tags.setdefault(tag, []).append(value)
            last = tag
    if start is not None:
        index += 1
        yield _Raw(index, start, problem=("unterminated_record", _UNTERMINATED))


_UNTERMINATED = "The RIS record has no ER line; it may be truncated and was not imported."


def _medline_record(index: int, line: int, tags: dict[str, list[str]]) -> _Raw:
    raw = _Raw(index, line)
    _map_tags(raw, tags, _MEDLINE_MAP)
    for tag in ("LID", "AID"):
        for value in tags.get(tag, []):
            raw.ids.append((tag, value))
            if value.lower().endswith("[doi]"):
                raw.add("doi", value[: -len("[doi]")])
    for tag in ("PMID", "PMC"):
        raw.ids.extend((tag, value) for value in tags.get(tag, []) if value)
    return raw


def _medline(text: str) -> Iterator[_Raw]:
    """Blank lines or a new PMID separate records; six-space lines continue a value."""
    index = 0
    start: int | None = None
    tags: dict[str, list[str]] = {}
    last: str | None = None
    problem: tuple[str, str] | None = None

    def finish() -> _Raw:
        nonlocal index
        index += 1
        raw = _medline_record(index, start or 0, tags)
        raw.problem = problem
        return raw

    for number, line in enumerate(_LINES.split(text), 1):
        if not line.strip():
            if start is not None:
                yield finish()
                start, tags, last, problem = None, {}, None, None
            continue
        if line[:1] in (" ", "\t"):
            if last is not None:
                tags[last][-1] = (tags[last][-1] + " " + line.strip()).strip()
            elif start is None:
                continue
            continue
        match = _MEDLINE_TAG.match(line)
        if match is None:
            if start is not None and problem is None:
                problem = ("malformed_line", f"Line {number} is not a MEDLINE tag or continuation.")
            continue
        tag, value = match.group(1), match.group(2).strip()
        if tag == "PMID" and start is not None and tags:
            yield finish()
            start, tags, last, problem = None, {}, None, None
        if start is None:
            start = number
        tags.setdefault(tag, []).append(value)
        last = tag
    if start is not None:
        yield finish()


def _csl_text(value: object) -> str:
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item.strip()), "")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return ""
    return str(value)


def _csl_people(value: object) -> Iterator[str]:
    for person in value if isinstance(value, list) else ():
        if isinstance(person, str):
            yield person
        elif isinstance(person, dict):
            literal = _csl_text(person.get("literal")).strip()
            family = _csl_text(person.get("family")).strip()
            given = _csl_text(person.get("given")).strip()
            yield literal or (f"{family}, {given}" if family and given else family or given)


def _csl_year(value: object) -> str:
    if isinstance(value, dict):
        parts = value.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            return _csl_text(parts[0][0])
        return _csl_text(value.get("raw")) or _csl_text(value.get("literal"))
    return _csl_text(value)


def _csl_record(index: int, item: object) -> _Raw:
    raw = _Raw(index, None)
    if not isinstance(item, dict):
        raw.problem = ("not_an_object", "The CSL-JSON entry is not an object.")
        return raw
    for name, key in (
        ("title", "title"),
        ("abstract", "abstract"),
        ("doi", "DOI"),
        ("pmid", "PMID"),
        ("journal", "container-title"),
        ("volume", "volume"),
        ("issue", "issue"),
        ("pages", "page"),
        ("url", "URL"),
        ("type", "type"),
    ):
        raw.add(name, _csl_text(item.get(key)))
    raw.add("year", _csl_year(item.get("issued")))
    for person in _csl_people(item.get("author")):
        raw.add("authors", person)
    keywords = _csl_text(item.get("keyword"))
    for keyword in keywords.split(";" if ";" in keywords else ","):
        raw.add("keywords", keyword)
    for key in ("id", "DOI", "PMID", "PMCID"):
        value = _csl_text(item.get(key)).strip()
        if value:
            raw.ids.append((key, value))
    return raw


def _csl_json(text: str) -> Iterator[_Raw]:
    """Accept an array, ``{"items": [...]}``, an object keyed by item ID, or one item."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReferenceFileError(
            "invalid_reference_file",
            f"CSL-JSON is not valid JSON (line {exc.lineno}, column {exc.colno}); "
            "nothing was imported.",
        ) from exc
    except RecursionError as exc:
        raise ReferenceFileError(
            "invalid_reference_file", "CSL-JSON nesting is too deep; nothing was imported."
        ) from exc
    items: list[object]
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
    elif isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
        items = list(data.values())
    elif isinstance(data, dict):
        items = [data]
    else:
        raise ReferenceFileError(
            "invalid_reference_file", "CSL-JSON must be an array or object of items."
        )
    for index, item in enumerate(items, 1):
        yield _csl_record(index, item)


_COLUMNS: dict[str, tuple[str, ...]] = {
    "title": ("title", "primary_title", "primary title", "article title", "document title", "ti"),
    "abstract": ("abstract", "abstract note", "abstract_note", "notes_abstract", "ab"),
    "authors": (
        "authors",
        "author",
        "author names",
        "first_authors",
        "first authors",
        "author full names",
        "af",
        "au",
    ),
    "year": ("year", "publication_year", "publication year", "py", "date"),
    "doi": ("doi", "di"),
    "pmid": ("pmid", "pubmed_id", "pubmed id", "pm"),
    "keywords": ("keywords", "keyword", "author keywords", "manual tags", "de"),
    "url": ("url", "link"),
    "journal": (
        "journal",
        "journal_name",
        "source title",
        "publication title",
        "secondary_title",
        "container-title",
        "so",
    ),
    "volume": ("volume", "vl"),
    "issue": ("issue", "is"),
    "pages": ("pages",),
    "start_page": ("bp",),
    "end_page": ("ep",),
    "type": ("type", "item type", "document type", "type_of_reference"),
}
# ASReview LAB's accepted label column names, in its documented order.
LABEL_COLUMNS = (
    "final_included",
    "label",
    "label_included",
    "included_label",
    "included_final",
    "included",
    "included_flag",
    "include",
)
_INCLUDED = frozenset({"1", "1.0", "true", "yes", "y", "included", "include", "relevant"})
_EXCLUDED = frozenset({"0", "0.0", "false", "no", "n", "excluded", "exclude", "irrelevant"})
_SPLIT = {"authors": ";", "keywords": ";"}


def _table(delimiter: str | None) -> Callable[[str], Iterator[_Raw]]:
    def parse(text: str) -> Iterator[_Raw]:
        first = next((line for line in _LINES.split(text) if line.strip()), "")
        chosen = delimiter or (";" if first.count(";") > first.count(",") else ",")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=chosen, strict=True)
        try:
            yield from _rows(reader)
        except csv.Error as exc:
            raise ReferenceFileError(
                "invalid_reference_file",
                f"The table is malformed near line {reader.line_num}: quoting or field size. "
                "Nothing was imported.",
            ) from exc

    return parse


def _rows(reader: Iterator[list[str]]) -> Iterator[_Raw]:
    header: list[str] | None = None
    previous = 0
    index = 0
    for row in reader:
        line_num = getattr(reader, "line_num", 0)
        start, previous = previous + 1, line_num
        if not any(cell.strip() for cell in row):
            continue
        if header is None:
            header = [cell.strip().lower() for cell in row]
            if len(set(header)) != len(header):
                raise ReferenceFileError(
                    "reference_columns_invalid", "The header repeats a column name."
                )
            if not any(alias in header for alias in _COLUMNS["title"]):
                raise ReferenceFileError(
                    "reference_columns_invalid",
                    "No title column was found (title or primary_title, case-insensitive).",
                )
            columns = {
                name: header.index(next(alias for alias in aliases if alias in header))
                for name, aliases in _COLUMNS.items()
                if any(alias in header for alias in aliases)
            }
            label = next((header.index(name) for name in LABEL_COLUMNS if name in header), None)
            continue
        index += 1
        raw = _Raw(index, start)
        if len(row) != len(header):
            raw.problem = (
                "column_count_mismatch",
                f"The row has {len(row)} fields; the header has {len(header)}.",
            )
            yield raw
            continue
        for name, column in columns.items():
            separator = _SPLIT.get(name)
            for value in row[column].split(separator) if separator else [row[column]]:
                raw.add(name, value)
        if label is not None:
            value = row[label].strip().lower()
            raw.label = (
                "included" if value in _INCLUDED else "excluded" if value in _EXCLUDED else None
            )
        yield raw
    if header is None:
        raise ReferenceFileError("reference_file_empty", "The table has no header row.")


_PARSERS: dict[ReferenceFormat, Callable[[str], Iterator[_Raw]]] = {
    "ris": _ris,
    "medline": _medline,
    "csl_json": _csl_json,
    "csv": _table(None),
    "tsv": _table("\t"),
}


def _file_name(value: str) -> str:
    if not 1 <= len(value) <= MAX_FILE_NAME_CHARACTERS or _UNSAFE.search(value):
        raise ReferenceFileError(
            "invalid_reference_file_name",
            f"File names need 1-{MAX_FILE_NAME_CHARACTERS} characters without control characters.",
        )
    return value


def parse_reference_file(
    content: bytes, *, file_name: str, format: FormatChoice = "auto"
) -> ParsedReferenceFile:
    """Parse one complete file within fixed bounds; never clips or guesses silently."""
    name = _file_name(file_name)
    if len(content) > MAX_FILE_BYTES:
        raise ReferenceFileError(
            "reference_file_too_large",
            f"Each file may have at most {MAX_FILE_BYTES} bytes; nothing was imported.",
            413,
        )
    text = _decode(content)
    chosen = detect_format(text, name) if format == "auto" else format
    records: list[ReferenceRecord] = []
    skipped: list[SkippedRecord] = []
    for raw in _PARSERS[chosen](text):
        if raw.record_index > MAX_RECORDS_PER_FILE:
            raise ReferenceFileError(
                "reference_file_too_many_records",
                f"Each file may have at most {MAX_RECORDS_PER_FILE} records; split the export.",
                413,
            )
        built = _build(raw)
        if isinstance(built, ReferenceRecord):
            records.append(built)
        else:
            skipped.append(built)
    if not records and not skipped:
        raise ReferenceFileError(
            "reference_file_empty",
            f"No {chosen} records were found. Check the export format; nothing was imported.",
        )
    return ParsedReferenceFile(
        file_name=name,
        format=chosen,
        format_detected=format == "auto",
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        records=tuple(records),
        skipped=tuple(skipped),
    )
