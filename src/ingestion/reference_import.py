"""Preview and commit reference-library imports into paper collections, fully offline.

A preview parses every uploaded file, reports skipped records, duplicates within the import
and against the target collection, and the papers that would be added; it stores the exact
uploaded bytes as a private draft but never changes documents, collections or screening.
Committing a draft adds only new papers through the existing atomic collection store and
records per-file counts and per-paper provenance. It never deletes or overwrites papers.
Imported include/exclude labels are counted and ignored: they never become decisions.
"""

from __future__ import annotations

import base64
import binascii
import json
import sqlite3
import threading
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from ingestion.chunking import stable_id
from ingestion.pubmed import PUBMED_ARTICLE_URL
from ingestion.reference_files import (
    FormatChoice,
    ParsedReferenceFile,
    ReferenceFileError,
    ReferenceFormat,
    ReferenceRecord,
    SkippedRecord,
    normalize_doi,
    normalize_pmid,
    parse_reference_file,
    publication_year,
    title_key,
)
from retrieval.models import Document
from retrieval.scope import MAX_DOCUMENT_IDS
from storage.collection_duplicates import DOI_FIELDS, MAX_METADATA_BYTES
from storage.document_catalog import MAX_SOURCE_CHARACTERS
from storage.document_store import DocumentsExistError
from storage.paper_collections import (
    CollectionEdit,
    CollectionError,
    CollectionId,
    CollectionName,
    PaperCollection,
    Revision,
    SQLitePaperCollections,
)

MAX_FILES = 5
DRAFT_LIFETIME = timedelta(hours=24)
MAX_HISTORY = 100
SOURCE_TYPE = "reference_import"
CandidateStatus = Literal["new", "duplicate_in_import", "duplicate_in_collection"]
MatchReason = Literal["doi", "pmid", "document_id", "title_year"]
DraftId = Annotated[
    str, StringConstraints(strict=True, min_length=36, max_length=36, pattern=r"^imd_[0-9a-f]{32}$")
]
ImportId = Annotated[
    str, StringConstraints(strict=True, min_length=36, max_length=36, pattern=r"^imp_[0-9a-f]{32}$")
]
NOTES = (
    "Preview stores the uploaded bytes as a private local draft for up to 24 hours so that "
    "commit imports exactly what you previewed. Preview never changes documents, "
    "collections or screening.",
    "Duplicates are matched by exact normalized DOI, then exact PMID, then identical "
    "imported content, then the same casefolded title without accents or punctuation (at "
    "least four words) and the same year. Fuzzy near-duplicates stay in: inspect duplicate "
    "candidates after importing.",
    "Imported include/exclude labels (ASReview label columns or N1 notes) are only counted. "
    "They are never stored and never become screening decisions; every imported paper "
    "starts unscreened.",
    "Commit never deletes, overwrites or reorders papers or members. Adding papers to an "
    "existing collection is a membership edit: the revision increments and earlier "
    "screening decisions on that collection become stale, as with any saved edit.",
    "Counts are per uploaded file and per import, not a PRISMA flow diagram. Records, "
    "titles, abstracts and file names may be sensitive; review before sharing.",
)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ImportTarget(_Model):
    """Exactly one of an existing collection at a known revision or a new collection name."""

    collection_id: CollectionId | None = None
    collection_revision: Revision | None = None
    new_collection_name: CollectionName | None = None

    @model_validator(mode="after")
    def _one_target(self) -> Self:
        existing = self.collection_id is not None and self.collection_revision is not None
        if existing == (self.new_collection_name is not None) or (
            (self.collection_id is None) != (self.collection_revision is None)
        ):
            raise ValueError(
                "Choose collection_id with collection_revision, or new_collection_name."
            )
        return self


class SkippedReference(_Model):
    record_index: int
    line: int | None
    reason: str
    message: str


class LabelCounts(_Model):
    """Imported labels, reported and then ignored."""

    included: int = 0
    excluded: int = 0
    unlabelled: int = 0


class ImportFileSummary(_Model):
    file_index: int
    file_name: str
    format: ReferenceFormat
    format_detected: bool
    sha256: str
    byte_count: int
    records_read: int
    records_skipped: int
    duplicates_in_import: int
    duplicates_in_collection: int
    papers_to_add: int
    labels: LabelCounts
    previously_imported: tuple[str, ...] = ()
    skipped_reasons: dict[str, int] = Field(default_factory=dict)
    skipped: tuple[SkippedReference, ...] = ()


class ImportCandidate(_Model):
    """One parsed record and what committing would do with it."""

    file_index: int
    record_index: int
    line: int | None
    title: str
    year: str
    doi: str
    pmid: str
    authors: tuple[str, ...]
    document_id: str
    status: CandidateStatus
    match_reason: MatchReason | None = None
    matched_document_id: str | None = None
    matched_record: tuple[int, int] | None = None
    existing_document: bool = False


class ImportTotals(_Model):
    files: int
    records_read: int
    records_skipped: int
    duplicates_in_import: int
    duplicates_in_collection: int
    papers_to_add: int
    existing_documents_reused: int
    labels: LabelCounts


class ReferenceImportPreview(_Model):
    """Complete dry run of one draft against one collection revision."""

    schema_version: Literal["1.0"] = "1.0"
    draft_id: DraftId
    expires_at: datetime
    target: ImportTarget
    collection_name: str
    current_members: int
    capacity: int = MAX_DOCUMENT_IDS
    can_commit: bool
    blocking_reason: Literal["nothing_new", "capacity_exceeded"] | None
    totals: ImportTotals
    files: tuple[ImportFileSummary, ...]
    candidates: tuple[ImportCandidate, ...]
    notes: tuple[str, ...] = NOTES


class ImportedPaper(_Model):
    document_id: str
    file_index: int
    record_index: int
    created: bool


class ReferenceImportRecord(_Model):
    """Saved provenance: per-file counts and where every added paper came from."""

    schema_version: Literal["1.0"] = "1.0"
    import_id: ImportId
    draft_id: DraftId
    collection_id: CollectionId
    collection_revision: Revision
    created_at: datetime
    totals: ImportTotals
    files: tuple[ImportFileSummary, ...]
    papers: tuple[ImportedPaper, ...] = Field(min_length=1, max_length=MAX_DOCUMENT_IDS)


class ReferenceImportResult(_Model):
    record: ReferenceImportRecord
    collection: PaperCollection


class ReferenceImportHistory(_Model):
    collection_id: CollectionId
    imports: tuple[ReferenceImportRecord, ...]


@dataclass(frozen=True)
class UploadedFile:
    """Exact uploaded bytes and the caller's chosen format."""

    file_name: str
    content: bytes
    format: FormatChoice = "auto"

    def to_json(self) -> dict[str, str]:
        return {
            "file_name": self.file_name,
            "format": self.format,
            "content": base64.b64encode(self.content).decode("ascii"),
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        if not isinstance(value, dict) or set(value) != {"file_name", "format", "content"}:
            raise ValueError("Invalid draft file.")
        try:
            content = base64.b64decode(value["content"], validate=True)
        except (binascii.Error, TypeError) as exc:
            raise ValueError("Invalid draft content.") from exc
        return cls(file_name=value["file_name"], content=content, format=value["format"])


def _bibliographic(record: ReferenceRecord) -> dict[str, str]:
    values = {
        "source_type": SOURCE_TYPE,
        "doi": record.doi,
        "pmid": record.pmid,
        "year": record.year,
        "authors": "; ".join(record.authors),
        "journal": record.journal,
        "volume": record.volume,
        "issue": record.issue,
        "pages": record.pages,
        "keywords": "; ".join(record.keywords),
        "url": record.url,
        "publication_type": record.publication_type,
    }
    return {key: value for key, value in values.items() if value}


def _source(record: ReferenceRecord) -> str:
    if record.doi:
        source = f"https://doi.org/{record.doi}"
    elif record.pmid:
        source = f"{PUBMED_ARTICLE_URL}/{record.pmid}/"
    else:
        source = record.url or SOURCE_TYPE
    return source if len(source) <= MAX_SOURCE_CHARACTERS else SOURCE_TYPE


def reference_document(record: ReferenceRecord, parsed: ParsedReferenceFile) -> Document:
    """Map a record to a corpus document; the ID depends on content, never on provenance."""
    bibliographic = _bibliographic(record)
    source = _source(record)
    identity = json.dumps(
        ["reference-import-v1", record.title, record.abstract, source, bibliographic],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    metadata = {
        **bibliographic,
        "import_file": parsed.file_name,
        "import_format": parsed.format,
        "import_record_index": str(record.record_index),
        "import_file_sha256": parsed.sha256,
        "import_original_ids": json.dumps(
            [list(pair) for pair in record.original_ids], ensure_ascii=False
        ),
    }
    if record.line is not None:
        metadata["import_line"] = str(record.line)
    return Document(
        document_id=stable_id(identity, "doc-ref-v1"),
        title=record.title,
        text=record.abstract,
        source=source,
        metadata=metadata,
    )


@dataclass(frozen=True)
class _Member:
    """Duplicate signals of one current collection member."""

    document_id: str
    doi: str
    pmid: str
    title_year: tuple[str, str] | None


@dataclass(frozen=True)
class _Prepared:
    file_index: int
    record: ReferenceRecord
    document: Document


@dataclass(frozen=True)
class _Plan:
    files: tuple[ImportFileSummary, ...]
    candidates: tuple[ImportCandidate, ...]
    documents: dict[str, Document]
    totals: ImportTotals


def _keys(member: _Member) -> tuple[tuple[MatchReason, object], ...]:
    return (
        ("doi", member.doi or None),
        ("pmid", member.pmid or None),
        ("document_id", member.document_id),
        ("title_year", member.title_year),
    )


def _signals(item: _Prepared) -> _Member:
    record = item.record
    key = title_key(record.title)
    return _Member(
        item.document.document_id,
        record.doi,
        record.pmid,
        (key, record.year) if key and record.year else None,
    )


def _prepare(
    parsed_files: Sequence[ParsedReferenceFile],
) -> tuple[list[_Prepared], dict[int, list[SkippedRecord]]]:
    prepared: list[_Prepared] = []
    skipped: dict[int, list[SkippedRecord]] = {}
    for file_index, parsed in enumerate(parsed_files, 1):
        skipped[file_index] = list(parsed.skipped)
        for record in parsed.records:
            document = reference_document(record, parsed)
            size = len(json.dumps(document.metadata, sort_keys=True).encode("utf-8"))
            if size > MAX_METADATA_BYTES:
                skipped[file_index].append(
                    SkippedRecord(
                        record.record_index,
                        record.line,
                        "metadata_too_large",
                        f"Stored metadata would exceed {MAX_METADATA_BYTES} bytes.",
                    )
                )
            else:
                prepared.append(_Prepared(file_index, record, document))
        skipped[file_index].sort(key=lambda item: item.record_index)
    return prepared, skipped


def _plan(
    parsed_files: Sequence[ParsedReferenceFile],
    prepared: Sequence[_Prepared],
    skipped: dict[int, list[SkippedRecord]],
    members: Sequence[_Member],
    existing_documents: set[str],
    history: Sequence[ReferenceImportRecord],
) -> _Plan:
    """Exact DOI, PMID, content ID, then title+year; collection matches win over import."""
    in_collection: dict[MatchReason, dict[object, str]] = {}
    for member in members:
        for reason, key in _keys(member):
            if key is not None:
                in_collection.setdefault(reason, {}).setdefault(key, member.document_id)
    in_import: dict[MatchReason, dict[object, tuple[int, int]]] = {}
    candidates: list[ImportCandidate] = []
    documents: dict[str, Document] = {}
    for item in prepared:
        keys = [(reason, key) for reason, key in _keys(_signals(item)) if key is not None]
        position = (item.file_index, item.record.record_index)
        status: CandidateStatus = "new"
        reason_found: MatchReason | None = None
        matched_document: str | None = None
        matched_record: tuple[int, int] | None = None
        for reason, key in keys:
            if key in in_collection.get(reason, {}):
                status, reason_found = "duplicate_in_collection", reason
                matched_document = in_collection[reason][key]
                break
        else:
            for reason, key in keys:
                if key in in_import.get(reason, {}):
                    status, reason_found = "duplicate_in_import", reason
                    matched_record = in_import[reason][key]
                    break
        if status == "new":
            for reason, key in keys:
                in_import.setdefault(reason, {}).setdefault(key, position)
            documents[item.document.document_id] = item.document
        candidates.append(
            ImportCandidate(
                file_index=item.file_index,
                record_index=item.record.record_index,
                line=item.record.line,
                title=item.record.title,
                year=item.record.year,
                doi=item.record.doi,
                pmid=item.record.pmid,
                authors=item.record.authors,
                document_id=item.document.document_id,
                status=status,
                match_reason=reason_found,
                matched_document_id=matched_document,
                matched_record=matched_record,
                existing_document=status == "new"
                and item.document.document_id in existing_documents,
            )
        )
    return _summaries(parsed_files, candidates, skipped, documents, history)


def _labels(records: Sequence[ReferenceRecord]) -> LabelCounts:
    included = sum(record.label == "included" for record in records)
    excluded = sum(record.label == "excluded" for record in records)
    return LabelCounts(
        included=included, excluded=excluded, unlabelled=len(records) - included - excluded
    )


def _summaries(
    parsed_files: Sequence[ParsedReferenceFile],
    candidates: Sequence[ImportCandidate],
    skipped: dict[int, list[SkippedRecord]],
    documents: dict[str, Document],
    history: Sequence[ReferenceImportRecord],
) -> _Plan:
    files: list[ImportFileSummary] = []
    for file_index, parsed in enumerate(parsed_files, 1):
        statuses = [item.status for item in candidates if item.file_index == file_index]
        reasons: dict[str, int] = {}
        for item in skipped[file_index]:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
        files.append(
            ImportFileSummary(
                file_index=file_index,
                file_name=parsed.file_name,
                format=parsed.format,
                format_detected=parsed.format_detected,
                sha256=parsed.sha256,
                byte_count=parsed.byte_count,
                records_read=len(parsed.records) + len(parsed.skipped),
                records_skipped=len(skipped[file_index]),
                duplicates_in_import=statuses.count("duplicate_in_import"),
                duplicates_in_collection=statuses.count("duplicate_in_collection"),
                papers_to_add=statuses.count("new"),
                labels=_labels(parsed.records),
                previously_imported=tuple(
                    record.import_id
                    for record in history
                    if any(file.sha256 == parsed.sha256 for file in record.files)
                ),
                skipped_reasons=dict(sorted(reasons.items())),
                skipped=tuple(
                    SkippedReference(
                        record_index=item.record_index,
                        line=item.line,
                        reason=item.reason,
                        message=item.message,
                    )
                    for item in skipped[file_index]
                ),
            )
        )
    totals = ImportTotals(
        files=len(files),
        records_read=sum(file.records_read for file in files),
        records_skipped=sum(file.records_skipped for file in files),
        duplicates_in_import=sum(file.duplicates_in_import for file in files),
        duplicates_in_collection=sum(file.duplicates_in_collection for file in files),
        papers_to_add=sum(file.papers_to_add for file in files),
        existing_documents_reused=sum(item.existing_document for item in candidates),
        labels=LabelCounts(
            included=sum(file.labels.included for file in files),
            excluded=sum(file.labels.excluded for file in files),
            unlabelled=sum(file.labels.unlabelled for file in files),
        ),
    )
    return _Plan(tuple(files), tuple(candidates), documents, totals)


def _metadata(value: object) -> dict[str, str]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else {}
    except (ValueError, RecursionError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {key: item for key, item in decoded.items() if isinstance(item, str)}


def _members(connection: sqlite3.Connection, document_ids: Sequence[str]) -> list[_Member]:
    """Read only titles and DOI/PMID/year metadata; unreadable metadata is no signal."""
    rows = {
        row["document_id"]: row
        for row in connection.execute(
            "SELECT document_id, "
            "CASE WHEN typeof(title) = 'text' THEN substr(title, 1, 4096) ELSE '' END AS title, "
            "CASE WHEN typeof(metadata) = 'text' THEN substr(metadata, 1, 65536) ELSE '' END "
            "AS metadata FROM documents WHERE document_id IN (SELECT value FROM json_each(?))",
            (json.dumps(list(document_ids)),),
        )
    }
    members: list[_Member] = []
    for document_id in document_ids:
        row = rows.get(document_id)
        if row is None:
            continue
        metadata = _metadata(row["metadata"])
        doi = next((metadata[name] for name in DOI_FIELDS if metadata.get(name, "").strip()), "")
        year = publication_year(
            next(
                (metadata[name] for name in _YEAR_FIELDS if metadata.get(name, "").strip()),
                "",
            )
        )
        key = title_key(row["title"])
        members.append(
            _Member(
                document_id,
                normalize_doi(doi),
                normalize_pmid(metadata.get("pmid", "")),
                (key, year) if key and year else None,
            )
        )
    return members


_YEAR_FIELDS = ("year", "published_year", "publication_year")
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS reference_import_staging (
        draft_id TEXT PRIMARY KEY NOT NULL,
        created_at TEXT NOT NULL,
        request TEXT NOT NULL,
        import_id TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reference_imports (
        import_id TEXT PRIMARY KEY NOT NULL,
        collection_id TEXT NOT NULL,
        collection_revision INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        record TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_reference_imports_collection "
    "ON reference_imports(collection_id, created_at)",
)


def _storage_error() -> CollectionError:
    return CollectionError(
        "reference_import_storage_error",
        "Import, collection or corpus storage is unavailable; nothing is reported as saved.",
        503,
    )


def parse_uploads(files: Sequence[UploadedFile]) -> list[ParsedReferenceFile]:
    """Parse every file or fail the whole request; messages never echo file contents."""
    if not 1 <= len(files) <= MAX_FILES:
        raise CollectionError(
            "invalid_reference_import", f"Upload 1-{MAX_FILES} reference files.", 422
        )
    parsed: list[ParsedReferenceFile] = []
    for index, upload in enumerate(files, 1):
        try:
            parsed.append(
                parse_reference_file(
                    upload.content, file_name=upload.file_name, format=upload.format
                )
            )
        except ReferenceFileError as exc:
            raise CollectionError(exc.code, f"File {index}: {exc}", exc.status_code) from exc
    return parsed


_DRAFT_ID = TypeAdapter(DraftId)


class SQLiteReferenceImports:
    """Drafts, atomic commits and import records in the corpus database."""

    def __init__(
        self,
        database_path: Path | str,
        collections: SQLitePaperCollections,
        ingest: Callable[[list[Document]], object],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=rw"
        self._collections = collections
        self._ingest = ingest
        self._clock = clock or (lambda: datetime.now(UTC))
        self._commit_lock = threading.Lock()
        try:
            with closing(sqlite3.connect(database_path)) as connection, connection:
                for statement in _SCHEMA:
                    connection.execute(statement)
        except sqlite3.Error as exc:
            raise _storage_error() from exc

    def _connect(self) -> closing[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_uri, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        return closing(connection)

    def _snapshot(
        self, target: ImportTarget, document_ids: Sequence[str]
    ) -> tuple[str, list[_Member], set[str], list[ReferenceImportRecord]]:
        """Resolve the target, its members, history and existing documents in one read."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            members: list[_Member] = []
            history: list[ReferenceImportRecord] = []
            if target.collection_id is not None:
                collection = SQLitePaperCollections.resolve_in_snapshot(
                    connection, target.collection_id
                )
                if collection.revision != target.collection_revision:
                    raise CollectionError(
                        "collection_revision_conflict",
                        "The collection changed; open its current revision and preview again.",
                        409,
                    )
                name = collection.name
                members = _members(connection, collection.document_ids)
                history = self._read_history(connection, collection.collection_id)
            else:
                name = target.new_collection_name or ""
                if connection.execute(
                    "SELECT 1 FROM paper_collections WHERE name = ?", (name,)
                ).fetchone():
                    raise CollectionError(
                        "collection_name_conflict",
                        "A collection with that exact, trimmed name already exists.",
                        409,
                    )
            existing = {
                row[0]
                for row in connection.execute(
                    "SELECT document_id FROM documents "
                    "WHERE document_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(list(document_ids)),),
                )
            }
        return name, members, existing, history

    @staticmethod
    def _read_history(
        connection: sqlite3.Connection, collection_id: str
    ) -> list[ReferenceImportRecord]:
        rows = connection.execute(
            "SELECT record FROM reference_imports WHERE collection_id = ? "
            "ORDER BY created_at DESC, import_id DESC LIMIT ?",
            (collection_id, MAX_HISTORY),
        ).fetchall()
        try:
            return [ReferenceImportRecord.model_validate_json(row[0]) for row in rows]
        except (ValidationError, TypeError) as exc:
            raise CollectionError(
                "invalid_reference_import_record",
                "A saved import record is invalid; nothing was repaired.",
                409,
            ) from exc

    def _planned(
        self, files: Sequence[UploadedFile], target: ImportTarget
    ) -> tuple[str, int, _Plan]:
        parsed = parse_uploads(files)
        prepared, skipped = _prepare(parsed)
        name, members, existing, history = self._snapshot(
            target, [item.document.document_id for item in prepared]
        )
        return name, len(members), _plan(parsed, prepared, skipped, members, existing, history)

    def preview(
        self, files: Sequence[UploadedFile], target: ImportTarget
    ) -> ReferenceImportPreview:
        """Dry run: report everything commit would do; only a private draft is stored."""
        try:
            name, current, plan = self._planned(files, target)
            created = self._clock()
            draft_id = f"imd_{uuid4().hex}"
            request = {
                "target": target.model_dump(mode="json"),
                "files": [upload.to_json() for upload in files],
                "additions": [item.document_id for item in plan.candidates if item.status == "new"],
            }
            with self._connect() as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM reference_import_staging WHERE created_at < ?",
                    ((created - DRAFT_LIFETIME).isoformat(),),
                )
                connection.execute(
                    "INSERT INTO reference_import_staging (draft_id, created_at, request) "
                    "VALUES (?, ?, ?)",
                    (draft_id, created.isoformat(), json.dumps(request)),
                )
        except sqlite3.Error as exc:
            raise _storage_error() from exc
        adding = plan.totals.papers_to_add
        blocking: Literal["nothing_new", "capacity_exceeded"] | None = (
            "nothing_new"
            if not adding
            else "capacity_exceeded"
            if current + adding > MAX_DOCUMENT_IDS
            else None
        )
        return ReferenceImportPreview(
            draft_id=draft_id,
            expires_at=created + DRAFT_LIFETIME,
            target=target,
            collection_name=name,
            current_members=current,
            can_commit=blocking is None,
            blocking_reason=blocking,
            totals=plan.totals,
            files=plan.files,
            candidates=plan.candidates,
        )

    def _draft(self, draft_id: str) -> tuple[ImportTarget, list[UploadedFile], tuple[str, ...]]:
        try:
            identifier = _DRAFT_ID.validate_python(draft_id)
        except ValidationError as exc:
            raise CollectionError(
                "invalid_reference_import", "Provide the exact draft_id from a preview.", 422
            ) from exc
        with self._connect() as connection:
            row = connection.execute(
                "SELECT created_at, request, import_id FROM reference_import_staging "
                "WHERE draft_id = ?",
                (identifier,),
            ).fetchone()
        try:
            expired = row is None or (
                datetime.fromisoformat(row["created_at"]) + DRAFT_LIFETIME < self._clock()
            )
            if not expired and row["import_id"] is None:
                request = json.loads(row["request"])
                target = ImportTarget.model_validate(request["target"])
                additions = request["additions"]
                if not isinstance(additions, list) or not all(
                    isinstance(item, str) for item in additions
                ):
                    raise ValueError("Invalid previewed additions.")
                files = [UploadedFile.from_json(item) for item in request["files"]]
                return target, files, tuple(additions)
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise CollectionError(
                "invalid_reference_import_record", "The saved draft is invalid.", 409
            ) from exc
        if expired:
            raise CollectionError(
                "reference_import_draft_not_found",
                "The preview draft is unknown or older than 24 hours. Preview the files again.",
                404,
            )
        raise CollectionError(
            "reference_import_already_committed",
            f"This preview was already imported as {row['import_id']}. Nothing was added again.",
            409,
        )

    def _absent(self, documents: Sequence[Document]) -> list[Document]:
        with self._connect() as connection:
            existing = {
                row[0]
                for row in connection.execute(
                    "SELECT document_id FROM documents "
                    "WHERE document_id IN (SELECT value FROM json_each(?))",
                    (json.dumps([document.document_id for document in documents]),),
                )
            }
        return [document for document in documents if document.document_id not in existing]

    def _ingest_new(self, documents: Sequence[Document]) -> set[str]:
        """Insert only absent documents; existing papers are never overwritten.

        The injected ingest is insert-only inside the store's write transaction. If another
        importer stores some of the same content-addressed papers first, nothing is written,
        those papers are reused and the rest are inserted on a fresh attempt.
        """
        for _ in range(3):
            absent = self._absent(documents)
            if not absent:
                return set()
            try:
                self._ingest(absent)
            except DocumentsExistError:
                continue
            return {document.document_id for document in absent}
        raise CollectionError(
            "reference_import_busy",
            "Other imports kept storing the same papers; retry the commit. No collection changed.",
            409,
        )

    def commit(self, draft_id: str) -> ReferenceImportResult:
        """Re-plan the draft at the previewed revision, then add only new papers atomically.

        Documents are stored before the collection save; if that save then fails (for
        example a stale revision), the new documents stay in the corpus unassigned and a
        later import reuses them by content ID.
        """
        with self._commit_lock:
            return self._commit(draft_id)

    def _commit(self, draft_id: str) -> ReferenceImportResult:
        """Serialized in-process, so concurrent commits never ingest the same draft twice."""
        try:
            target, files, previewed = self._draft(draft_id)
            name, current, plan = self._planned(files, target)
            additions = [item for item in plan.candidates if item.status == "new"]
            if tuple(item.document_id for item in additions) != previewed:
                # Member metadata can change without a revision change; never add unseen papers.
                raise CollectionError(
                    "reference_import_preview_stale",
                    "The papers to add changed since the preview (for example a member's DOI, "
                    "title or year changed). Preview again; nothing was written.",
                    409,
                )
            if not additions:
                raise CollectionError(
                    "reference_import_nothing_new",
                    "Every record is a duplicate or was skipped; nothing was written.",
                    409,
                )
            if current + len(additions) > MAX_DOCUMENT_IDS:
                raise CollectionError(
                    "reference_import_capacity_exceeded",
                    f"A collection holds at most {MAX_DOCUMENT_IDS} papers; this import would "
                    f"add {len(additions)} to {current}. Nothing was written.",
                    422,
                )
            identifiers = tuple(item.document_id for item in additions)
            created = self._ingest_new([plan.documents[item] for item in identifiers])
            saved: list[ReferenceImportRecord] = []

            def record_import(connection: sqlite3.Connection, collection: PaperCollection) -> None:
                record = ReferenceImportRecord(
                    import_id=f"imp_{uuid4().hex}",
                    draft_id=draft_id,
                    collection_id=collection.collection_id,
                    collection_revision=collection.revision,
                    created_at=self._clock(),
                    totals=plan.totals,
                    files=tuple(file.model_copy(update={"skipped": ()}) for file in plan.files),
                    papers=tuple(
                        ImportedPaper(
                            document_id=item.document_id,
                            file_index=item.file_index,
                            record_index=item.record_index,
                            created=item.document_id in created,
                        )
                        for item in additions
                    ),
                )
                if (
                    connection.execute(
                        "UPDATE reference_import_staging SET import_id = ? "
                        "WHERE draft_id = ? AND import_id IS NULL",
                        (record.import_id, draft_id),
                    ).rowcount
                    != 1
                ):
                    raise CollectionError(
                        "reference_import_already_committed",
                        "This preview was already imported. Nothing was added again.",
                        409,
                    )
                connection.execute(
                    "INSERT INTO reference_imports "
                    "(import_id, collection_id, collection_revision, created_at, record) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        record.import_id,
                        record.collection_id,
                        record.collection_revision,
                        record.created_at.isoformat(),
                        record.model_dump_json(),
                    ),
                )
                saved.append(record)

            if target.collection_id is None or target.collection_revision is None:
                collection = self._collections.create(
                    name=name, document_ids=identifiers, before_commit=record_import
                )
            else:
                collection = self._collections.edit(
                    target.collection_id,
                    changes=CollectionEdit(
                        name=name,
                        expected_revision=target.collection_revision,
                        add_document_ids=identifiers,
                    ),
                    before_commit=record_import,
                )
        except sqlite3.Error as exc:
            raise _storage_error() from exc
        return ReferenceImportResult(record=saved[0], collection=collection)

    def history(self, collection_id: str) -> ReferenceImportHistory:
        """Return up to 100 saved import records for an existing collection, newest first."""
        collection = self._collections.get(collection_id)
        try:
            with self._connect() as connection:
                records = self._read_history(connection, collection.collection_id)
        except sqlite3.Error as exc:
            raise _storage_error() from exc
        return ReferenceImportHistory(
            collection_id=collection.collection_id, imports=tuple(records)
        )
