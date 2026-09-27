"""Transactional, revisioned document selections; never copies or deletes corpus data."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

from retrieval.scope import MAX_DOCUMENT_ID_LENGTH, MAX_DOCUMENT_IDS, DocumentIds, DocumentIdsInput

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_NAME_LENGTH = 120
MAX_REVISION = 2**63 - 1
# ensure_ascii JSON represents one astral code point as two six-character surrogate escapes.
_MAX_MEMBERSHIP_CHARACTERS = MAX_DOCUMENT_IDS * (12 * MAX_DOCUMENT_ID_LENGTH + 4) + 2

CollectionId = Annotated[
    str,
    StringConstraints(strict=True, min_length=36, max_length=36, pattern=r"^col_[0-9a-f]{32}$"),
]
Revision = Annotated[int, Field(strict=True, ge=1, le=MAX_REVISION)]
_COLLECTION_ID = TypeAdapter(CollectionId)
_REVISION = TypeAdapter(Revision)
_DOCUMENT_IDS = TypeAdapter(DocumentIds)


def _readable_name(value: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Collection names cannot contain control characters.")
    return value


CollectionName = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=MAX_NAME_LENGTH),
    AfterValidator(_readable_name),
]


class CollectionSelection(BaseModel):
    """A complete selection, with no caller-owned mutable membership."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: CollectionName
    document_ids: DocumentIds


class CollectionReplacement(CollectionSelection):
    """Full replacement with mandatory optimistic concurrency control."""

    expected_revision: Revision


class PaperCollection(CollectionSelection):
    """Current saved metadata, not a snapshot of document contents."""

    collection_id: CollectionId
    revision: Revision


class CollectionSummary(BaseModel):
    """List metadata without document bodies, memberships, or run information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    name: CollectionName
    revision: Revision
    document_count: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)


class CollectionPage(BaseModel):
    collections: list[CollectionSummary] = Field(max_length=MAX_PAGE_SIZE)
    next_cursor: CollectionId | None


class _PageRequest(BaseModel):
    limit: int = Field(default=DEFAULT_PAGE_SIZE, strict=True, ge=1, le=MAX_PAGE_SIZE)
    cursor: CollectionId | None = None


class CollectionError(ValueError):
    """Public diagnostic with bounded fields, never a raw SQLite record or exception."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        *,
        missing_document_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.missing_document_ids = missing_document_ids


def _invalid_record() -> CollectionError:
    return CollectionError(
        "invalid_collection_record", "Saved collection metadata is invalid.", 409
    )


_SELECT = """
SELECT substr(collection_id, 1, 37) AS collection_id,
       substr(name, 1, 121) AS name, revision,
       substr(document_ids, 1, :membership_limit + 1) AS document_ids,
       typeof(collection_id) != 'text' OR typeof(name) != 'text'
         OR typeof(document_ids) != 'text' OR typeof(revision) != 'integer'
         OR instr(collection_id, char(0)) > 0
         OR instr(name, char(0)) > 0 OR instr(document_ids, char(0)) > 0
         OR length(document_ids) > :membership_limit AS invalid_record
FROM paper_collections
"""


class SQLitePaperCollections:
    """Persist metadata in the corpus database; each operation owns its connection."""

    def __init__(self, database_path: Path | str) -> None:
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=rw"
        try:
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS paper_collections (
                        collection_id TEXT PRIMARY KEY NOT NULL,
                        name TEXT NOT NULL UNIQUE COLLATE BINARY,
                        revision INTEGER NOT NULL CHECK(revision >= 1),
                        document_ids TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise self._storage_error() from exc

    @staticmethod
    def _storage_error() -> CollectionError:
        return CollectionError(
            "collection_storage_error", "Collection storage is unavailable; retry later.", 503
        )

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        try:
            with (
                closing(sqlite3.connect(self._database_uri, uri=True, timeout=5)) as connection,
                connection,
            ):
                connection.row_factory = sqlite3.Row
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield connection
        except sqlite3.Error as exc:
            raise self._storage_error() from exc

    @staticmethod
    def _record(row: sqlite3.Row) -> PaperCollection:
        if row["invalid_record"]:
            raise _invalid_record()
        try:
            identifiers = json.loads(row["document_ids"])
            collection = PaperCollection(
                collection_id=row["collection_id"],
                name=row["name"],
                revision=row["revision"],
                document_ids=identifiers,
            )
        except (ValueError, TypeError, RecursionError) as exc:
            raise _invalid_record() from exc
        # Saved data must already be normalized; never silently repair a corrupt selection.
        if collection.name != row["name"] or list(collection.document_ids) != identifiers:
            raise _invalid_record()
        return collection

    def _get(self, connection: sqlite3.Connection, collection_id: str) -> PaperCollection:
        row = connection.execute(
            _SELECT + " WHERE collection_id = :collection_id",
            {"collection_id": collection_id, "membership_limit": _MAX_MEMBERSHIP_CHARACTERS},
        ).fetchone()
        if row is None:
            raise CollectionError("collection_not_found", "Collection does not exist.", 404)
        return self._record(row)

    @staticmethod
    def _missing_documents(
        connection: sqlite3.Connection, document_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        rows = connection.execute(
            "SELECT document_id FROM documents "
            "WHERE document_id IN (SELECT value FROM json_each(?))",
            (json.dumps(document_ids),),
        )
        existing = {row[0] for row in rows}
        return tuple(identifier for identifier in document_ids if identifier not in existing)

    def _validate_documents(
        self, connection: sqlite3.Connection, document_ids: tuple[str, ...]
    ) -> None:
        missing = self._missing_documents(connection, document_ids)
        if missing:
            raise CollectionError(
                "unknown_documents",
                "All selected documents must exist in the corpus.",
                422,
                missing_document_ids=missing,
            )

    @staticmethod
    def _check_name(connection: sqlite3.Connection, name: str, collection_id: str) -> None:
        if connection.execute(
            "SELECT 1 FROM paper_collections WHERE name = ? AND collection_id != ?",
            (name, collection_id),
        ).fetchone():
            raise CollectionError(
                "collection_name_conflict", "A collection with that exact name already exists.", 409
            )

    @staticmethod
    def _check_revision(
        connection: sqlite3.Connection, collection_id: str, expected_revision: int
    ) -> None:
        row = connection.execute(
            "SELECT revision FROM paper_collections WHERE collection_id = ?", (collection_id,)
        ).fetchone()
        if row is None:
            raise CollectionError("collection_not_found", "Collection does not exist.", 404)
        try:
            revision = _REVISION.validate_python(row[0])
        except ValidationError as exc:
            raise _invalid_record() from exc
        if revision != expected_revision:
            raise CollectionError(
                "collection_revision_conflict",
                "Collection changed; read it again before replacing or deleting.",
                409,
            )

    def create(self, *, name: str, document_ids: DocumentIdsInput) -> PaperCollection:
        selection = CollectionSelection.model_validate({"name": name, "document_ids": document_ids})
        collection = PaperCollection(
            **selection.model_dump(), collection_id=f"col_{uuid4().hex}", revision=1
        )
        with self._connection(write=True) as connection:
            self._validate_documents(connection, collection.document_ids)
            self._check_name(connection, collection.name, collection.collection_id)
            connection.execute(
                "INSERT INTO paper_collections (collection_id, name, revision, document_ids) "
                "VALUES (?, ?, ?, ?)",
                (collection.collection_id, collection.name, 1, json.dumps(collection.document_ids)),
            )
        return collection

    def get(self, collection_id: str) -> PaperCollection:
        identifier = _COLLECTION_ID.validate_python(collection_id)
        with self._connection() as connection:
            return self._get(connection, identifier)

    def list_collections(
        self, *, limit: int = DEFAULT_PAGE_SIZE, cursor: str | None = None
    ) -> CollectionPage:
        options = _PageRequest(limit=limit, cursor=cursor)
        with self._connection() as connection:
            rows = connection.execute(
                _SELECT + " WHERE (:cursor IS NULL OR collection_id > :cursor)"
                " ORDER BY collection_id ASC LIMIT :fetch_limit",
                {
                    "cursor": options.cursor,
                    "fetch_limit": options.limit + 1,
                    "membership_limit": _MAX_MEMBERSHIP_CHARACTERS,
                },
            ).fetchall()
        records = [self._record(row) for row in rows]
        summaries = [
            CollectionSummary(
                collection_id=record.collection_id,
                name=record.name,
                revision=record.revision,
                document_count=len(record.document_ids),
            )
            for record in records[: options.limit]
        ]
        return CollectionPage(
            collections=summaries,
            next_cursor=summaries[-1].collection_id if len(records) > options.limit else None,
        )

    def replace(
        self,
        collection_id: str,
        *,
        name: str,
        document_ids: DocumentIdsInput,
        expected_revision: int,
    ) -> PaperCollection:
        identifier = _COLLECTION_ID.validate_python(collection_id)
        replacement = CollectionReplacement.model_validate(
            {"name": name, "document_ids": document_ids, "expected_revision": expected_revision}
        )
        with self._connection(write=True) as connection:
            self._check_revision(connection, identifier, replacement.expected_revision)
            if replacement.expected_revision == MAX_REVISION:
                raise CollectionError(
                    "collection_revision_exhausted", "Collection revision cannot increase.", 409
                )
            self._validate_documents(connection, replacement.document_ids)
            self._check_name(connection, replacement.name, identifier)
            collection = PaperCollection(
                collection_id=identifier,
                name=replacement.name,
                document_ids=replacement.document_ids,
                revision=replacement.expected_revision + 1,
            )
            connection.execute(
                "UPDATE paper_collections SET name = ?, document_ids = ?, revision = ? "
                "WHERE collection_id = ? AND revision = ?",
                (
                    collection.name,
                    json.dumps(collection.document_ids),
                    collection.revision,
                    identifier,
                    replacement.expected_revision,
                ),
            )
        return collection

    def delete(self, collection_id: str, *, expected_revision: int) -> None:
        identifier = _COLLECTION_ID.validate_python(collection_id)
        revision = _REVISION.validate_python(expected_revision)
        with self._connection(write=True) as connection:
            self._check_revision(connection, identifier, revision)
            connection.execute(
                "DELETE FROM paper_collections WHERE collection_id = ? AND revision = ?",
                (identifier, revision),
            )

    def resolve(self, collection_id: str) -> tuple[str, ...]:
        """Freeze membership and check existence in one read transaction, before any await."""
        identifier = _COLLECTION_ID.validate_python(collection_id)
        with self._connection() as connection:
            collection = self._get(connection, identifier)
            if self._missing_documents(connection, collection.document_ids):
                raise CollectionError(
                    "collection_documents_missing",
                    "Collection references missing documents; replace or delete its metadata.",
                    409,
                )
            return collection.document_ids

    def validate_document_ids(self, document_ids: DocumentIdsInput) -> tuple[str, ...]:
        """Check an explicit selection with the same existence rules, without saving metadata."""
        selection = _DOCUMENT_IDS.validate_python(document_ids)
        with self._connection() as connection:
            self._validate_documents(connection, selection)
        return selection
