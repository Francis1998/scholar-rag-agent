"""Bounded, model-free literal passage search in the current persisted corpus."""

import base64
import binascii
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from retrieval.scope import MAX_DOCUMENT_ID_LENGTH, MAX_DOCUMENT_IDS, DocumentIds
from storage.document_catalog import (
    MAX_SOURCE_CHARACTERS,
    MAX_TITLE_CHARACTERS,
    DocumentCatalogError,
    Identity,
    _prefix,
)
from storage.document_chunks import (
    MAX_CHUNK_ID_CHARACTERS,
    MAX_CURSOR_CHARACTERS,
    ChunkIdentity,
    DocumentChunksError,
    document_chunks_url,
)
from storage.paper_collections import CollectionId, Revision, SQLitePaperCollections

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
MAX_QUERY_CHARACTERS = 200
MAX_EXCERPT_CHARACTERS = 800
CONTEXT_BEFORE_CHARACTERS = 120
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 262144
SEARCH_TIMEOUT_SECONDS = 5


def _unicode(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Text must contain valid Unicode.") from exc
    return value


def _valid_text(value: bytes, *, encoding: str) -> bool:
    # SQLite permits malformed TEXT; instr alone can otherwise invent Unicode offsets.
    try:
        value.decode(encoding)
    except UnicodeDecodeError:
        return False
    return True


SearchQuery = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=MAX_QUERY_CHARACTERS),
    AfterValidator(_unicode),
]
SearchCursor = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=MAX_CURSOR_CHARACTERS),
    AfterValidator(_unicode),
]


class LiteralSearchRequest(BaseModel):
    """Omission means whole corpus; an explicit empty/null scope never does."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: SearchQuery
    document_ids: DocumentIds | SkipJsonSchema[None] = None
    collection_id: CollectionId | SkipJsonSchema[None] = None
    limit: int = Field(default=DEFAULT_LIMIT, strict=True, ge=1, le=MAX_LIMIT)
    cursor: SearchCursor | SkipJsonSchema[None] = None

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The literal query cannot be blank.")
        return value

    @field_validator("document_ids", "collection_id", "cursor", mode="before")
    @classmethod
    def nonnull_option(cls, value: object) -> object:
        if value is None:
            raise ValueError("Omit unused options instead of sending null.")
        return value

    @model_validator(mode="after")
    def exact_scope(self) -> Self:
        if self.document_ids is not None and self.collection_id is not None:
            raise ValueError("Supply document_ids or collection_id, not both.")
        for identifier in self.document_ids or ():
            _unicode(identifier)
        return self


class LiteralSearchError(ValueError):
    """Sanitized all-or-nothing search diagnostic, shared by Python and HTTP."""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _invalid_cursor() -> LiteralSearchError:
    return LiteralSearchError(
        "invalid_search_cursor",
        "Use the returned next_cursor with the same exact query and scope; "
        "restart pagination after a collection revision changes.",
        422,
    )


class LiteralMatch(BaseModel):
    """First literal occurrence in one chunk; offsets are half-open Unicode characters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: Identity
    chunk_id: ChunkIdentity
    title: str = Field(max_length=MAX_TITLE_CHARACTERS)
    title_truncated: bool
    source: str = Field(max_length=MAX_SOURCE_CHARACTERS)
    source_truncated: bool
    match_start: int = Field(strict=True, ge=0)
    match_end: int = Field(strict=True, ge=1)
    excerpt: str = Field(max_length=MAX_EXCERPT_CHARACTERS)
    excerpt_start: int = Field(strict=True, ge=0)
    excerpt_end: int = Field(strict=True, ge=1)
    excerpt_truncated_before: bool
    excerpt_truncated_after: bool
    text_characters: int = Field(strict=True, ge=1)
    inspection_url: str | None = Field(max_length=1600)


class LiteralSearchPage(BaseModel):
    """Current-corpus matches in binary document/chunk ID order, without relevance scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["literal_passages"] = "literal_passages"
    query: SearchQuery
    document_ids: tuple[Identity, ...] | None = Field(max_length=MAX_DOCUMENT_IDS)
    collection_id: CollectionId | None
    collection_revision: Revision | None
    matches: list[LiteralMatch] = Field(max_length=MAX_LIMIT)
    next_cursor: SearchCursor | None

    def to_json(self) -> str:
        """Use the exact same bounded UTF-8 representation in Python and HTTP."""
        content = self.model_dump_json()
        if len(content.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise LiteralSearchError(
                "search_response_too_large",
                "Search exceeds the 262144-byte UTF-8 response limit; request a smaller limit.",
                413,
            )
        return content


class _Cursor(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    version: int = Field(ge=1, le=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: Annotated[Identity, AfterValidator(_unicode)]
    chunk_id: Annotated[ChunkIdentity, AfterValidator(_unicode)]

    def encode(self) -> str:
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")


def _decode_cursor(value: str, fingerprint: str) -> _Cursor:
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        cursor = _Cursor.model_validate_json(raw)
    except (ValueError, binascii.Error) as exc:
        raise _invalid_cursor() from exc
    if cursor.fingerprint != fingerprint or cursor.encode() != value:
        raise _invalid_cursor()
    return cursor


_SEARCH_SQL = """
WITH candidates AS (
    SELECT c.document_id, c.chunk_id, c.title, c.source, c.text,
           typeof(c.document_id) != 'text' OR typeof(c.chunk_id) != 'text'
             OR typeof(c.title) != 'text' OR typeof(c.source) != 'text'
             OR typeof(c.text) != 'text'
             OR CASE WHEN typeof(c.text) = 'text'
                          AND length(CAST(c.text AS BLOB)) <= :text_bytes
                     THEN instr(c.text, char(0)) > 0
                          OR NOT valid_search_text(CAST(c.text AS BLOB))
                     ELSE 0 END
             OR NOT EXISTS (
                 SELECT 1 FROM documents AS d
                 WHERE d.document_id COLLATE BINARY = c.document_id
             ) AS invalid_record,
           length(CAST(c.text AS BLOB)) > :text_bytes AS too_large,
           CASE WHEN length(CAST(c.text AS BLOB)) <= :text_bytes
                THEN instr(c.text, :query) ELSE 0 END AS position
    FROM chunks AS c
    WHERE (:scope IS NULL OR c.document_id COLLATE BINARY IN (
        SELECT value FROM json_each(:scope)
    ))
      AND (:after_document IS NULL OR
           (c.document_id COLLATE BINARY, c.chunk_id COLLATE BINARY)
               > (:after_document, :after_chunk))
)
SELECT substr(CAST(document_id AS BLOB), 1, :document_bytes) AS document_id,
       substr(CAST(chunk_id AS BLOB), 1, :chunk_bytes) AS chunk_id,
       substr(CAST(title AS BLOB), 1, :title_bytes) AS title,
       substr(CAST(source AS BLOB), 1, :source_bytes) AS source,
       invalid_record, too_large, position - 1 AS match_start,
       CASE WHEN invalid_record OR too_large THEN 0 ELSE length(text) END AS text_characters,
       max(0, position - 1 - :context_before) AS excerpt_start,
       CASE WHEN invalid_record OR too_large THEN X'' ELSE
           CAST(substr(text, max(1, position - :context_before), :excerpt_characters) AS BLOB)
       END AS excerpt
FROM candidates
WHERE invalid_record OR too_large OR position > 0
ORDER BY candidates.document_id COLLATE BINARY, candidates.chunk_id COLLATE BINARY
LIMIT :fetch_limit
"""


class SQLiteLiteralSearch:
    """Open existing storage read-only; never construct a retriever, model, or schema."""

    def __init__(self, database_path: Path | str) -> None:
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"

    def search(self, request: LiteralSearchRequest) -> LiteralSearchPage:
        """Read one snapshot; separate cursor requests do not freeze corpus contents."""
        options = LiteralSearchRequest.model_validate(
            {name: getattr(request, name) for name in request.model_fields_set}
        )
        deadline = time.monotonic() + SEARCH_TIMEOUT_SECONDS
        try:
            with closing(sqlite3.connect(self._database_uri, uri=True, timeout=5)) as connection:
                connection.row_factory = sqlite3.Row
                connection.set_progress_handler(lambda: time.monotonic() >= deadline, 1000)
                connection.execute("BEGIN")
                encoding: str = connection.execute("PRAGMA encoding").fetchone()[0]
                connection.create_function(
                    "valid_search_text",
                    1,
                    partial(_valid_text, encoding=encoding),
                    deterministic=True,
                )
                document_ids = options.document_ids
                revision = None
                if options.collection_id is not None:
                    collection = SQLitePaperCollections.resolve_in_snapshot(
                        connection, options.collection_id
                    )
                    document_ids = collection.document_ids
                    revision = collection.revision
                scope = None if document_ids is None else tuple(sorted(document_ids))
                fingerprint = hashlib.sha256(
                    json.dumps(
                        [options.query, scope, options.collection_id, revision],
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                after = _decode_cursor(options.cursor, fingerprint) if options.cursor else None
                rows = connection.execute(
                    _SEARCH_SQL,
                    {
                        "query": options.query,
                        "scope": None if scope is None else json.dumps(scope),
                        "after_document": after.document_id if after else None,
                        "after_chunk": after.chunk_id if after else None,
                        "fetch_limit": options.limit + 1,
                        "document_bytes": 4 * (MAX_DOCUMENT_ID_LENGTH + 1),
                        "chunk_bytes": 4 * (MAX_CHUNK_ID_CHARACTERS + 1),
                        "title_bytes": 4 * (MAX_TITLE_CHARACTERS + 1),
                        "source_bytes": 4 * (MAX_SOURCE_CHARACTERS + 1),
                        "text_bytes": MAX_TEXT_BYTES,
                        "context_before": CONTEXT_BEFORE_CHARACTERS,
                        "excerpt_characters": MAX_EXCERPT_CHARACTERS,
                    },
                ).fetchall()
        except sqlite3.Error as exc:
            if time.monotonic() >= deadline:
                raise LiteralSearchError(
                    "search_timeout",
                    "Literal search exceeded its read deadline; narrow scope.",
                    504,
                ) from exc
            raise LiteralSearchError(
                "search_storage_unavailable", "Corpus or collection storage is unavailable.", 503
            ) from exc
        matches = [self._match(row, encoding, options.query) for row in rows]
        if len({(match.document_id, match.chunk_id) for match in matches}) != len(matches):
            raise DocumentChunksError()
        page = matches[: options.limit]
        result = LiteralSearchPage(
            query=options.query,
            document_ids=scope,
            collection_id=options.collection_id,
            collection_revision=revision,
            matches=page,
            next_cursor=(
                _Cursor(
                    version=1,
                    fingerprint=fingerprint,
                    document_id=page[-1].document_id,
                    chunk_id=page[-1].chunk_id,
                ).encode()
                if len(matches) > options.limit
                else None
            ),
        )
        result.to_json()
        return result

    @staticmethod
    def _match(row: sqlite3.Row, encoding: str, query: str) -> LiteralMatch:
        if row["invalid_record"]:
            raise DocumentChunksError()
        if row["too_large"]:
            raise LiteralSearchError(
                "search_text_read_limit",
                "An encountered passage exceeds the 4194304-byte stored-text read limit.",
                413,
            )
        try:
            document_id = _prefix(row["document_id"], encoding, MAX_DOCUMENT_ID_LENGTH)
            title = _prefix(row["title"], encoding, MAX_TITLE_CHARACTERS)
            source = _prefix(row["source"], encoding, MAX_SOURCE_CHARACTERS)
            excerpt = _prefix(row["excerpt"], encoding, MAX_EXCERPT_CHARACTERS)
            start = row["match_start"]
            excerpt_start = row["excerpt_start"]
            end = start + len(query)
            if (
                excerpt[start - excerpt_start : end - excerpt_start] != query
                or excerpt_start + len(excerpt) > row["text_characters"]
            ):
                raise DocumentChunksError()
            return LiteralMatch(
                document_id=document_id,
                chunk_id=_prefix(row["chunk_id"], encoding, MAX_CHUNK_ID_CHARACTERS),
                title=title[:MAX_TITLE_CHARACTERS],
                title_truncated=len(title) > MAX_TITLE_CHARACTERS,
                source=source[:MAX_SOURCE_CHARACTERS],
                source_truncated=len(source) > MAX_SOURCE_CHARACTERS,
                match_start=start,
                match_end=end,
                excerpt=excerpt,
                excerpt_start=excerpt_start,
                excerpt_end=excerpt_start + len(excerpt),
                excerpt_truncated_before=excerpt_start > 0,
                excerpt_truncated_after=excerpt_start + len(excerpt) < row["text_characters"],
                text_characters=row["text_characters"],
                inspection_url=document_chunks_url(document_id),
            )
        except (DocumentCatalogError, ValidationError, UnicodeError) as exc:
            raise DocumentChunksError() from exc
