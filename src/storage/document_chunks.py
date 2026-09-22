"""Bounded, read-only evidence pages from the existing persisted corpus."""

import base64
import binascii
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from storage.document_catalog import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MAX_SOURCE_CHARACTERS,
    MAX_TITLE_CHARACTERS,
    DocumentCatalogError,
    Identity,
    _prefix,
)

MAX_CHUNK_ID_CHARACTERS = 256
MAX_TEXT_CHARACTERS = 4000
MAX_CURSOR_CHARACTERS = 4096
MAX_CHUNK_INDEX = 2**63 - 1

ChunkIdentity = Annotated[str, Field(strict=True, min_length=1, max_length=MAX_CHUNK_ID_CHARACTERS)]
ChunkCursor = Annotated[str, Field(strict=True, min_length=1, max_length=MAX_CURSOR_CHARACTERS)]


class StoredChunk(BaseModel):
    """Stored evidence prefix and provenance, not a retrieval result or a citation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: ChunkIdentity
    document_id: Identity
    chunk_index: int | None = Field(ge=0, le=MAX_CHUNK_INDEX, strict=True)
    title: str = Field(max_length=MAX_TITLE_CHARACTERS)
    title_truncated: bool
    source: str = Field(max_length=MAX_SOURCE_CHARACTERS)
    source_truncated: bool
    text: str = Field(max_length=MAX_TEXT_CHARACTERS)
    text_truncated: bool


class DocumentChunksPage(BaseModel):
    """Exact document scope with ascending chunk-ID keyset continuation."""

    document_id: Identity
    chunks: list[StoredChunk] = Field(max_length=MAX_PAGE_SIZE)
    next_cursor: ChunkCursor | None


class _PageRequest(BaseModel):
    document_id: Identity
    limit: int = Field(default=DEFAULT_PAGE_SIZE, strict=True, ge=1, le=MAX_PAGE_SIZE)
    cursor: ChunkCursor | None = None


class _Cursor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: int = Field(ge=1, le=1)
    document_id: Identity
    chunk_id: ChunkIdentity

    def encode(self) -> str:
        return base64.urlsafe_b64encode(self.model_dump_json().encode("utf-8")).decode().rstrip("=")


class DocumentNotFoundError(LookupError):
    """No document exists, even if orphan chunk rows carry the requested ID."""

    code = "document_not_found"

    def __init__(self) -> None:
        super().__init__("The requested document is not in the stored corpus.")


class ChunkCursorError(ValueError):
    """Malformed, unsupported, or differently scoped continuation."""

    code = "invalid_chunk_cursor"

    def __init__(self) -> None:
        super().__init__("Use the next_cursor returned for this exact document.")


class DocumentChunksError(ValueError):
    """Invalid projected data, without revealing private stored payloads."""

    code = "invalid_chunk_record"

    def __init__(self) -> None:
        super().__init__("Saved chunk evidence or provenance is invalid or unsupported.")


def _decode_cursor(value: str, document_id: str) -> _Cursor:
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        cursor = _Cursor.model_validate_json(raw)
    except (ValueError, binascii.Error) as exc:
        raise ChunkCursorError() from exc
    if cursor.document_id != document_id or cursor.encode() != value:
        raise ChunkCursorError()
    return cursor


_CHUNKS_SQL = """
SELECT coalesce(substr(CAST(chunk_id AS BLOB), 1, :identity_bytes), X'') AS chunk_id,
       coalesce(substr(CAST(title AS BLOB), 1, :title_bytes), X'') AS title,
       coalesce(substr(CAST(source AS BLOB), 1, :source_bytes), X'') AS source,
       coalesce(substr(CAST(text AS BLOB), 1, :text_bytes), X'') AS text,
       typeof(chunk_id) != 'text' OR typeof(title) != 'text'
           OR typeof(source) != 'text' OR typeof(text) != 'text' AS invalid_record,
       CASE WHEN typeof(metadata) = 'text' AND json_valid(metadata) THEN
           CASE WHEN json_type(metadata) = 'object' THEN
               CASE WHEN json_type(metadata, '$.chunk_index') IS NULL THEN NULL
                    WHEN json_type(metadata, '$.chunk_index') IN ('text', 'integer')
                    THEN coalesce(substr(
                        CAST(CAST(json_extract(metadata, '$.chunk_index') AS TEXT) AS BLOB),
                        1, 84), X'')
                    ELSE X'00' END
               ELSE X'00' END
           ELSE X'00' END AS chunk_index
FROM chunks AS c
WHERE c.document_id = :document_id
"""
_FIRST_PAGE_SQL = _CHUNKS_SQL + " ORDER BY c.chunk_id ASC LIMIT :fetch_limit"
_NEXT_PAGE_SQL = (
    _CHUNKS_SQL + " AND c.chunk_id > :cursor ORDER BY c.chunk_id ASC LIMIT :fetch_limit"
)


def _chunk_index(value: object, encoding: str) -> int | None:
    if value is None:
        return None
    text = _prefix(value, encoding, 20)
    if not text.isascii() or not text.isdecimal():
        raise DocumentChunksError()
    ordinal = int(text)
    if ordinal > MAX_CHUNK_INDEX or str(ordinal) != text:
        raise DocumentChunksError()
    return ordinal


class SQLiteDocumentChunks:
    """Read an existing corpus without initializing storage or rebuilding retrieval."""

    def __init__(self, database_path: Path | str) -> None:
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"

    def list_chunks(
        self,
        document_id: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> DocumentChunksPage:
        """Read one document snapshot; separate page requests are not a frozen snapshot."""
        options = _PageRequest(document_id=document_id, limit=limit, cursor=cursor)
        after = _decode_cursor(options.cursor, options.document_id) if options.cursor else None
        with closing(sqlite3.connect(self._database_uri, uri=True)) as connection:
            connection.execute("BEGIN")
            exists = connection.execute(
                "SELECT 1 FROM documents WHERE document_id = ?", (options.document_id,)
            ).fetchone()
            if exists is None:
                raise DocumentNotFoundError()
            encoding: str = connection.execute("PRAGMA encoding").fetchone()[0]
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                _NEXT_PAGE_SQL if after else _FIRST_PAGE_SQL,
                {
                    "document_id": options.document_id,
                    "cursor": after.chunk_id if after else None,
                    "fetch_limit": options.limit + 1,
                    "identity_bytes": 4 * (MAX_CHUNK_ID_CHARACTERS + 1),
                    "title_bytes": 4 * (MAX_TITLE_CHARACTERS + 1),
                    "source_bytes": 4 * (MAX_SOURCE_CHARACTERS + 1),
                    "text_bytes": 4 * (MAX_TEXT_CHARACTERS + 1),
                },
            ).fetchall()
        chunks = []
        for row in rows:
            if row["invalid_record"]:
                raise DocumentChunksError()
            try:
                title = _prefix(row["title"], encoding, MAX_TITLE_CHARACTERS)
                source = _prefix(row["source"], encoding, MAX_SOURCE_CHARACTERS)
                text = _prefix(row["text"], encoding, MAX_TEXT_CHARACTERS)
                chunks.append(
                    StoredChunk(
                        chunk_id=_prefix(row["chunk_id"], encoding, MAX_CHUNK_ID_CHARACTERS),
                        document_id=options.document_id,
                        chunk_index=_chunk_index(row["chunk_index"], encoding),
                        title=title[:MAX_TITLE_CHARACTERS],
                        title_truncated=len(title) > MAX_TITLE_CHARACTERS,
                        source=source[:MAX_SOURCE_CHARACTERS],
                        source_truncated=len(source) > MAX_SOURCE_CHARACTERS,
                        text=text[:MAX_TEXT_CHARACTERS],
                        text_truncated=len(text) > MAX_TEXT_CHARACTERS,
                    )
                )
            except (DocumentCatalogError, ValidationError) as exc:
                raise DocumentChunksError() from exc
        page = chunks[: options.limit]
        return DocumentChunksPage(
            document_id=options.document_id,
            chunks=page,
            next_cursor=(
                _Cursor(
                    version=1, document_id=options.document_id, chunk_id=page[-1].chunk_id
                ).encode()
                if len(chunks) > options.limit
                else None
            ),
        )
