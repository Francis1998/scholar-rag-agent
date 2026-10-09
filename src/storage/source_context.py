"""Bounded source-order reading around an exact chunk in the current corpus."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from storage.document_catalog import DocumentCatalogError, Identity, _prefix
from storage.document_chunks import (
    _CHUNK_PROJECTION_SQL,
    MAX_CHUNK_ID_CHARACTERS,
    MAX_CHUNK_INDEX,
    ChunkIdentity,
    DocumentChunksError,
    DocumentNotFoundError,
    StoredChunk,
    _chunk_index,
    _projection_limits,
    _stored_chunk,
)

DEFAULT_CONTEXT_NEIGHBORS = 2
MAX_CONTEXT_NEIGHBORS = 5
MAX_CONTEXT_DOCUMENT_CHUNKS = 2048
MAX_CONTEXT_METADATA_BYTES = 8192


def _unicode_identity(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Identifiers must contain valid Unicode.") from exc
    return value


ContextDocumentIdentity = Annotated[Identity, AfterValidator(_unicode_identity)]
ContextChunkIdentity = Annotated[ChunkIdentity, AfterValidator(_unicode_identity)]
_Ordinal = Annotated[int, Field(strict=True, ge=0, le=MAX_CHUNK_INDEX)]
_Count = Annotated[int, Field(strict=True, ge=0, le=MAX_CONTEXT_NEIGHBORS)]


class _ContextRequest(BaseModel):
    document_id: ContextDocumentIdentity
    chunk_id: ContextChunkIdentity
    before: _Count = DEFAULT_CONTEXT_NEIGHBORS
    after: _Count = DEFAULT_CONTEXT_NEIGHBORS


class _Position(BaseModel):
    chunk_id: ChunkIdentity
    chunk_index: _Ordinal


class ContextChunk(StoredChunk):
    """A bounded stored passage with a required source ordinal and explicit anchor."""

    chunk_index: _Ordinal
    is_anchor: bool


class SourceContext(BaseModel):
    """One current-corpus snapshot in numeric source order, not a saved run or ranking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: Identity
    anchor_chunk_id: ChunkIdentity
    anchor_chunk_index: _Ordinal
    before: _Count
    after: _Count
    returned_before: _Count
    returned_after: _Count
    has_more_before: bool
    has_more_after: bool
    chunks: list[ContextChunk] = Field(min_length=1, max_length=2 * MAX_CONTEXT_NEIGHBORS + 1)


class SourceContextError(ValueError):
    """A sanitized all-or-nothing failure shared by Python, JSON, and HTML readers."""

    def __init__(self, code: str, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


_ORDER_SQL = """
WITH bounded AS (
    SELECT chunk_id,
           typeof(chunk_id) != 'text' AS invalid_identity,
           length(CAST(metadata AS BLOB)) > :metadata_bytes AS metadata_too_large,
           CASE WHEN typeof(metadata) = 'text'
                     AND length(CAST(metadata AS BLOB)) <= :metadata_bytes
                THEN metadata ELSE '' END AS metadata
    FROM chunks
    WHERE document_id COLLATE BINARY = :document_id
    LIMIT :fetch_limit
)
SELECT coalesce(substr(CAST(chunk_id AS BLOB), 1, :identity_bytes), X'') AS chunk_id,
       invalid_identity, metadata_too_large,
       CASE WHEN json_valid(metadata) AND instr(metadata, char(0)) = 0 THEN
           CASE WHEN json_type(metadata) = 'object'
                     AND (SELECT count(*) FROM json_each(metadata)
                          WHERE key = 'chunk_index') = 1
                     AND json_type(metadata, '$.chunk_index') IN ('text', 'integer')
                THEN coalesce(substr(
                    CAST(CAST(json_extract(metadata, '$.chunk_index') AS TEXT) AS BLOB),
                    1, 84), X'')
                ELSE X'00' END
           ELSE X'00' END AS chunk_index
FROM bounded
"""
_WINDOW_SQL = (
    _CHUNK_PROJECTION_SQL  # noqa: S608 - static fragments; all selectors are bound parameters
    + """
WHERE c.document_id COLLATE BINARY = :document_id
  AND c.chunk_id COLLATE BINARY IN (SELECT value FROM json_each(:chunk_ids))
LIMIT :fetch_limit
"""
)


def _positions(rows: list[sqlite3.Row], encoding: str) -> list[_Position]:
    if len(rows) > MAX_CONTEXT_DOCUMENT_CHUNKS:
        raise SourceContextError(
            "source_context_document_too_large",
            "Source ordering validation is limited to 2048 stored chunks per document.",
        )
    positions = []
    for row in rows:
        if row["metadata_too_large"]:
            raise SourceContextError(
                "source_context_metadata_too_large",
                "Source ordering metadata exceeds 8192 stored bytes per chunk.",
            )
        try:
            if row["invalid_identity"]:
                raise DocumentChunksError()
            ordinal = _chunk_index(row["chunk_index"], encoding)
            if ordinal is None:
                raise DocumentChunksError()
            positions.append(
                _Position(
                    chunk_id=_prefix(row["chunk_id"], encoding, MAX_CHUNK_ID_CHARACTERS),
                    chunk_index=ordinal,
                )
            )
        except (DocumentCatalogError, DocumentChunksError, ValidationError) as exc:
            raise SourceContextError(
                "invalid_source_order",
                "Every stored chunk in this document needs a valid identity and exactly one "
                "nonnegative canonical chunk_index. No source order was guessed.",
            ) from exc
    if len({position.chunk_id for position in positions}) != len(positions) or len(
        {position.chunk_index for position in positions}
    ) != len(positions):
        raise SourceContextError(
            "ambiguous_source_order",
            "Duplicate chunk identities or source indices make this document's order ambiguous.",
        )
    return sorted(positions, key=lambda position: position.chunk_index)


class SQLiteSourceContext:
    """Inspect existing storage without initialization, retrieval, models, or writes."""

    def __init__(self, database_path: Path | str) -> None:
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"

    def read(
        self,
        document_id: str,
        chunk_id: str,
        *,
        before: int = DEFAULT_CONTEXT_NEIGHBORS,
        after: int = DEFAULT_CONTEXT_NEIGHBORS,
    ) -> SourceContext:
        """Read nearest stored neighbors by unique source index, allowing index gaps."""
        options = _ContextRequest(
            document_id=document_id, chunk_id=chunk_id, before=before, after=after
        )
        try:
            with closing(sqlite3.connect(self._database_uri, uri=True)) as connection:
                connection.execute("BEGIN")
                documents = connection.execute(
                    "SELECT 1 FROM documents WHERE document_id COLLATE BINARY = ? LIMIT 2",
                    (options.document_id,),
                ).fetchall()
                if not documents:
                    raise DocumentNotFoundError()
                if len(documents) != 1:
                    raise DocumentChunksError()
                anchor_exists = connection.execute(
                    "SELECT 1 FROM chunks WHERE document_id COLLATE BINARY = ? "
                    "AND chunk_id COLLATE BINARY = ? LIMIT 1",
                    (options.document_id, options.chunk_id),
                ).fetchone()
                if anchor_exists is None:
                    raise SourceContextError(
                        "chunk_not_found",
                        "The requested chunk is not in this document in the current corpus.",
                        404,
                    )
                encoding: str = connection.execute("PRAGMA encoding").fetchone()[0]
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    _ORDER_SQL,
                    {
                        "document_id": options.document_id,
                        "identity_bytes": 4 * (MAX_CHUNK_ID_CHARACTERS + 1),
                        "metadata_bytes": MAX_CONTEXT_METADATA_BYTES,
                        "fetch_limit": MAX_CONTEXT_DOCUMENT_CHUNKS + 1,
                    },
                ).fetchall()
                positions = _positions(rows, encoding)
                anchor = next(
                    index
                    for index, position in enumerate(positions)
                    if position.chunk_id == options.chunk_id
                )
                start = max(0, anchor - options.before)
                stop = min(len(positions), anchor + options.after + 1)
                selected = positions[start:stop]
                rows = connection.execute(
                    _WINDOW_SQL,
                    {
                        "document_id": options.document_id,
                        "chunk_ids": json.dumps([position.chunk_id for position in selected]),
                        "fetch_limit": len(selected),
                        **_projection_limits(),
                    },
                ).fetchall()
        except sqlite3.Error as exc:
            raise SourceContextError(
                "source_context_storage_unavailable",
                "The existing corpus database could not be read.",
                503,
            ) from exc
        stored = [_stored_chunk(row, encoding, options.document_id) for row in rows]
        by_id = {chunk.chunk_id: chunk for chunk in stored}
        if len(stored) != len(selected) or len(by_id) != len(selected):
            raise DocumentChunksError()
        chunks = []
        for position in selected:
            chunk = by_id.get(position.chunk_id)
            if chunk is None or chunk.chunk_index != position.chunk_index:
                raise DocumentChunksError()
            chunks.append(
                ContextChunk(
                    **chunk.model_dump(),
                    is_anchor=chunk.chunk_id == options.chunk_id,
                )
            )
        return SourceContext(
            document_id=options.document_id,
            anchor_chunk_id=options.chunk_id,
            anchor_chunk_index=positions[anchor].chunk_index,
            before=options.before,
            after=options.after,
            returned_before=anchor - start,
            returned_after=stop - anchor - 1,
            has_more_before=start > 0,
            has_more_after=stop < len(positions),
            chunks=chunks,
        )
