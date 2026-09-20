"""Read-only, bounded corpus discovery from the existing document tables."""

import codecs
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from retrieval.scope import MAX_DOCUMENT_ID_LENGTH

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_TITLE_CHARACTERS = 300
MAX_SOURCE_CHARACTERS = 512


def _exact_identity(value: str) -> str:
    if value != value.strip():
        raise ValueError("Document identifiers cannot have surrounding whitespace.")
    return value


Identity = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=MAX_DOCUMENT_ID_LENGTH),
    AfterValidator(_exact_identity),
]
SourceFilter = Annotated[str, Field(strict=True, min_length=1, max_length=MAX_SOURCE_CHARACTERS)]
TitleFilter = Annotated[str, Field(strict=True, min_length=1, max_length=MAX_TITLE_CHARACTERS)]


class DocumentSummary(BaseModel):
    """Selectable document identity and bounded labels, never body or metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: Identity
    title: str = Field(max_length=MAX_TITLE_CHARACTERS)
    title_truncated: bool
    source: str = Field(max_length=MAX_SOURCE_CHARACTERS)
    source_truncated: bool
    chunk_count: int = Field(strict=True, ge=0)


class DocumentCatalogPage(BaseModel):
    """Document-ID-ordered page with an exclusive continuation cursor."""

    documents: list[DocumentSummary] = Field(max_length=MAX_PAGE_SIZE)
    next_cursor: Identity | None


class _PageRequest(BaseModel):
    limit: int = Field(default=DEFAULT_PAGE_SIZE, strict=True, ge=1, le=MAX_PAGE_SIZE)
    cursor: Identity | None = None
    source: SourceFilter | None = None
    title: TitleFilter | None = None


class DocumentCatalogError(ValueError):
    """Invalid saved labels or identifiers, without echoing private stored content."""

    code = "invalid_document_record"

    def __init__(self) -> None:
        super().__init__("Saved document summary data is invalid or not selectable.")


_CATALOG_SQL = """
SELECT substr(CAST(d.document_id AS BLOB), 1, :identity_bytes) AS document_id,
       substr(CAST(d.title AS BLOB), 1, :title_bytes) AS title,
       substr(CAST(d.source AS BLOB), 1, :source_bytes) AS source,
       typeof(d.document_id) != 'text' OR typeof(d.title) != 'text'
           OR typeof(d.source) != 'text' AS invalid_record,
       (SELECT count(*) FROM chunks AS c WHERE c.document_id = d.document_id) AS chunk_count
FROM documents AS d
WHERE (:cursor IS NULL OR d.document_id > :cursor)
  AND (:source IS NULL OR d.source = :source)
  AND (:title IS NULL OR instr(lower(d.title), lower(:title)) > 0)
ORDER BY d.document_id ASC
LIMIT :fetch_limit
"""


def _prefix(value: object, encoding: str, characters: int) -> str:
    if not isinstance(value, bytes):
        raise DocumentCatalogError()
    try:
        # A bounded byte prefix may end inside a Unicode code point.
        return codecs.getincrementaldecoder(encoding)().decode(
            value, final=len(value) < 4 * (characters + 1)
        )
    except UnicodeDecodeError as exc:
        raise DocumentCatalogError() from exc


class SQLiteDocumentCatalog:
    """Inspect an existing corpus without loading chunks, indexing, or writing data."""

    def __init__(self, database_path: Path | str) -> None:
        """Open existing storage read-only; never initialize or backfill a corpus."""
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"

    def list_documents(
        self,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        source: str | None = None,
        title: str | None = None,
    ) -> DocumentCatalogPage:
        """Read a filtered page; changes between requests are not a frozen snapshot."""
        options = _PageRequest(limit=limit, cursor=cursor, source=source, title=title)
        with closing(sqlite3.connect(self._database_uri, uri=True)) as connection:
            encoding: str = connection.execute("PRAGMA encoding").fetchone()[0]
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                _CATALOG_SQL,
                {
                    "cursor": options.cursor,
                    "source": options.source,
                    "title": options.title,
                    "fetch_limit": options.limit + 1,
                    "identity_bytes": 4 * (MAX_DOCUMENT_ID_LENGTH + 1),
                    "title_bytes": 4 * (MAX_TITLE_CHARACTERS + 1),
                    "source_bytes": 4 * (MAX_SOURCE_CHARACTERS + 1),
                },
            ).fetchall()
        summaries = []
        for row in rows:
            if row["invalid_record"]:
                raise DocumentCatalogError()
            identifier = _prefix(row["document_id"], encoding, MAX_DOCUMENT_ID_LENGTH)
            document_title = _prefix(row["title"], encoding, MAX_TITLE_CHARACTERS)
            document_source = _prefix(row["source"], encoding, MAX_SOURCE_CHARACTERS)
            try:
                summaries.append(
                    DocumentSummary(
                        document_id=identifier,
                        title=document_title[:MAX_TITLE_CHARACTERS],
                        title_truncated=len(document_title) > MAX_TITLE_CHARACTERS,
                        source=document_source[:MAX_SOURCE_CHARACTERS],
                        source_truncated=len(document_source) > MAX_SOURCE_CHARACTERS,
                        chunk_count=row["chunk_count"],
                    )
                )
            except ValidationError as exc:
                raise DocumentCatalogError() from exc
        page = summaries[: options.limit]
        return DocumentCatalogPage(
            documents=page,
            next_cursor=page[-1].document_id if len(summaries) > options.limit else None,
        )
