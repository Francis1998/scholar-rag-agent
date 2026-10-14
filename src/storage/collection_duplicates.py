"""Bounded duplicate candidates from one read-only saved-collection snapshot."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.markdown import literal_block
from retrieval.duplicate_paper_cluster import DuplicatePaperClusterer, PaperRef
from retrieval.scope import MAX_DOCUMENT_ID_LENGTH, MAX_DOCUMENT_IDS
from storage.document_catalog import MAX_SOURCE_CHARACTERS, Identity
from storage.paper_collections import (
    CollectionError,
    CollectionId,
    CollectionName,
    Revision,
    SQLitePaperCollections,
)

MAX_TITLE_CHARACTERS = 512
MAX_DOI_CHARACTERS = 512
MAX_METADATA_BYTES = 16384
MAX_RESPONSE_BYTES = 262144
DuplicateFormat = Literal["json", "markdown"]
DOIField = Literal["doi", "paper_doi", "work_doi"]
DOI_FIELDS: tuple[DOIField, ...] = ("doi", "paper_doi", "work_doi")
TitleThreshold = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
_WARNINGS = (
    "Candidates are not proof of identical scientific works or scientific validity. "
    "Different DOI values or conflicting titles can still be linked by another heuristic.",
    "Groups include transitive links; not every pair in a group necessarily matches. "
    "Match reasons describe cluster-building links, not an exhaustive pairwise explanation.",
    "DOI uses the first nonblank stored string in doi, paper_doi, work_doi order. "
    "Only local normalization is applied; no DOI is resolved or externally validated. "
    "An absent, blank, or normalization-empty DOI string is an absent heuristic signal. "
    "Malformed stored metadata instead fails inspection explicitly.",
    "Blank titles never fuzzy-match. Title similarity is normalized text similarity, "
    "not semantic equivalence; candidates can be missed and unrelated papers can match.",
    "One SQLite read snapshot per inspection, not frozen document contents across requests. "
    "A collection revision protects membership, not later edits to document metadata.",
    "No records are deleted, hidden, relabeled, screened, or selected automatically. "
    "Choose exact document IDs explicitly before preview or human screening.",
    "Review before sharing: collection names, document IDs, titles, source labels, and DOI "
    "strings may be sensitive. Document bodies and unrelated metadata are not returned.",
)


def _unicode(value: str) -> str:
    value.encode("utf-8")
    return value


_Text = Annotated[str, Field(strict=True), AfterValidator(_unicode)]
_Identity = Annotated[Identity, AfterValidator(_unicode)]


class _InspectionRequest(BaseModel):
    collection_id: CollectionId
    collection_revision: Revision
    title_threshold: TitleThreshold = 0.92


class DuplicateMember(BaseModel):
    """Exact stored labels and the chosen raw DOI string, never a canonical survivor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: _Identity
    title: _Text = Field(max_length=MAX_TITLE_CHARACTERS)
    source: _Text = Field(max_length=MAX_SOURCE_CHARACTERS)
    doi: _Text = Field(max_length=MAX_DOI_CHARACTERS)
    doi_field: DOIField | None


class DuplicateGroup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster_id: str = Field(strict=True, pattern=r"^cluster-[0-9]{4}$")
    members: tuple[DuplicateMember, ...] = Field(min_length=2, max_length=MAX_DOCUMENT_IDS)
    match_reasons: tuple[
        Annotated[str, Field(pattern=r"^(doi|document_id|title:(0\.[0-9]{3}|1\.000))$")], ...
    ] = Field(min_length=1, max_length=MAX_DOCUMENT_IDS - 1)


def _bounded(content: str) -> str:
    if len(content.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise CollectionError(
            "duplicate_report_too_large",
            "Complete duplicate report exceeds the 262144-byte UTF-8 limit; no partial report.",
            413,
        )
    return content


class CollectionDuplicateReport(BaseModel):
    """Immutable complete partition of a bounded selection, not a deduplication decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    title_threshold: TitleThreshold
    inspected_count: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    document_ids: tuple[_Identity, ...] = Field(min_length=1, max_length=MAX_DOCUMENT_IDS)
    groups: tuple[DuplicateGroup, ...] = Field(max_length=MAX_DOCUMENT_IDS // 2)
    singletons: tuple[DuplicateMember, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    warnings: tuple[_Text, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def _complete_partition(self) -> Self:
        members = [member for group in self.groups for member in group.members]
        identities = [member.document_id for member in (*members, *self.singletons)]
        if (
            self.inspected_count != len(self.document_ids)
            or self.document_ids != tuple(sorted(set(self.document_ids)))
            or tuple(sorted(identities)) != self.document_ids
        ):
            raise ValueError("Every inspected identity must appear exactly once in the report.")
        return self

    def _json(self) -> str:
        return (
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    def to_json(self) -> str:
        """Return complete bounded JSON, identical to the HTTP response bytes."""
        return _bounded(self._json())

    def to_markdown(self) -> str:
        """Keep the complete report literal, including hostile links, HTML, and fences."""
        return _bounded(
            "# Collection duplicate candidates\n\n"
            "Read-only heuristic inspection, not a same-work judgment or automatic selection.\n\n"
            + literal_block(self._json(), "json")
        )


def _invalid_document() -> CollectionError:
    return CollectionError(
        "invalid_duplicate_document", "Selected document metadata is invalid.", 409
    )


def _input_too_large() -> CollectionError:
    return CollectionError(
        "duplicate_input_too_large",
        "Selected document exceeds the title/source/DOI or stored metadata limit; "
        "no fields were clipped and no partial report was produced.",
        413,
    )


def _metadata_pairs(pairs: list[tuple[str, object]]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in pairs:
        if key in metadata or not isinstance(value, str):
            raise _invalid_document()
        _unicode(key)
        metadata[key] = _unicode(value)
    return metadata


_SIZES_SQL = """
SELECT typeof(document_id) != 'text' OR typeof(title) != 'text'
           OR typeof(source) != 'text' OR typeof(metadata) != 'text' AS invalid_record,
       length(CAST(document_id AS BLOB)) AS identity_bytes,
       length(CAST(title AS BLOB)) AS title_bytes,
       length(CAST(source AS BLOB)) AS source_bytes,
       length(CAST(metadata AS BLOB)) AS metadata_bytes
FROM documents WHERE document_id COLLATE BINARY = ? LIMIT 2
"""
_DOCUMENT_SQL = """
SELECT CAST(document_id AS BLOB) AS document_id, CAST(title AS BLOB) AS title,
       CAST(source AS BLOB) AS source, CAST(metadata AS BLOB) AS metadata
FROM documents WHERE document_id COLLATE BINARY = ? LIMIT 1
"""


def _member(connection: sqlite3.Connection, encoding: str, document_id: str) -> DuplicateMember:
    sizes = connection.execute(_SIZES_SQL, (document_id,)).fetchall()
    if len(sizes) != 1 or sizes[0]["invalid_record"]:
        raise _invalid_document()
    limits = {
        "identity_bytes": 4 * MAX_DOCUMENT_ID_LENGTH,
        "title_bytes": 4 * MAX_TITLE_CHARACTERS,
        "source_bytes": 4 * MAX_SOURCE_CHARACTERS,
        "metadata_bytes": MAX_METADATA_BYTES,
    }
    if any(sizes[0][field] > limit for field, limit in limits.items()):
        raise _input_too_large()
    # The size check and full, bounded read share the caller's existing read transaction.
    row = connection.execute(_DOCUMENT_SQL, (document_id,)).fetchone()
    record = {
        field: row[field].decode(encoding)
        for field in ("document_id", "title", "source", "metadata")
    }
    if len(record["title"]) > MAX_TITLE_CHARACTERS or len(record["source"]) > MAX_SOURCE_CHARACTERS:
        raise _input_too_large()
    if record["document_id"] != document_id:
        raise _invalid_document()
    try:
        metadata = json.loads(record["metadata"], object_pairs_hook=_metadata_pairs)
    except (ValueError, RecursionError) as exc:
        raise _invalid_document() from exc
    if not isinstance(metadata, dict):
        raise _invalid_document()
    for field in DOI_FIELDS:
        if len(metadata.get(field, "")) > MAX_DOI_CHARACTERS:
            raise _input_too_large()
    chosen: DOIField | None = None
    for candidate in DOI_FIELDS:
        if metadata.get(candidate, "").strip():
            chosen = candidate
            break
    return DuplicateMember(
        document_id=record["document_id"],
        title=record["title"],
        source=record["source"],
        doi=metadata[chosen] if chosen is not None else "",
        doi_field=chosen,
    )


class SQLiteCollectionDuplicates:
    """Inspect existing storage only; construction never initializes tables or providers."""

    def __init__(self, database_path: Path | str) -> None:
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"

    def inspect(
        self,
        collection_id: str,
        *,
        collection_revision: int,
        title_threshold: float = 0.92,
    ) -> CollectionDuplicateReport:
        """Resolve and read at most 100 documents in one snapshot, then cluster without I/O."""
        try:
            options = _InspectionRequest(
                collection_id=collection_id,
                collection_revision=collection_revision,
                title_threshold=title_threshold,
            )
        except ValidationError as exc:
            raise CollectionError(
                "invalid_duplicate_request", "Invalid duplicate inspection fields or bounds.", 422
            ) from exc
        try:
            with closing(sqlite3.connect(self._database_uri, uri=True, timeout=5)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("BEGIN")
                collection = SQLitePaperCollections.resolve_in_snapshot(
                    connection, options.collection_id
                )
                if collection.revision != options.collection_revision:
                    raise CollectionError(
                        "collection_revision_conflict",
                        "Collection changed; read it again before inspecting candidates.",
                        409,
                    )
                encoding = connection.execute("PRAGMA encoding").fetchone()[0]
                members = tuple(
                    _member(connection, encoding, identity)
                    for identity in sorted(collection.document_ids)
                )
            by_id = {member.document_id: member for member in members}
            clusters = DuplicatePaperClusterer(options.title_threshold).cluster(
                [
                    PaperRef(
                        document_id=member.document_id,
                        title=member.title,
                        source=member.source,
                        doi=member.doi,
                    )
                    for member in members
                ]
            )
            report = CollectionDuplicateReport(
                collection_id=collection.collection_id,
                collection_name=collection.name,
                collection_revision=collection.revision,
                title_threshold=options.title_threshold,
                inspected_count=len(members),
                document_ids=tuple(by_id),
                groups=tuple(
                    DuplicateGroup(
                        cluster_id=cluster.cluster_id,
                        members=tuple(by_id[paper.document_id] for paper in cluster.papers),
                        match_reasons=cluster.match_reasons,
                    )
                    for cluster in clusters
                    if cluster.size > 1
                ),
                singletons=tuple(
                    by_id[cluster.papers[0].document_id]
                    for cluster in clusters
                    if cluster.size == 1
                ),
                warnings=_WARNINGS,
            )
            report.to_json()
            report.to_markdown()
            return report
        except sqlite3.Error as exc:
            raise CollectionError(
                "duplicate_storage_unavailable",
                "Collection or document storage is unavailable.",
                503,
            ) from exc
        except (ValidationError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise _invalid_document() from exc
