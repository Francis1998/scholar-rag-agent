"""Cited-only bibliography downloads from bounded, completed frozen evidence."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent.evidence import SHA256, CaptureLimits, EvidenceBundle, EvidenceSource
from retrieval.bibtex_export import BibTeXExporter
from retrieval.models import Chunk
from storage.evidence_export import BoundedRunEvents, EvidenceExporter, EvidenceExportError

MAX_SOURCES = CaptureLimits().max_sources
MAX_RESPONSE_BYTES = 262144
BibliographyIdentity = Annotated[str, Field(strict=True, min_length=1, max_length=256)]
CitationNumber = Annotated[int, Field(strict=True, ge=1, le=50)]
BibliographyFormat = Literal["bibtex", "json"]
_IDENTITY = TypeAdapter(BibliographyIdentity)
BIBLIOGRAPHIC_FIELDS = (
    "doi",
    "paper_doi",
    "work_doi",
    "authors",
    "author",
    "year",
    "published_year",
    "publication_year",
    "published_at",
    "date",
    "publication_date",
    "journal",
    "venue",
    "container_title",
    "booktitle",
    "entry_type",
    "bibtex_type",
    "publication_type",
    "type",
    "url",
    "landing_url",
)
_WARNINGS = (
    "Uses saved final-answer citation records, not prose citation parsing. "
    "Citation grounding checks token overlap, not truth or entailment.",
    "Only captured bibliographic metadata is used, without verification or enrichment. "
    "Author, year, and entry-type formatting is heuristic; review before importing.",
    "Review before sharing: identifiers, titles, and metadata may be sensitive. "
    "BibTeX is data, not a safe executable LaTeX document.",
)


class BibliographyError(EvidenceExportError):
    """A stable, privacy-preserving bibliography failure."""


def _invalid_record() -> BibliographyError:
    return BibliographyError(
        "invalid_run_record", "Saved evidence is inconsistent, invalid, or an unsupported version."
    )


def _bounded(content: str) -> str:
    if len(content.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise BibliographyError(
            "bibliography_too_large",
            "Bibliography exceeds the 262144-byte UTF-8 download limit; no partial export.",
            413,
        )
    return content


class CitedChunk(BaseModel):
    """Map a frozen chunk to every corresponding final-answer citation position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: BibliographyIdentity
    evidence_rank: int = Field(ge=1, le=50)
    citation_numbers: list[CitationNumber] = Field(min_length=1, max_length=50)


class BibliographySource(BaseModel):
    """One exact document, with metadata from its first cited frozen evidence rank."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: BibliographyIdentity
    first_evidence_rank: int = Field(ge=1, le=50)
    title: str
    metadata: dict[str, str]
    cited_chunks: list[CitedChunk] = Field(min_length=1, max_length=50)


class SavedBibliography(BaseModel):
    """Versioned bibliography provenance, not a new evidence bundle or citation standard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    evidence_schema_version: Literal["1.0"] = "1.0"
    run_id: BibliographyIdentity
    context_sha256: SHA256
    sources: list[BibliographySource] = Field(max_length=50)
    bibtex: str
    warnings: list[str] = Field(max_length=104)

    def to_json(self) -> str:
        """Return the exact, bounded UTF-8 JSON representation used by the API."""
        return _bounded(
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    def to_bibtex(self) -> str:
        """Return the bounded BibTeX text; no prologue or fabricated empty entry."""
        return _bounded(self.bibtex)


def _metadata(chunk: Chunk) -> dict[str, str]:
    return {key: chunk.metadata[key] for key in BIBLIOGRAPHIC_FIELDS if key in chunk.metadata}


def _project(bundle: EvidenceBundle) -> SavedBibliography:
    citations = bundle.answer.citations
    if len(citations) > MAX_SOURCES:
        raise BibliographyError(
            "bibliography_limit_exceeded", "Saved answer exceeds 50 citation records."
        )
    source_by_id = {source.chunk.chunk_id: source for source in bundle.snapshot.sources}
    positions: dict[str, list[int]] = {}
    for number, citation in enumerate(citations, 1):
        source = source_by_id.get(citation.chunk_id)
        if source is None or citation.document_id != source.chunk.document_id:
            raise BibliographyError(
                "invalid_bibliography_citation",
                "A final-answer citation is missing from, or disagrees with, the frozen snapshot.",
            )
        positions.setdefault(citation.chunk_id, []).append(number)

    grouped: dict[str, list[EvidenceSource]] = {}
    for source in bundle.snapshot.sources:
        if source.chunk.chunk_id in positions:
            grouped.setdefault(source.chunk.document_id, []).append(source)
    sources = []
    chosen_chunks = []
    warnings = list(_WARNINGS)
    for document_id, cited in grouped.items():
        first = cited[0]
        metadata = _metadata(first.chunk)
        if any(
            source.chunk.title != first.chunk.title or _metadata(source.chunk) != metadata
            for source in cited[1:]
        ):
            warnings.append(
                f"Conflicting captured title or bibliographic metadata for the document at "
                f"evidence rank {first.rank}; using only that first cited rank, "
                "without merging fields from other cited chunks."
            )
        if not first.chunk.title.strip():
            warnings.append(
                f"Evidence rank {first.rank} has a blank captured title; the formatter emits "
                "'Untitled' as a display placeholder, not a recovered title."
            )
        sources.append(
            BibliographySource(
                document_id=document_id,
                first_evidence_rank=first.rank,
                title=first.chunk.title,
                metadata=metadata,
                cited_chunks=[
                    CitedChunk(
                        chunk_id=source.chunk.chunk_id,
                        evidence_rank=source.rank,
                        citation_numbers=positions[source.chunk.chunk_id],
                    )
                    for source in cited
                ],
            )
        )
        chosen_chunks.append(first.chunk.model_copy(update={"metadata": metadata}))
    if not citations:
        warnings.append(
            "No final-answer citations were saved; the bibliography is explicitly empty, "
            "even if the frozen context contains retrieved sources."
        )
    return SavedBibliography(
        evidence_schema_version=bundle.schema_version,
        run_id=bundle.run_id,
        context_sha256=bundle.snapshot.context_sha256,
        sources=sources,
        bibtex=BibTeXExporter().export_chunks(chosen_chunks),
        warnings=warnings,
    )


class SQLiteSavedBibliography:
    """Read an existing event database; never initialize stores or consult a current corpus."""

    def __init__(self, database_path: Path | str) -> None:
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"

    def export(self, run_id: str) -> SavedBibliography:
        """Validate completed frozen evidence and require both downloads to fit exactly."""
        try:
            _IDENTITY.validate_python(run_id)
            run_id.encode("utf-8")
        except (ValidationError, UnicodeError) as exc:
            raise BibliographyError(
                "invalid_bibliography_request",
                "The run ID must contain 1 to 256 valid characters; format must be bibtex or json.",
                422,
            ) from exc
        try:
            with closing(sqlite3.connect(self._database_uri, uri=True, timeout=5)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("BEGIN")
                encoding: str = connection.execute("PRAGMA encoding").fetchone()[0]
                bundle = EvidenceExporter(BoundedRunEvents(connection, encoding)).export(run_id)
            result = _project(bundle)
            result.to_bibtex()
            result.to_json()
            return result
        except sqlite3.Error as exc:
            raise BibliographyError(
                "bibliography_storage_unavailable", "Saved evidence storage is unavailable.", 503
            ) from exc
        except (ValidationError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise _invalid_record() from exc
