"""Export Document / Chunk / SearchResult records to BibTeX.

Inspired by Zotero / PaperQA citation export and scholarly RAG bibliography
pipelines. Pure local metadata transform with no network calls. Distinct from
retrieval gates and from live DOI connectors. Local exporter for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 literature workflows.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from retrieval.models import Chunk, Document, SearchResult

_NON_KEY = re.compile(r"[^a-zA-Z0-9]+")
_DOI_FIELDS = ("doi", "paper_doi", "work_doi")
_AUTHOR_FIELDS = ("authors", "author")
_YEAR_FIELDS = ("year", "published_year", "publication_year")
_VENUE_FIELDS = ("journal", "venue", "container_title", "booktitle")
_TYPE_FIELDS = ("entry_type", "bibtex_type", "publication_type", "type")
_ALLOWED_TYPES = frozenset(
    {
        "article",
        "inproceedings",
        "incollection",
        "book",
        "phdthesis",
        "mastersthesis",
        "techreport",
        "misc",
        "unpublished",
    }
)


def _escape(value: str) -> str:
    """Escape BibTeX special characters in field values."""
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _first_meta(metadata: dict[str, str], fields: Sequence[str]) -> str:
    for field in fields:
        value = metadata.get(field, "").strip()
        if value:
            return value
    return ""


def _cite_key(document_id: str, title: str, metadata: dict[str, str]) -> str:
    doi = _first_meta(metadata, _DOI_FIELDS)
    raw = doi or document_id or title or "ref"
    key = _NON_KEY.sub("_", raw.strip().lower()).strip("_")
    return key or "ref"


def _entry_type(metadata: dict[str, str]) -> str:
    raw = _first_meta(metadata, _TYPE_FIELDS).lower().replace(" ", "").replace("-", "")
    aliases = {
        "conference": "inproceedings",
        "proceedings": "inproceedings",
        "preprint": "misc",
        "journalarticle": "article",
        "journal": "article",
    }
    candidate = aliases.get(raw, raw)
    if candidate in _ALLOWED_TYPES:
        return candidate
    return "article" if _first_meta(metadata, _VENUE_FIELDS) else "misc"


def _authors_field(metadata: dict[str, str]) -> str:
    raw = _first_meta(metadata, _AUTHOR_FIELDS)
    if not raw:
        return ""
    if " and " in raw.lower():
        return raw
    parts = [part.strip() for part in re.split(r"[,;|]", raw) if part.strip()]
    if len(parts) <= 1:
        return raw
    return " and ".join(parts)


def _year_field(metadata: dict[str, str]) -> str:
    year = _first_meta(metadata, _YEAR_FIELDS)
    if year:
        match = re.search(r"(19|20)\d{2}", year)
        return match.group(0) if match else year
    for field in ("published_at", "date", "publication_date"):
        value = metadata.get(field, "").strip()
        if not value:
            continue
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return match.group(0)
    return ""


def _format_entry(
    *,
    document_id: str,
    title: str,
    metadata: dict[str, str],
) -> str:
    """Format one BibTeX entry from normalized scholarly fields."""
    meta = dict(metadata)
    cite = _cite_key(document_id, title, meta)
    entry_type = _entry_type(meta)
    fields: list[tuple[str, str]] = [("title", title.strip() or "Untitled")]
    authors = _authors_field(meta)
    if authors:
        fields.append(("author", authors))
    year = _year_field(meta)
    if year:
        fields.append(("year", year))
    venue = _first_meta(meta, _VENUE_FIELDS)
    if venue:
        venue_key = "booktitle" if entry_type == "inproceedings" else "journal"
        fields.append((venue_key, venue))
    doi = _first_meta(meta, _DOI_FIELDS)
    if doi:
        fields.append(("doi", doi))
    url = meta.get("url", "").strip() or meta.get("landing_url", "").strip()
    if url:
        fields.append(("url", url))
    body = ",\n".join(f"  {key} = {{{_escape(value)}}}" for key, value in fields)
    return f"@{entry_type}{{{cite},\n{body}\n}}"


class BibTeXExporter:
    """Pure transform from Document / Chunk / SearchResult to BibTeX.

    Deduplicates by ``document_id`` when exporting collections (first wins for
    documents/chunks; highest score wins for search results). Inputs are not
    mutated. Local exporter for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines (not a DOI connector).
    """

    def export_document(self, document: Document) -> str:
        """Return a single BibTeX entry for ``document``."""
        return _format_entry(
            document_id=document.document_id,
            title=document.title,
            metadata=document.metadata,
        )

    def export_chunk(self, chunk: Chunk) -> str:
        """Return a single BibTeX entry for ``chunk`` provenance."""
        return _format_entry(
            document_id=chunk.document_id,
            title=chunk.title,
            metadata=chunk.metadata,
        )

    def export_result(self, result: SearchResult) -> str:
        """Return a single BibTeX entry for a scored retrieval hit."""
        return self.export_chunk(result.chunk)

    def export_documents(self, documents: Sequence[Document]) -> str:
        """Export documents as BibTeX, deduplicated by ``document_id``."""
        seen: set[str] = set()
        entries: list[str] = []
        for document in documents:
            key = document.document_id.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            entries.append(self.export_document(document))
        return "\n\n".join(entries)

    def export_chunks(self, chunks: Sequence[Chunk]) -> str:
        """Export chunks as BibTeX, deduplicated by ``document_id``."""
        seen: set[str] = set()
        entries: list[str] = []
        for chunk in chunks:
            key = chunk.document_id.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            entries.append(self.export_chunk(chunk))
        return "\n\n".join(entries)

    def export_results(self, results: Sequence[SearchResult]) -> str:
        """Export ranked hits as BibTeX; highest score wins per document."""
        best: dict[str, SearchResult] = {}
        for result in results:
            key = result.chunk.document_id.strip().lower()
            current = best.get(key)
            if current is None or result.score > current.score:
                best[key] = result
        ordered = sorted(best.values(), key=lambda item: item.score, reverse=True)
        return "\n\n".join(self.export_result(item) for item in ordered)
