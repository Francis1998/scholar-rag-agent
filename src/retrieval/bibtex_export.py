"""Export Document / Chunk / SearchResult records to BibTeX.

Inspired by Zotero / PaperQA citation export and scholarly RAG bibliography
pipelines. Pure local metadata transform with no network calls. Distinct from
retrieval gates and from live DOI connectors; independent of model providers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from retrieval.models import Chunk, Document, SearchResult

_NON_KEY = re.compile(r"[^a-zA-Z0-9]+")
_DOI_FIELDS = ("doi", "paper_doi", "work_doi")
_AUTHOR_FIELDS = ("authors", "author")
_NAME_SEPARATORS = re.compile(r"[;|]")
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
# Brace-free control words in a balanced group, so even unmatched braces stay inside the field.
_UNMATCHED_BRACES = {"{": "{\\textbraceleft}", "}": "{\\textbraceright}"}
# LaTeX text-mode escapes; multi-letter control words use the same balanced, brace-free groups.
_LATEX_TEXT = {
    "\\": "{\\textbackslash}",
    "%": "\\%",
    "&": "\\&",
    "#": "\\#",
    "$": "\\$",
    "_": "\\_",
    "~": "{\\textasciitilde}",
    "^": "{\\textasciicircum}",
}
_ALREADY_ESCAPED = frozenset("%&#$_")
# biblatex types doi as verbatim and url as a URI; styles typeset both with \url, not as text.
_VERBATIM_FIELDS = frozenset({"doi", "url"})


def _escape(value: str, *, verbatim: bool = False) -> str:
    """Escape one field value for BibTeX's brace counting and, in text fields, for LaTeX.

    BibTeX and Biber count every brace, even after a backslash: matched pairs keep the
    ``\\{``/``\\}`` escapes, while an unmatched brace becomes a balanced LaTeX group.
    Verbatim fields otherwise only double backslashes. Text fields escape
    ``\\ % & # $ _ ~ ^`` so they typeset literally, keeping an already escaped
    ``\\%``, ``\\&``, ``\\#``, ``\\$`` or ``\\_`` as written.
    """
    unmatched: set[int] = set()
    opened: list[int] = []
    for index, character in enumerate(value):
        if character == "{":
            opened.append(index)
        elif character == "}":
            if opened:
                opened.pop()
            else:
                unmatched.add(index)
    unmatched.update(opened)
    escaped: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if index in unmatched:
            escaped.append(_UNMATCHED_BRACES[character])
        elif character in "{}":
            escaped.append("\\" + character)
        elif verbatim:
            escaped.append("\\\\" if character == "\\" else character)
        elif character == "\\" and value[index + 1 : index + 2] in _ALREADY_ESCAPED:
            escaped.append(value[index : index + 2])
            index += 1
        else:
            escaped.append(_LATEX_TEXT.get(character, character))
        index += 1
    return "".join(escaped)


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


def _initials(part: str) -> bool:
    """Return whether ``part`` is only given-name initials, such as ``J.`` or ``J.-P.``."""
    letters = part.replace(".", " ").replace("-", " ").split()
    return bool(letters) and all(len(letter) == 1 and letter.isupper() for letter in letters)


def _authors_field(metadata: dict[str, str]) -> str:
    """Join author names with BibTeX ``and``, keeping ``Family, Given`` names whole."""
    raw = _first_meta(metadata, _AUTHOR_FIELDS)
    if not raw:
        return ""
    if " and " in raw.lower():
        return raw
    if _NAME_SEPARATORS.search(raw):
        # Semicolons/pipes separate names (bioRxiv: "Doe, J.; Smith, A."); commas stay inside.
        parts = [part.strip() for part in _NAME_SEPARATORS.split(raw) if part.strip()]
    else:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) % 2 == 0 and all(_initials(given) for given in parts[1::2]):
            # Initials are never a whole name, so "Doe, J., Lee, B." is two Family, Given pairs.
            parts = [
                f"{family}, {given}" for family, given in zip(parts[::2], parts[1::2], strict=True)
            ]
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
    citation_key: str | None = None,
) -> str:
    """Format one BibTeX entry from normalized scholarly fields."""
    meta = dict(metadata)
    cite = _cite_key(document_id, title, meta) if citation_key is None else citation_key
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
    body = ",\n".join(
        f"  {key} = {{{_escape(value, verbatim=key in _VERBATIM_FIELDS)}}}" for key, value in fields
    )
    return f"@{entry_type}{{{cite},\n{body}\n}}"


def _format_collection(records: Sequence[Document | Chunk]) -> str:
    unique: dict[str, Document | Chunk] = {}
    for record in records:
        unique.setdefault(record.document_id, record)
    base_keys = [
        _cite_key(record.document_id, record.title, record.metadata) for record in unique.values()
    ]
    # Reserve natural keys before allocating suffixes for normalized collisions.
    reserved = set(base_keys)
    used: set[str] = set()
    next_suffix: dict[str, int] = {}
    entries: list[str] = []
    for record, base in zip(unique.values(), base_keys, strict=True):
        key = base
        if key in used:
            suffix = next_suffix.get(base, 2)
            key = f"{base}_{suffix}"
            while key in used or key in reserved:
                suffix += 1
                key = f"{base}_{suffix}"
            next_suffix[base] = suffix + 1
        used.add(key)
        entries.append(
            _format_entry(
                document_id=record.document_id,
                title=record.title,
                metadata=record.metadata,
                citation_key=key,
            )
        )
    return "\n\n".join(entries)


class BibTeXExporter:
    """Pure transform from Document / Chunk / SearchResult to BibTeX.

    Collections deduplicate exact ``document_id`` values and assign unique
    citation keys. First wins for documents/chunks; highest score wins for
    search results. Inputs are not mutated and no model or network is used.
    Text fields are LaTeX-escaped; ``doi`` and ``url`` stay verbatim for ``\\url``.
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
        """Export first-seen exact document identities with unique citation keys."""
        return _format_collection(documents)

    def export_chunks(self, chunks: Sequence[Chunk]) -> str:
        """Export first-seen exact chunk document identities with unique citation keys."""
        return _format_collection(chunks)

    def export_results(self, results: Sequence[SearchResult]) -> str:
        """Export ranked hits as BibTeX; highest score wins per document."""
        best: dict[str, SearchResult] = {}
        for result in results:
            key = result.chunk.document_id
            current = best.get(key)
            if current is None or result.score > current.score:
                best[key] = result
        ordered = sorted(best.values(), key=lambda item: item.score, reverse=True)
        return self.export_chunks([item.chunk for item in ordered])
