"""Zenodo open-science repository ingestion connector.

Zenodo (https://zenodo.org), operated by CERN, is a large general-purpose
open-science repository that mints DOIs for publications, datasets, software,
and other research outputs across every discipline. It complements the existing
sources (arXiv, Semantic Scholar, OpenAlex, PubMed, Crossref, Europe PMC, DOAJ,
DBLP, HAL, OpenAIRE) with strong coverage of long-tail and self-deposited
outputs that never reach a traditional publisher.

Its public REST search endpoint takes a free-text ``q`` query (Elasticsearch
query-string syntax) and returns hits under ``hits.hits``, each carrying a
``metadata`` object with ``title``, an ordered ``creators`` list (each with a
``name``), an HTML ``description``, a ``publication_date``, and a ``doi``, plus a
top-level ``doi`` and ``links`` (the ``html`` landing page). The HTML
description is reduced to plain text; when a record has no description a concise
author/year descriptor is synthesised. One request can ingest several outputs
for a topic, and the endpoint is unauthenticated (anonymous requests are capped
at a ``size`` of 25).
"""

import html
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

ZENODO_BASE_URL = "https://zenodo.org/api/records"
_ANONYMOUS_PAGE_SIZE_CAP = 25
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class ZenodoConnector:
    """Search Zenodo and normalize matching records into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Zenodo documents matching a query.

        Args:
            query: Free-text Zenodo query.
            max_results: Maximum number of records to fetch (anonymous Zenodo
                requests cap ``size`` at 25).

        Returns:
            Normalized documents for the matching records. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "q": query.strip(),
            "size": min(max_results, _ANONYMOUS_PAGE_SIZE_CAP),
            "page": 1,
            "sort": "bestmatch",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(ZENODO_BASE_URL, params=params)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Zenodo ``records`` JSON payload into documents.

        Args:
            payload: Decoded Zenodo response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each record in ``hits.hits``.
        """
        if not isinstance(payload, dict):
            return []
        hits = payload.get("hits")
        if not isinstance(hits, dict):
            return []
        records = hits.get("hits")
        if not isinstance(records, list):
            return []

        documents: list[Document] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, record: dict[str, object]) -> Document | None:
        """Build a document from one Zenodo record.

        Args:
            record: A single record from ``hits.hits``.

        Returns:
            Normalized document, or None when the record carries no usable title.
        """
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        title = cls._as_str(metadata.get("title")).strip()
        if not title:
            return None
        authors = cls._extract_authors(metadata.get("creators"))
        abstract = cls._strip_html(metadata.get("description"))
        year = cls._extract_year(metadata.get("publication_date"))
        doi = cls._extract_doi(record, metadata)
        source = cls._resolve_source(record, doi, title)
        text = abstract if abstract else cls._build_descriptor(authors, year)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "zenodo",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
            },
        )

    @staticmethod
    def _extract_year(publication_date: object) -> str:
        """Extract a four-digit year from a Zenodo ``publication_date``.

        Zenodo dates are usually ISO calendar dates such as ``2025-02-10``, but
        some records carry free-text placeholders (``unpublished``, ``TBA``).
        Taking ``[:4]`` unconditionally leaked those placeholders into
        ``metadata['year']``. Only values matching ``^\\d{4}`` are accepted.

        Args:
            publication_date: The raw ``publication_date`` field.

        Returns:
            The four-digit year string, or an empty string when absent/invalid.
        """
        if not isinstance(publication_date, str):
            return ""
        match = _YEAR_PREFIX_PATTERN.match(publication_date.strip())
        return match.group(1) if match else ""

    @staticmethod
    def _extract_authors(creators: object) -> list[str]:
        """Extract ordered creator names from a Zenodo ``creators`` list.

        Each creator is an object carrying a ``name``; bare string entries are
        also tolerated.

        Args:
            creators: The ``creators`` value.

        Returns:
            Ordered creator names, empty when none are present.
        """
        if not isinstance(creators, list):
            return []
        names: list[str] = []
        for entry in creators:
            if isinstance(entry, str):
                name = entry.strip()
            elif isinstance(entry, dict):
                name = ZenodoConnector._as_str(entry.get("name")).strip()
            else:
                name = ""
            if name:
                names.append(name)
        return names

    @staticmethod
    def _extract_doi(record: dict[str, object], metadata: dict[str, object]) -> str:
        """Extract the DOI from a Zenodo record.

        Zenodo exposes the DOI both at the record top level and under
        ``metadata.doi``; the top-level value is preferred, falling back to the
        metadata copy.

        Args:
            record: The record object.
            metadata: The record's ``metadata`` object.

        Returns:
            The DOI string when present, else an empty string.
        """
        doi = ZenodoConnector._as_str(record.get("doi")).strip()
        if doi:
            return doi
        return ZenodoConnector._as_str(metadata.get("doi")).strip()

    @classmethod
    def _resolve_source(cls, record: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for a Zenodo record.

        The ``links.html`` landing page is preferred, then ``links.self``, then a
        DOI link, and finally the title as an anchor of last resort.

        Args:
            record: The record object.
            doi: The normalized DOI, if any.
            title: The title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        links = record.get("links")
        if isinstance(links, dict):
            for key in ("html", "self"):
                url = cls._as_str(links.get(key)).strip()
                if url:
                    return url
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _build_descriptor(authors: list[str], year: str) -> str:
        """Compose a citation-style descriptor used when no description exists.

        Args:
            authors: Ordered creator names.
            year: Publication year, if any.

        Returns:
            A single-line descriptor, or an empty string when nothing is known.
        """
        parts: list[str] = []
        if authors:
            parts.append("By " + ", ".join(authors))
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @classmethod
    def _strip_html(cls, description: object) -> str:
        """Reduce an HTML Zenodo description to collapsed plain text.

        Zenodo stores descriptions as HTML (for example ``<p>...</p>``) with
        entity-encoded characters. Tags are removed, entities are decoded, and
        surrounding whitespace is collapsed so the stored text is readable prose
        rather than leaking raw markup.

        Args:
            description: The raw ``description`` field.

        Returns:
            Plain-text description, or an empty string when unavailable.
        """
        if not isinstance(description, str) or not description.strip():
            return ""
        without_tags = _HTML_TAG_PATTERN.sub(" ", description)
        return " ".join(html.unescape(without_tags).split())

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar Zenodo field value to a string.

        Args:
            value: A raw scalar Zenodo field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
