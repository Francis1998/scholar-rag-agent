"""Crossref works relation enrichment ingestion connector.

Crossref (https://www.crossref.org) indexes scholarly works with optional
``relation`` metadata — typed links such as ``is-referenced-by``, ``has-review``,
``is-preprint-of``, and similar assertions registered against a DOI. This
connector queries the public works API and normalizes each hit while enriching
documents with those relation type keys for searchable provenance alongside
Crossref works, members, Event Data, and DataCite related identifiers.

Free-text queries use:

``GET https://api.crossref.org/works?query=...&rows={n}``

DOI-shaped queries (``10.1234/example``, ``doi:10.1234/...``, or DOI URLs)
resolve a single work via:

``GET https://api.crossref.org/works/{doi}``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from ingestion.crossref import CrossrefConnector
from retrieval.models import Document

CROSSREF_WORKS_URL = "https://api.crossref.org/works"

_DOI_PATTERN = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\s,;<>\"']+)",
    re.IGNORECASE,
)
_TRAILING_DOI_CHARS = ".,;:)]}"


class CrossrefRelationsConnector:
    """Search Crossref works and normalize them with relation-type enrichment."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional contact email added to requests so Crossref routes
                traffic to its faster, polite API pool. When omitted,
                ``CROSSREF_MAILTO`` (then ``OPENALEX_MAILTO``) is read from the
                environment when present.
        """
        self._mailto = (
            mailto
            or os.environ.get("CROSSREF_MAILTO", "").strip()
            or os.environ.get("OPENALEX_MAILTO", "").strip()
            or None
        )

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Crossref works enriched with relation metadata.

        Args:
            query: Free-text bibliographic search or a DOI identifier.
            max_results: Maximum number of works to return for free-text
                search (ignored for single-DOI lookups beyond returning one).

        Returns:
            Normalized work documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        doi = self._extract_doi(stripped)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            if doi:
                payload = await self._fetch_work(client, doi)
                return self._parse_single(payload)

            params: dict[str, str | int] = {"query": stripped, "rows": max_results}
            if self._mailto:
                params["mailto"] = self._mailto
            payload = await self._fetch_search(client, params)
        return self._parse_results(payload, max_results)

    async def _fetch_search(
        self,
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch a Crossref works search payload, returning {} on API failure."""
        try:
            response = await client.get(CROSSREF_WORKS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    async def _fetch_work(self, client: httpx.AsyncClient, doi: str) -> object:
        """Fetch one work by DOI, returning {} on failure."""
        params: dict[str, str] = {}
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            response = await client.get(f"{CROSSREF_WORKS_URL}/{doi}", params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Crossref ``work-list`` JSON payload into documents."""
        if not isinstance(payload, dict):
            return []
        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        items = message.get("items")
        if not isinstance(items, list):
            return []

        documents: list[Document] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _parse_single(cls, payload: object) -> list[Document]:
        """Parse a single Crossref ``work`` JSON payload into documents."""
        if not isinstance(payload, dict):
            return []
        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        document = cls._build_document(message)
        return [document] if document is not None else []

    @classmethod
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build a relation-enriched document from one Crossref work object."""
        title = CrossrefConnector._first_string(item.get("title")).strip()
        if not title:
            return None

        doi = cls._as_str(item.get("DOI")).strip()
        source = f"https://doi.org/{doi}" if doi else title
        abstract = CrossrefConnector._strip_jats(item.get("abstract"))
        year = CrossrefConnector._resolve_year(item)
        relation_types, relation_count = cls._extract_relations(item.get("relation"))
        relation_summary = ", ".join(relation_types)
        text = cls._build_text(abstract, title, year, relation_summary, relation_count)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "crossref_relations",
                "doi": doi,
                "year": year,
                "relation_types": relation_summary,
                "relation_count": str(relation_count),
            },
        )

    @classmethod
    def _extract_relations(cls, relation: object) -> tuple[list[str], int]:
        """Extract ordered relation type keys and total related-object count."""
        if not isinstance(relation, dict):
            return [], 0

        types: list[str] = []
        count = 0
        for key, value in relation.items():
            relation_type = cls._as_str(key).strip()
            if not relation_type:
                continue
            if relation_type not in types:
                types.append(relation_type)
            if isinstance(value, list):
                count += len(value)
            elif value is not None:
                count += 1
        return types, count

    @staticmethod
    def _build_text(
        abstract: str,
        title: str,
        year: str,
        relation_summary: str,
        relation_count: int,
    ) -> str:
        """Compose searchable text from abstract/title and relation enrichment."""
        parts: list[str] = []
        if abstract:
            parts.append(abstract)
        else:
            descriptor = title
            if year:
                descriptor = f"{title} ({year})"
            parts.append(descriptor)
        if relation_summary:
            parts.append(f"Relations: {relation_summary}.")
            if relation_count:
                parts.append(f"Related objects: {relation_count}.")
        return " ".join(parts).strip()

    @classmethod
    def _extract_doi(cls, query: str) -> str:
        """Extract a DOI from a query when present."""
        match = _DOI_PATTERN.search(query.strip())
        if not match:
            return ""
        return match.group(1).strip().rstrip(_TRAILING_DOI_CHARS)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar Crossref field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
