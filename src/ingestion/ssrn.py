"""SSRN preprint DOI bridge ingestion connector.

Social Science Research Network (SSRN) preprints are indexed in Crossref under the
``10.2139`` DOI prefix. This connector searches SSRN abstracts via Crossref
works filtered to that prefix and also resolves direct SSRN-shaped DOIs, then
normalizes each hit into a :class:`Document` with stable provenance.

Free-text queries use:

``GET https://api.crossref.org/works?query=...&filter=prefix:10.2139&rows={n}``

DOI-shaped queries (``10.2139/ssrn.3537853``, ``doi:10.2139/...``, or DOI URLs)
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
_SSRN_DOI_PREFIX = "10.2139/"
_SSRN_DOI_PATTERN = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.2139/\S+)",
    re.IGNORECASE,
)


class SsrnConnector:
    """Search SSRN preprints via Crossref and normalize works into documents."""

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
        """Return normalized SSRN preprint documents matching a query.

        Args:
            query: Free-text bibliographic search or an SSRN DOI identifier.
            max_results: Maximum number of works to return for free-text
                search (ignored for single-DOI lookups beyond returning one).

        Returns:
            Normalized SSRN documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        doi = self._extract_ssrn_doi(stripped)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            if doi is not None:
                payload = await self._fetch_work(client, doi)
                return self._parse_single(payload)

            params: dict[str, str | int] = {
                "query": stripped,
                "filter": "prefix:10.2139",
                "rows": max_results,
            }
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
        """Fetch one SSRN work by DOI, returning {} on failure."""
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
        """Build a document from one Crossref work object."""
        title = cls._first_string(item.get("title"))
        if not title:
            return None

        doi = cls._as_str(item.get("DOI")).strip().lower()
        if doi and not doi.startswith(_SSRN_DOI_PREFIX):
            return None

        source = f"https://doi.org/{doi}" if doi else cls._ssrn_landing_url(item)
        if not source:
            return None

        abstract = CrossrefConnector._strip_jats(item.get("abstract"))
        authors = cls._extract_authors(item.get("author"))
        year = CrossrefConnector._resolve_year(item)
        container = cls._first_string(item.get("container-title"))
        text = abstract or cls._build_descriptor(title, authors, year, container)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "ssrn",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors[:12]),
                "container": container,
                "ssrn_url": cls._ssrn_landing_url(item),
            },
        )

    @classmethod
    def _extract_ssrn_doi(cls, query: str) -> str | None:
        """Extract an SSRN DOI from a query when present."""
        doi_match = _SSRN_DOI_PATTERN.search(query.strip())
        if doi_match:
            return doi_match.group(1).rstrip(".,;)")
        if query.strip().lower().startswith(_SSRN_DOI_PREFIX):
            return query.strip().lower()
        return None

    @staticmethod
    def _ssrn_landing_url(item: dict[str, object]) -> str:
        """Return the SSRN abstract landing URL when present."""
        resource = item.get("resource")
        if isinstance(resource, dict):
            primary = resource.get("primary")
            if isinstance(primary, dict):
                url = primary.get("URL")
                if isinstance(url, str) and url.strip():
                    return url.strip()
        url_field = item.get("URL")
        if isinstance(url_field, str) and url_field.strip():
            return url_field.strip()
        return ""

    @staticmethod
    def _first_string(value: object) -> str:
        """Return the first non-empty string in a Crossref list field."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str) and entry.strip():
                    return entry
        return ""

    @staticmethod
    def _extract_authors(value: object) -> list[str]:
        """Extract author display names from a Crossref author list."""
        if not isinstance(value, list):
            return []
        authors: list[str] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            given = SsrnConnector._as_str(entry.get("given")).strip()
            family = SsrnConnector._as_str(entry.get("family")).strip()
            name = " ".join(part for part in (given, family) if part)
            if name:
                authors.append(name)
        return authors

    @staticmethod
    def _build_descriptor(title: str, authors: list[str], year: str, container: str) -> str:
        """Compose searchable descriptor text when no abstract is available."""
        parts = [f"SSRN preprint: {title}"]
        if authors:
            parts.append("Authors: " + ", ".join(authors[:8]))
        if year:
            parts.append(f"Year: {year}")
        if container:
            parts.append(f"Container: {container}")
        return ". ".join(parts) + "."

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar Crossref field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
