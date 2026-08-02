"""DataCite related-identifier enrichment ingestion connector.

DataCite (https://datacite.org) registers DOIs for research data, software, and
other scholarly outputs. This connector queries the public ``dois`` endpoint
and normalizes each hit while enriching documents with ``relatedIdentifiers``
from record attributes — version links, companion datasets, citations to other
DOIs, and alternate identifiers. It is distinct from ``datacite.py``, which
focuses on core bibliographic metadata without related-identifier enrichment.

Free-text queries use:

``GET https://api.datacite.org/dois?query=...``
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from ingestion.datacite import DataCiteConnector
from retrieval.models import Document

DATACITE_SEARCH_URL = "https://api.datacite.org/dois"
_PAGE_SIZE_CAP = 100
_FLOAT_YEAR_PATTERN = re.compile(r"^(\d{4})\.0+$")


class DataciteRelatedConnector:
    """Search DataCite and normalize DOI records with related-identifier enrichment."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized DataCite documents enriched with related identifiers.

        Args:
            query: Free-text DataCite query.
            max_results: Maximum number of DOI records to fetch.

        Returns:
            Normalized documents for matching DOI records. An empty list is
            returned when the query is blank, non-positive, or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "query": query.strip(),
            "page[size]": min(max_results, _PAGE_SIZE_CAP),
        }

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_results(payload, max_results)

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch the DataCite dois endpoint, returning {} on failure."""
        try:
            response = await client.get(DATACITE_SEARCH_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a DataCite ``dois`` JSON:API payload into enriched documents."""
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, list):
            return []

        documents: list[Document] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build an enriched document from one DataCite DOI resource."""
        attributes = item.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}

        title = DataCiteConnector._extract_title(attributes.get("titles")).strip()
        if not title:
            return None

        authors = DataCiteConnector._extract_creators(attributes.get("creators"))
        abstract = DataCiteConnector._extract_description(attributes.get("descriptions"))
        year = cls._extract_year(attributes.get("publicationYear"))
        doi = cls._as_str(attributes.get("doi")).strip() or cls._as_str(item.get("id")).strip()
        publisher = DataCiteConnector._extract_publisher(attributes.get("publisher"))
        resource_type = DataCiteConnector._extract_resource_type(attributes.get("types"))
        related_entries = cls._extract_related_identifiers(attributes.get("relatedIdentifiers"))
        related_summary = cls._format_related_identifiers(related_entries)
        source = cls._resolve_source(attributes, doi, title)

        text = cls._build_text(abstract, related_summary)
        if not text:
            text = DataCiteConnector._build_descriptor(authors, year, publisher)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "datacite_related",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "publisher": publisher,
                "resource_type": resource_type,
                "related_identifiers": related_summary,
                "related_identifier_count": str(len(related_entries)),
            },
        )

    @classmethod
    def _extract_related_identifiers(cls, related: object) -> list[dict[str, str]]:
        """Extract normalized related-identifier entries from attributes."""
        if not isinstance(related, list):
            return []
        entries: list[dict[str, str]] = []
        for item in related:
            if not isinstance(item, dict):
                continue
            identifier = cls._as_str(item.get("relatedIdentifier")).strip()
            if not identifier:
                continue
            entries.append(
                {
                    "relation_type": cls._as_str(item.get("relationType")).strip(),
                    "identifier_type": cls._as_str(item.get("relatedIdentifierType")).strip(),
                    "identifier": identifier,
                }
            )
        return entries

    @classmethod
    def _format_related_identifiers(cls, entries: list[dict[str, str]]) -> str:
        """Format related identifiers as a semicolon-joined summary string."""
        formatted: list[str] = []
        for entry in entries:
            relation = entry.get("relation_type", "")
            identifier_type = entry.get("identifier_type", "")
            identifier = entry.get("identifier", "")
            parts = [part for part in (relation, identifier_type, identifier) if part]
            if parts:
                formatted.append(" ".join(parts))
        return "; ".join(formatted)

    @staticmethod
    def _build_text(abstract: str, related_summary: str) -> str:
        """Compose searchable text from abstract and related identifiers."""
        parts: list[str] = []
        if abstract:
            parts.append(" ".join(abstract.split()))
        if related_summary:
            parts.append(f"Related identifiers: {related_summary}.")
        return " ".join(parts).strip()

    @staticmethod
    def _extract_year(publication_year: object) -> str:
        """Extract the publication year from DataCite ``publicationYear``."""
        if isinstance(publication_year, int) and not isinstance(publication_year, bool):
            return str(publication_year)
        if isinstance(publication_year, float) and publication_year.is_integer():
            return str(int(publication_year))
        if isinstance(publication_year, str):
            stripped = publication_year.strip()
            if stripped.isdigit():
                return stripped
            match = _FLOAT_YEAR_PATTERN.match(stripped)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def _extract_resource_type(types: object) -> str:
        """Extract ``resourceTypeGeneral`` from the DataCite ``types`` object."""
        if not isinstance(types, dict):
            return ""
        return DataciteRelatedConnector._as_str(types.get("resourceTypeGeneral")).strip()

    @classmethod
    def _resolve_source(cls, attributes: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for a DataCite record."""
        url = cls._as_str(attributes.get("url")).strip()
        if url:
            return url
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar DataCite field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
