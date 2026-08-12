"""DataCite reports ingestion connector.

DataCite indexes research reports and similar grey-literature DOIs alongside
datasets and software. This connector searches report-like DOI records via:

``GET https://api.datacite.org/dois?query=...&resource-type-id=report``

and normalizes each hit into a :class:`Document`. It is distinct from
``datacite.py`` (general DOI registry search) and ``datacite_events.py``
(Event Data relationships).

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

DATACITE_DOIS_URL = "https://api.datacite.org/dois"
_PAGE_SIZE_CAP = 100
_FLOAT_YEAR_PATTERN = re.compile(r"^(\d{4})\.0+$")
_REPORT_RESOURCE_TYPES = frozenset(
    {
        "report",
        "reports",
        "technical report",
        "technical-report",
        "research report",
        "research-report",
    }
)


class DataCiteReportsConnector:
    """Search DataCite for report DOIs and normalize matching records."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized DataCite report documents matching a query.

        Blank queries, non-positive limits, failed requests, and malformed
        payloads yield an empty list.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        params: dict[str, str | int] = {
            "query": stripped,
            "resource-type-id": "report",
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
        """Fetch DataCite DOIs, returning an empty payload on failure."""
        try:
            response = await client.get(DATACITE_DOIS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a DataCite ``dois`` JSON:API payload into report documents."""
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
        """Build a document from one DataCite report DOI resource."""
        attributes = cls._as_dict(item.get("attributes"))
        title = cls._extract_title(attributes.get("titles")).strip()
        doi = cls._as_str(attributes.get("doi")).strip() or cls._as_str(item.get("id")).strip()
        if not title and not doi:
            return None
        if not title:
            title = f"DataCite report {doi}"

        resource_type = cls._extract_resource_type(attributes.get("types"))
        # Defense in depth: keep report-like records when the API returns mixed types.
        if resource_type and resource_type.lower() not in _REPORT_RESOURCE_TYPES:
            # Still accept when resource-type-id filter was applied and type is empty-ish
            # or uses resourceTypeGeneral Report casing variants.
            general = cls._resource_type_general(attributes.get("types")).lower()
            if general not in _REPORT_RESOURCE_TYPES and "report" not in resource_type.lower():
                return None

        authors = cls._extract_creators(attributes.get("creators"))
        abstract = cls._extract_description(attributes.get("descriptions"))
        year = cls._extract_year(attributes.get("publicationYear"))
        publisher = cls._extract_publisher(attributes.get("publisher"))
        source = cls._resolve_source(attributes, doi, title)
        text = (
            " ".join(abstract.split())
            if abstract
            else cls._build_descriptor(authors, year, publisher, doi)
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "datacite_reports",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "publisher": publisher,
                "resource_type": resource_type or "Report",
            },
        )

    @staticmethod
    def _extract_title(titles: object) -> str:
        """Extract the primary title from DataCite ``titles``."""
        if not isinstance(titles, list):
            return ""
        for entry in titles:
            if isinstance(entry, dict):
                title = DataCiteReportsConnector._as_str(entry.get("title")).strip()
                if title:
                    return title
            elif isinstance(entry, str) and entry.strip():
                return entry.strip()
        return ""

    @staticmethod
    def _extract_creators(creators: object) -> list[str]:
        """Extract ordered creator names from DataCite ``creators``."""
        if not isinstance(creators, list):
            return []
        names: list[str] = []
        for entry in creators:
            if isinstance(entry, str):
                name = entry.strip()
            elif isinstance(entry, dict):
                name = DataCiteReportsConnector._as_str(entry.get("name")).strip()
                if not name:
                    given = DataCiteReportsConnector._as_str(entry.get("givenName")).strip()
                    family = DataCiteReportsConnector._as_str(entry.get("familyName")).strip()
                    name = " ".join(part for part in (given, family) if part)
            else:
                name = ""
            if name:
                names.append(name)
        return names

    @staticmethod
    def _extract_description(descriptions: object) -> str:
        """Prefer Abstract descriptions, else the first non-empty text."""
        if not isinstance(descriptions, list):
            return ""
        fallback = ""
        for entry in descriptions:
            if not isinstance(entry, dict):
                continue
            text = DataCiteReportsConnector._as_str(entry.get("description")).strip()
            if not text:
                continue
            dtype = DataCiteReportsConnector._as_str(entry.get("descriptionType")).strip().lower()
            if dtype == "abstract":
                return text
            if not fallback:
                fallback = text
        return fallback

    @staticmethod
    def _extract_year(publication_year: object) -> str:
        """Normalize DataCite ``publicationYear`` to a four-digit string."""
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
    def _extract_publisher(publisher: object) -> str:
        """Extract publisher name from string or nested object forms."""
        if isinstance(publisher, str):
            return publisher.strip()
        if isinstance(publisher, dict):
            return DataCiteReportsConnector._as_str(publisher.get("name")).strip()
        return ""

    @classmethod
    def _extract_resource_type(cls, types: object) -> str:
        """Prefer specific resourceType, else resourceTypeGeneral."""
        type_map = cls._as_dict(types)
        specific = cls._as_str(type_map.get("resourceType")).strip()
        if specific:
            return specific
        return cls._as_str(type_map.get("resourceTypeGeneral")).strip()

    @classmethod
    def _resource_type_general(cls, types: object) -> str:
        """Return ``resourceTypeGeneral`` when present."""
        return cls._as_str(cls._as_dict(types).get("resourceTypeGeneral")).strip()

    @staticmethod
    def _resolve_source(attributes: dict[str, object], doi: str, title: str) -> str:
        """Prefer landing URL, then DOI URL, then title."""
        url = DataCiteReportsConnector._as_str(attributes.get("url")).strip()
        if url:
            return url
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _build_descriptor(
        authors: list[str],
        year: str,
        publisher: str,
        doi: str,
    ) -> str:
        """Compose searchable text when DataCite omits a description."""
        parts = ["DataCite research report."]
        if authors:
            parts.append(f"Authors: {', '.join(authors)}.")
        if publisher:
            parts.append(f"Publisher: {publisher}.")
        if year:
            parts.append(f"Year: {year}.")
        if doi:
            parts.append(f"DOI: {doi}.")
        return " ".join(parts)

    @staticmethod
    def _as_dict(value: object) -> dict[str, object]:
        """Return a dict value or an empty dict."""
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar DataCite fields to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else str(value)
        return ""
