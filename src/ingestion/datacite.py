"""DataCite DOI registry ingestion connector.

DataCite (https://datacite.org) is the leading DOI registration agency for
research data, software, and other non-traditional scholarly outputs. It
complements the existing sources (arXiv, Semantic Scholar, OpenAlex, PubMed,
Crossref, Europe PMC, DOAJ, DBLP, HAL, OpenAIRE, Zenodo, Figshare, CORE) with
authoritative DOI metadata for datasets and research objects that Crossref
(journals/proceedings) does not cover as deeply.

Its public REST endpoint ``GET https://api.datacite.org/dois`` takes a free-text
``query`` parameter and returns JSON:API ``data`` items, each with
``attributes`` carrying ``doi``, ``titles``, ``creators``, ``descriptions``,
``publicationYear``, ``publisher``, ``url``, and ``types.resourceTypeGeneral``.
One request can ingest several DOI records for a topic. The endpoint is
unauthenticated.
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

DATACITE_SEARCH_URL = "https://api.datacite.org/dois"
_PAGE_SIZE_CAP = 100
_FLOAT_YEAR_PATTERN = re.compile(r"^(\d{4})\.0+$")


class DataCiteConnector:
    """Search DataCite and normalize matching DOI records into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized DataCite documents matching a query.

        Args:
            query: Free-text DataCite query.
            max_results: Maximum number of DOI records to fetch (``page[size]``
                is capped at 100).

        Returns:
            Normalized documents for the matching DOI records. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "query": query.strip(),
            "page[size]": min(max_results, _PAGE_SIZE_CAP),
        }

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(DATACITE_SEARCH_URL, params=params)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a DataCite ``dois`` JSON:API payload into documents.

        Args:
            payload: Decoded DataCite response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each item in ``data``.
        """
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
        """Build a document from one DataCite DOI resource.

        Args:
            item: A single JSON:API resource from ``data``.

        Returns:
            Normalized document, or None when the record carries no usable title.
        """
        attributes = item.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        title = cls._extract_title(attributes.get("titles")).strip()
        if not title:
            return None
        authors = cls._extract_creators(attributes.get("creators"))
        abstract = cls._extract_description(attributes.get("descriptions"))
        year = cls._extract_year(attributes.get("publicationYear"))
        doi = cls._as_str(attributes.get("doi")).strip() or cls._as_str(item.get("id")).strip()
        publisher = cls._extract_publisher(attributes.get("publisher"))
        resource_type = cls._extract_resource_type(attributes.get("types"))
        source = cls._resolve_source(attributes, doi, title)
        if abstract:
            text = " ".join(abstract.split())
        else:
            text = cls._build_descriptor(authors, year, publisher)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "datacite",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "publisher": publisher,
                "resource_type": resource_type,
            },
        )

    @staticmethod
    def _extract_title(titles: object) -> str:
        """Extract the primary title from a DataCite ``titles`` list.

        Args:
            titles: The ``attributes.titles`` value.

        Returns:
            The first non-empty title string, or empty when none exist.
        """
        if not isinstance(titles, list):
            return ""
        for entry in titles:
            if isinstance(entry, dict):
                title = DataCiteConnector._as_str(entry.get("title")).strip()
                if title:
                    return title
            elif isinstance(entry, str) and entry.strip():
                return entry.strip()
        return ""

    @staticmethod
    def _extract_creators(creators: object) -> list[str]:
        """Extract ordered creator names from a DataCite ``creators`` list.

        Args:
            creators: The ``attributes.creators`` value.

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
                name = DataCiteConnector._as_str(entry.get("name")).strip()
                if not name:
                    given = DataCiteConnector._as_str(entry.get("givenName")).strip()
                    family = DataCiteConnector._as_str(entry.get("familyName")).strip()
                    name = " ".join(part for part in (given, family) if part)
            else:
                name = ""
            if name:
                names.append(name)
        return names

    @staticmethod
    def _extract_description(descriptions: object) -> str:
        """Extract preferred abstract-like text from ``descriptions``.

        Prefer ``descriptionType == \"Abstract\"`` when present; otherwise take
        the first non-empty description.

        Args:
            descriptions: The ``attributes.descriptions`` value.

        Returns:
            Collapsed description text, or empty when unavailable.
        """
        if not isinstance(descriptions, list):
            return ""
        fallback = ""
        for entry in descriptions:
            if not isinstance(entry, dict):
                continue
            text = DataCiteConnector._as_str(entry.get("description")).strip()
            if not text:
                continue
            dtype = DataCiteConnector._as_str(entry.get("descriptionType")).strip().lower()
            if dtype == "abstract":
                return text
            if not fallback:
                fallback = text
        return fallback

    @staticmethod
    def _extract_year(publication_year: object) -> str:
        """Extract the publication year from DataCite ``publicationYear``.

        Args:
            publication_year: The raw ``publicationYear`` field.

        Returns:
            The year as a string, or an empty string when absent/invalid.
        """
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
        """Extract a publisher name from a string or object field.

        Newer DataCite responses may return ``publisher`` as
        ``{\"name\": \"...\"}`` rather than a bare string.

        Args:
            publisher: The ``attributes.publisher`` value.

        Returns:
            Publisher name, or empty when absent.
        """
        if isinstance(publisher, dict):
            return DataCiteConnector._as_str(publisher.get("name")).strip()
        return DataCiteConnector._as_str(publisher).strip()

    @staticmethod
    def _extract_resource_type(types: object) -> str:
        """Extract ``resourceTypeGeneral`` from the DataCite ``types`` object.

        Args:
            types: The ``attributes.types`` value.

        Returns:
            The general resource type string, or empty when absent.
        """
        if not isinstance(types, dict):
            return ""
        return DataCiteConnector._as_str(types.get("resourceTypeGeneral")).strip()

    @classmethod
    def _resolve_source(cls, attributes: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for a DataCite record.

        ``attributes.url`` is preferred, then a DOI link, and finally the title
        as an anchor of last resort.

        Args:
            attributes: The DOI ``attributes`` object.
            doi: The normalized DOI, if any.
            title: The title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        url = cls._as_str(attributes.get("url")).strip()
        if url:
            return url
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _build_descriptor(authors: list[str], year: str, publisher: str) -> str:
        """Compose a citation-style descriptor used when no description exists.

        Args:
            authors: Ordered creator names.
            year: Publication year, if any.
            publisher: Publisher name, if any.

        Returns:
            A single-line descriptor, or an empty string when nothing is known.
        """
        parts: list[str] = []
        if authors:
            parts.append("By " + ", ".join(authors))
        if publisher:
            parts.append(f"via {publisher}")
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar DataCite field value to a string.

        Args:
            value: A raw scalar DataCite field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
