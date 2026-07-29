"""Dryad research data repository ingestion connector.

Dryad (https://datadryad.org) is a curated open-access repository for research
data associated with scholarly publications. It complements Zenodo, Figshare,
DataCite, and OSF with strong coverage of publisher-linked datasets across
life sciences, ecology, and related disciplines.

Its public REST search endpoint accepts a free-text ``q`` query and returns
datasets under ``_embedded['stash:datasets']``, each carrying ``title``,
``authors``, an HTML ``abstract``, ``publicationDate``, ``identifier`` (a
``doi:`` URI), and ``sharingLink``. The HTML abstract is reduced to plain text
when present. One request can ingest several datasets for a topic, and the
endpoint is unauthenticated.
"""

from __future__ import annotations

import html
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

DRYAD_SEARCH_URL = "https://datadryad.org/api/v2/search"
_PAGE_SIZE_CAP = 100
_DATASETS_KEY = "stash:datasets"
_DOI_PREFIX = "doi:"
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class DryadConnector:
    """Search Dryad and normalize matching datasets into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Dryad documents matching a query.

        Args:
            query: Free-text Dryad query.
            max_results: Maximum number of datasets to fetch (``per_page`` is
                capped at 100).

        Returns:
            Normalized documents for the matching datasets. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "q": query.strip(),
            "per_page": min(max_results, _PAGE_SIZE_CAP),
        }

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(DRYAD_SEARCH_URL, params=params)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Dryad search JSON payload into documents.

        Args:
            payload: Decoded Dryad response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each dataset in the embedded list.
        """
        datasets = cls._extract_datasets(payload)
        documents: list[Document] = []
        for item in datasets:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _extract_datasets(cls, payload: object) -> list[object]:
        """Return dataset objects from a Dryad search payload."""
        if not isinstance(payload, dict):
            return []
        embedded = payload.get("_embedded")
        if not isinstance(embedded, dict):
            return []
        datasets = embedded.get(_DATASETS_KEY)
        if isinstance(datasets, list):
            return datasets
        return []

    @classmethod
    def _build_document(cls, dataset: dict[str, object]) -> Document | None:
        """Build a document from one Dryad dataset record.

        Args:
            dataset: A single dataset object from the search response.

        Returns:
            Normalized document, or None when the dataset carries no usable title.
        """
        title = cls._as_str(dataset.get("title")).strip()
        if not title:
            return None
        doi = cls._normalize_doi(cls._as_str(dataset.get("identifier")).strip())
        year = cls._extract_year(dataset.get("publicationDate"))
        authors = cls._extract_authors(dataset.get("authors"))
        abstract = cls._strip_html(dataset.get("abstract"))
        field_of_science = cls._as_str(dataset.get("fieldOfScience")).strip()
        dryad_id = cls._as_str(dataset.get("id")).strip()
        source = cls._resolve_source(dataset, doi, title)
        text = abstract if abstract else cls._build_descriptor(authors, year, field_of_science)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "dryad",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "field_of_science": field_of_science,
                "dryad_id": dryad_id,
            },
        )

    @staticmethod
    def _normalize_doi(identifier: str) -> str:
        """Normalize a Dryad ``doi:`` identifier to a bare DOI string."""
        if not identifier:
            return ""
        value = identifier.strip()
        if value.lower().startswith(_DOI_PREFIX):
            return value[len(_DOI_PREFIX) :].strip()
        return value

    @staticmethod
    def _extract_year(publication_date: object) -> str:
        """Extract a four-digit year from a Dryad ``publicationDate``.

        Args:
            publication_date: The raw ``publicationDate`` field.

        Returns:
            The four-digit year string, or an empty string when absent/invalid.
        """
        if not isinstance(publication_date, str):
            return ""
        match = _YEAR_PREFIX_PATTERN.match(publication_date.strip())
        return match.group(1) if match else ""

    @classmethod
    def _extract_authors(cls, authors: object) -> list[str]:
        """Extract ordered author names from a Dryad ``authors`` list."""
        if not isinstance(authors, list):
            return []
        names: list[str] = []
        for entry in authors:
            if not isinstance(entry, dict):
                continue
            first = cls._as_str(entry.get("firstName")).strip()
            last = cls._as_str(entry.get("lastName")).strip()
            name = " ".join(part for part in (first, last) if part)
            if name:
                names.append(name)
        return names

    @classmethod
    def _resolve_source(cls, dataset: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for a Dryad dataset.

        ``sharingLink`` is preferred, then a DOI link, and finally the title
        as an anchor of last resort.

        Args:
            dataset: The dataset object.
            doi: The normalized DOI, if any.
            title: The title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        sharing_link = cls._as_str(dataset.get("sharingLink")).strip()
        if sharing_link:
            return sharing_link
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _build_descriptor(authors: list[str], year: str, field_of_science: str) -> str:
        """Compose a descriptor used when no abstract exists."""
        parts: list[str] = []
        if authors:
            parts.append("By " + ", ".join(authors))
        if field_of_science:
            parts.append(f"({field_of_science})")
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @classmethod
    def _strip_html(cls, abstract: object) -> str:
        """Reduce an HTML Dryad abstract to collapsed plain text."""
        if not isinstance(abstract, str) or not abstract.strip():
            return ""
        without_tags = _HTML_TAG_PATTERN.sub(" ", abstract)
        return " ".join(html.unescape(without_tags).split())

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar Dryad field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
