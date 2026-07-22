"""Open Science Framework (OSF) ingestion connector.

OSF (https://osf.io) hosts multidisciplinary research projects, preprints, and
registrations. Its public JSON:API endpoints expose searchable preprints and
registered study records without requiring an API token, complementing the
existing source mix (PDF, arXiv, Semantic Scholar, OpenAlex, PubMed, Crossref,
Europe PMC, DOAJ, DBLP, HAL, OpenAIRE, Zenodo, Figshare, CORE, bioRxiv/medRxiv,
and NASA ADS) with transparent open-science workflows and preregistered
research plans.

This connector queries both ``/v2/preprints/`` and ``/v2/registrations/`` with
``filter[search]`` and normalizes JSON:API ``data`` entries into
:class:`Document` objects. OSF is a public service, so transient network errors,
HTTP errors, and malformed payloads return an empty result for the affected
endpoint rather than breaking optional ingestion flows.
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OSF_API_BASE = "https://api.osf.io/v2"
_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("preprint", f"{OSF_API_BASE}/preprints/"),
    ("registration", f"{OSF_API_BASE}/registrations/"),
)
_PAGE_SIZE_CAP = 100
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class OsfConnector:
    """Search OSF preprints and registrations and normalize them into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized OSF preprint and registration documents.

        Args:
            query: Free-text OSF search query.
            max_results: Maximum total number of documents to return.

        Returns:
            Normalized documents for matching OSF records. Blank queries,
            non-positive ``max_results``, unavailable endpoints, and malformed
            responses yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        params: dict[str, str | int] = {
            "filter[search]": stripped,
            "page[size]": min(max_results, _PAGE_SIZE_CAP),
            "embed": "contributors",
        }

        documents: list[Document] = []
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for resource_type, url in _ENDPOINTS:
                payload = await self._fetch_payload(client, url, params)
                documents.extend(self._parse_results(payload, resource_type, max_results))
        return documents[:max_results]

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str | int],
    ) -> object:
        """Fetch one OSF endpoint, returning an empty payload on API failure.

        Args:
            client: Shared HTTP client.
            url: OSF JSON:API endpoint URL.
            params: Query parameters for OSF search.

        Returns:
            Decoded JSON payload, or ``{}`` when the API is unavailable or
            returns invalid JSON.
        """
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(
        cls,
        payload: object,
        resource_type: str,
        max_results: int,
    ) -> list[Document]:
        """Parse an OSF JSON:API search payload into documents.

        Args:
            payload: Decoded OSF JSON payload.
            resource_type: ``preprint`` or ``registration``.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each titled entry under ``data``.
        """
        if not isinstance(payload, dict):
            return []
        entries = payload.get("data")
        if not isinstance(entries, list):
            return []

        documents: list[Document] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item, resource_type)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object], resource_type: str) -> Document | None:
        """Build a document from one OSF JSON:API entry.

        Args:
            item: A single OSF preprint or registration entry.
            resource_type: ``preprint`` or ``registration``.

        Returns:
            Normalized document, or None when the entry carries no usable title.
        """
        attributes = item.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}

        title = cls._as_str(attributes.get("title")).strip()
        if not title:
            return None

        description = cls._as_str(attributes.get("description")).strip()
        abstract = " ".join(description.split())
        authors = cls._extract_authors(item)
        year = cls._extract_year(attributes)
        doi = cls._extract_doi(item, attributes)
        osf_id = cls._as_str(item.get("id")).strip()
        category = cls._as_str(attributes.get("category")).strip()
        source = cls._resolve_source(item, doi, osf_id, title)
        text = abstract if abstract else cls._build_descriptor(authors, year, category)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "osf",
                "resource_type": resource_type,
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "category": category,
                "osf_id": osf_id,
            },
        )

    @classmethod
    def _extract_authors(cls, item: dict[str, object]) -> list[str]:
        """Extract contributor display names from common OSF embedded shapes.

        Args:
            item: A JSON:API entry, optionally carrying ``embeds.contributors``.

        Returns:
            Ordered contributor names, empty when no embedded contributors are
            available.
        """
        contributors = cls._embedded_contributors(item)
        names: list[str] = []
        for contributor in contributors:
            name = cls._contributor_name(contributor)
            if name:
                names.append(name)
        return names

    @staticmethod
    def _embedded_contributors(item: dict[str, object]) -> list[object]:
        """Return the embedded contributor array when present."""
        embeds = item.get("embeds")
        if not isinstance(embeds, dict):
            return []
        contributors = embeds.get("contributors")
        if not isinstance(contributors, dict):
            return []
        data = contributors.get("data")
        return data if isinstance(data, list) else []

    @classmethod
    def _contributor_name(cls, contributor: object) -> str:
        """Extract one contributor's full name from OSF embed variants."""
        if isinstance(contributor, str):
            return contributor.strip()
        if not isinstance(contributor, dict):
            return ""

        direct = cls._name_from_attributes(contributor.get("attributes"))
        if direct:
            return direct

        embeds = contributor.get("embeds")
        if not isinstance(embeds, dict):
            return ""
        user = embeds.get("users") or embeds.get("user")
        if not isinstance(user, dict):
            return ""
        user_data = user.get("data")
        if isinstance(user_data, dict):
            return cls._name_from_attributes(user_data.get("attributes"))
        return cls._name_from_attributes(user.get("attributes"))

    @classmethod
    def _name_from_attributes(cls, attributes: object) -> str:
        """Extract a contributor name from an attributes object."""
        if not isinstance(attributes, dict):
            return ""
        for key in ("full_name", "name", "bibliographic"):
            name = cls._as_str(attributes.get(key)).strip()
            if name:
                return name
        return ""

    @classmethod
    def _extract_year(cls, attributes: dict[str, object]) -> str:
        """Extract a year from the best available OSF date field.

        Args:
            attributes: OSF entry attributes.

        Returns:
            Four-digit year, or an empty string when dates are absent/invalid.
        """
        for key in ("date_published", "date_registered", "date_created", "date_modified"):
            value = cls._as_str(attributes.get(key)).strip()
            match = _YEAR_PREFIX_PATTERN.match(value)
            if match:
                return match.group(1)
        return ""

    @classmethod
    def _extract_doi(cls, item: dict[str, object], attributes: dict[str, object]) -> str:
        """Extract a DOI from known OSF attribute/link fields.

        Args:
            item: OSF JSON:API entry.
            attributes: OSF entry attributes.

        Returns:
            DOI string when present, else an empty string.
        """
        for key in ("doi", "article_doi", "preprint_doi"):
            doi = cls._as_str(attributes.get(key)).strip()
            if doi:
                return doi

        links = item.get("links")
        if isinstance(links, dict):
            for key in ("doi", "preprint_doi"):
                doi = cls._doi_from_url(cls._as_str(links.get(key)).strip())
                if doi:
                    return doi

        identifiers = attributes.get("identifiers")
        if isinstance(identifiers, list):
            for identifier in identifiers:
                doi = cls._doi_from_identifier(identifier)
                if doi:
                    return doi
        return ""

    @classmethod
    def _doi_from_identifier(cls, identifier: object) -> str:
        """Extract a DOI from an identifier object or scalar."""
        if isinstance(identifier, str):
            return cls._doi_from_url(identifier)
        if not isinstance(identifier, dict):
            return ""
        kind = cls._as_str(identifier.get("category") or identifier.get("type")).strip().lower()
        value = cls._as_str(identifier.get("value") or identifier.get("uri")).strip()
        if kind == "doi" and value:
            return cls._doi_from_url(value)
        return ""

    @staticmethod
    def _doi_from_url(value: str) -> str:
        """Normalize DOI URLs into bare DOI strings."""
        if not value:
            return ""
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if value.lower().startswith(prefix):
                return value[len(prefix) :].strip()
        return value.strip()

    @classmethod
    def _resolve_source(cls, item: dict[str, object], doi: str, osf_id: str, title: str) -> str:
        """Resolve the canonical OSF landing URL for a record."""
        links = item.get("links")
        if isinstance(links, dict):
            for key in ("html", "iri", "self"):
                url = cls._as_str(links.get(key)).strip()
                if url:
                    return url
        if osf_id:
            return f"https://osf.io/{osf_id}/"
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _build_descriptor(authors: list[str], year: str, category: str) -> str:
        """Compose a descriptor when OSF does not provide a description."""
        parts: list[str] = []
        if authors:
            parts.append("By " + ", ".join(authors))
        if category:
            parts.append(f"in {category}")
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar OSF value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
