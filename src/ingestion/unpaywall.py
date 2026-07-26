"""Unpaywall DOI open-access lookup ingestion connector.

Unpaywall (https://unpaywall.org) aggregates legal open-access locations for DOI
records. Its public API is DOI-centric: callers fetch one record at
``GET https://api.unpaywall.org/v2/{doi}?email=...`` and receive publication
metadata plus the best known OA landing page and PDF URL.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

UNPAYWALL_API_BASE = "https://api.unpaywall.org/v2"

_DOI_PATTERN = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\s,;<>\"']+)",
    re.IGNORECASE,
)
_TRAILING_DOI_CHARS = ".,;:)]}"
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class UnpaywallConnector:
    """Resolve DOI queries against Unpaywall open-access location metadata."""

    def __init__(self, email: str | None = None) -> None:
        """Create a connector.

        Args:
            email: Contact email required by the Unpaywall API. When omitted,
                ``UNPAYWALL_EMAIL`` is read from the environment. Requests are
                skipped when no email is configured.
        """
        self._email = (email or os.environ.get("UNPAYWALL_EMAIL", "")).strip()

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Unpaywall documents for DOI(s) in a query.

        Args:
            query: DOI-like string or free text containing one or more DOI
                identifiers.
            max_results: Maximum number of DOI lookups to issue.

        Returns:
            Normalized documents carrying OA landing/PDF location metadata. A
            blank query, non-positive ``max_results``, missing email, or query
            with no DOI identifiers returns an empty list without HTTP requests.
        """
        if max_results <= 0 or not query.strip() or not self._email:
            return []

        dois = self._extract_dois(query)[:max_results]
        if not dois:
            return []

        documents: list[Document] = []
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for doi in dois:
                payload = await self._fetch_payload(client, doi)
                document = self._build_document(payload, doi)
                if document is not None:
                    documents.append(document)
                if len(documents) >= max_results:
                    break
        return documents

    async def _fetch_payload(self, client: httpx.AsyncClient, doi: str) -> object:
        """Fetch one Unpaywall DOI record, returning ``None`` on lookup failure."""
        try:
            response = await client.get(
                f"{UNPAYWALL_API_BASE}/{doi}",
                params={"email": self._email},
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return None

    @classmethod
    def _build_document(cls, payload: object, requested_doi: str) -> Document | None:
        """Build a document from one Unpaywall API payload."""
        if not isinstance(payload, dict) or not payload:
            return None

        doi = cls._as_str(payload.get("doi")).strip() or requested_doi
        title = cls._as_str(payload.get("title")).strip() or f"Unpaywall OA record for {doi}"
        authors = cls._extract_authors(payload.get("z_authors"))
        year = cls._extract_year(payload)
        journal = cls._as_str(payload.get("journal_name")).strip()
        publisher = cls._as_str(payload.get("publisher")).strip()
        genre = cls._as_str(payload.get("genre")).strip()
        is_oa = cls._bool_str(payload.get("is_oa"))
        oa_status = cls._as_str(payload.get("oa_status")).strip()
        location = cls._best_oa_location(payload)
        landing_url = cls._location_url(location, ("url_for_landing_page", "url"))
        pdf_url = cls._location_url(location, ("url_for_pdf",))
        host_type = cls._as_str(location.get("host_type")).strip()
        version = cls._as_str(location.get("version")).strip()
        license_name = cls._as_str(location.get("license")).strip()
        doi_url = cls._as_str(payload.get("doi_url")).strip()
        source = landing_url or pdf_url or doi_url or (f"https://doi.org/{doi}" if doi else title)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=cls._build_text(authors, year, journal, is_oa, oa_status, landing_url, pdf_url),
            source=source,
            metadata={
                "source_type": "unpaywall",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "journal": journal,
                "publisher": publisher,
                "genre": genre,
                "is_oa": is_oa,
                "oa_status": oa_status,
                "landing_url": landing_url,
                "pdf_url": pdf_url,
                "host_type": host_type,
                "version": version,
                "license": license_name,
            },
        )

    @staticmethod
    def _extract_dois(query: str) -> list[str]:
        """Extract unique DOI identifiers from free text."""
        dois: list[str] = []
        seen: set[str] = set()
        for match in _DOI_PATTERN.finditer(query):
            doi = match.group(1).strip().rstrip(_TRAILING_DOI_CHARS)
            key = doi.lower()
            if doi and key not in seen:
                seen.add(key)
                dois.append(doi)
        return dois

    @classmethod
    def _extract_authors(cls, value: object) -> list[str]:
        """Extract ordered author names from Unpaywall ``z_authors`` entries."""
        if not isinstance(value, list):
            return []
        authors: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                name = entry.strip()
            elif isinstance(entry, dict):
                name = cls._as_str(entry.get("name")).strip()
                if not name:
                    given = cls._as_str(entry.get("given")).strip()
                    family = cls._as_str(entry.get("family")).strip()
                    name = " ".join(part for part in (given, family) if part)
            else:
                name = ""
            if name:
                authors.append(name)
        return authors

    @classmethod
    def _extract_year(cls, payload: dict[str, object]) -> str:
        """Resolve year from Unpaywall ``year`` or ``published_date`` fields."""
        year = cls._as_str(payload.get("year")).strip()
        if year.isdigit():
            return year
        published_date = cls._as_str(payload.get("published_date")).strip()
        match = _YEAR_PREFIX_PATTERN.match(published_date)
        return match.group(1) if match else ""

    @staticmethod
    def _best_oa_location(payload: dict[str, object]) -> dict[str, object]:
        """Return the best OA location object, falling back to the first location."""
        best = payload.get("best_oa_location")
        if isinstance(best, dict):
            return best

        locations = payload.get("oa_locations")
        if not isinstance(locations, list):
            return {}
        for location in locations:
            if isinstance(location, dict) and any(
                UnpaywallConnector._as_str(location.get(key)).strip()
                for key in ("url_for_landing_page", "url_for_pdf", "url")
            ):
                return location
        return {}

    @classmethod
    def _location_url(cls, location: dict[str, object], keys: tuple[str, ...]) -> str:
        """Return the first non-empty URL from an OA location object."""
        for key in keys:
            url = cls._as_str(location.get(key)).strip()
            if url:
                return url
        return ""

    @staticmethod
    def _build_text(
        authors: list[str],
        year: str,
        journal: str,
        is_oa: str,
        oa_status: str,
        landing_url: str,
        pdf_url: str,
    ) -> str:
        """Compose searchable text summarizing OA availability and URLs."""
        parts: list[str] = []
        if authors:
            parts.append("By " + ", ".join(authors))
        if journal:
            parts.append(f"in {journal}")
        if year:
            parts.append(f"({year})")
        status = oa_status or {"true": "open", "false": "closed"}.get(is_oa, "unknown")
        parts.append(f"OA status: {status}")
        if landing_url:
            parts.append(f"Landing page: {landing_url}")
        if pdf_url:
            parts.append(f"PDF: {pdf_url}")
        return " ".join(parts)

    @staticmethod
    def _bool_str(value: object) -> str:
        """Coerce a boolean API field to a lowercase metadata string."""
        if isinstance(value, bool):
            return str(value).lower()
        return ""

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar Unpaywall values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
