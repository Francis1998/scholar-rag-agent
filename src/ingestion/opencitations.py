"""OpenCitations Meta and Index ingestion connector.

OpenCitations (https://opencitations.net) publishes open bibliographic metadata
and citation links. Its public Meta search endpoint is not available at the
documented ``/meta/v1/search`` routes, so this connector is DOI-centric: it
extracts DOI-shaped identifiers from a query, fetches OpenCitations Meta records
via ``GET https://api.opencitations.net/meta/v1/metadata/doi:{doi}``, and then
adds best-effort citation/reference counts from the Index v2 API.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENCITATIONS_META_URL = "https://api.opencitations.net/meta/v1/metadata"
OPENCITATIONS_INDEX_URL = "https://api.opencitations.net/index/v2"

_DOI_PATTERN = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\s,;<>\"']+)",
    re.IGNORECASE,
)
_BRACKET_METADATA_PATTERN = re.compile(r"\s*\[[^\]]*\]")
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")
_TRAILING_DOI_CHARS = ".,;:)]}"


class OpenCitationsConnector:
    """Resolve DOI queries against OpenCitations and normalize metadata records."""

    def __init__(self, access_token: str | None = None) -> None:
        """Create a connector.

        Args:
            access_token: Optional OpenCitations access token. When omitted,
                ``OPENCITATIONS_ACCESS_TOKEN`` is read from the environment.
                Public metadata can be fetched without a token, but OpenCitations
                recommends sending one for applications.
        """
        env_token = os.environ.get("OPENCITATIONS_ACCESS_TOKEN", "").strip()
        resolved = (access_token or env_token or "").strip()
        self._headers = {"authorization": resolved} if resolved else None

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized OpenCitations documents for DOI(s) in a query.

        Args:
            query: Free text that may contain one or more DOI identifiers.
            max_results: Maximum number of DOI metadata records to fetch.

        Returns:
            Normalized documents for DOI records found in OpenCitations Meta. A
            blank query, non-positive ``max_results``, or a query with no DOI
            returns an empty list without issuing HTTP requests.
        """
        if max_results <= 0 or not query.strip():
            return []

        dois = self._extract_dois(query)[:max_results]
        if not dois:
            return []

        ids = "__".join(f"doi:{doi}" for doi in dois)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(f"{OPENCITATIONS_META_URL}/{ids}", headers=self._headers)
            response.raise_for_status()
            payload = response.json()
            counts = await self._fetch_counts(client, dois)

        return self._parse_results(payload, max_results, counts)

    async def _fetch_counts(
        self,
        client: httpx.AsyncClient,
        dois: list[str],
    ) -> dict[str, dict[str, str]]:
        """Fetch best-effort citation/reference counts for each DOI.

        Args:
            client: Open HTTP client.
            dois: DOI identifiers already sent to Meta.

        Returns:
            Mapping keyed by lowercased DOI with string ``citation_count`` and
            ``reference_count`` values. Count failures are represented by empty
            strings so metadata ingestion does not fail on a slow Index call.
        """
        counts: dict[str, dict[str, str]] = {}
        for doi in dois:
            counts[doi.lower()] = {
                "citation_count": await self._fetch_count(client, "citation-count", doi),
                "reference_count": await self._fetch_count(client, "reference-count", doi),
            }
        return counts

    async def _fetch_count(self, client: httpx.AsyncClient, operation: str, doi: str) -> str:
        """Fetch one Index v2 count value.

        Args:
            client: Open HTTP client.
            operation: ``citation-count`` or ``reference-count``.
            doi: DOI identifier without the ``doi:`` prefix.

        Returns:
            Count as a string, or empty when unavailable.
        """
        url = f"{OPENCITATIONS_INDEX_URL}/{operation}/doi:{doi}"
        try:
            response = await client.get(url, headers=self._headers, timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return ""
        return self._extract_count(response.json())

    @classmethod
    def _parse_results(
        cls,
        payload: object,
        max_results: int,
        counts: dict[str, dict[str, str]],
    ) -> list[Document]:
        """Parse an OpenCitations Meta metadata payload into documents.

        Args:
            payload: Decoded Meta response.
            max_results: Upper bound on the number of documents returned.
            counts: Best-effort Index count metadata keyed by DOI.

        Returns:
            Normalized documents for each metadata record.
        """
        if not isinstance(payload, list):
            return []

        documents: list[Document] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item, counts)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(
        cls,
        record: dict[str, object],
        counts: dict[str, dict[str, str]],
    ) -> Document | None:
        """Build a document from one OpenCitations Meta record.

        Args:
            record: A single object from the Meta ``metadata`` endpoint.
            counts: Best-effort Index count metadata keyed by DOI.

        Returns:
            Normalized document, or None when the record carries no usable title.
        """
        title = cls._as_str(record.get("title")).strip()
        if not title:
            return None
        identifiers = cls._as_str(record.get("id")).strip()
        doi = cls._extract_identifier(identifiers, "doi")
        authors = cls._extract_people(record.get("author"))
        pub_date = cls._as_str(record.get("pub_date")).strip()
        year = cls._extract_year(pub_date)
        venue = cls._strip_bracket_metadata(cls._as_str(record.get("venue")).strip())
        work_type = cls._as_str(record.get("type")).strip()
        source = f"https://doi.org/{doi}" if doi else title
        count_metadata = counts.get(doi.lower(), {}) if doi else {}
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=cls._build_descriptor(authors, year, venue, work_type),
            source=source,
            metadata={
                "source_type": "opencitations",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "venue": venue,
                "publication_date": pub_date,
                "type": work_type,
                "citation_count": count_metadata.get("citation_count", ""),
                "reference_count": count_metadata.get("reference_count", ""),
                "identifiers": identifiers,
            },
        )

    @staticmethod
    def _extract_dois(query: str) -> list[str]:
        """Extract unique DOI identifiers from free text.

        Args:
            query: Free-text query or DOI list.

        Returns:
            Ordered DOI identifiers without ``doi:`` or DOI URL prefixes.
        """
        dois: list[str] = []
        seen: set[str] = set()
        for match in _DOI_PATTERN.finditer(query):
            doi = match.group(1).strip().rstrip(_TRAILING_DOI_CHARS)
            key = doi.lower()
            if doi and key not in seen:
                seen.add(key)
                dois.append(doi)
        return dois

    @staticmethod
    def _extract_identifier(identifiers: str, prefix: str) -> str:
        """Read a prefixed identifier from OpenCitations ``id`` text.

        Args:
            identifiers: Space-separated ID string such as ``doi:... omid:...``.
            prefix: Identifier prefix to extract.

        Returns:
            Identifier value without the prefix, or empty when absent.
        """
        needle = f"{prefix}:"
        for entry in identifiers.split():
            if entry.lower().startswith(needle):
                return entry[len(needle) :].strip()
        return ""

    @classmethod
    def _extract_people(cls, value: object) -> list[str]:
        """Extract author/editor names from an OpenCitations people field.

        Args:
            value: Field such as ``"Ada [omid:...]; Alan [omid:...]"``.

        Returns:
            Ordered names with bracketed OpenCitations IDs removed.
        """
        text = cls._as_str(value).strip()
        if not text:
            return []
        names: list[str] = []
        for part in text.split(";"):
            name = cls._strip_bracket_metadata(part.strip())
            if name:
                names.append(name)
        return names

    @staticmethod
    def _strip_bracket_metadata(value: str) -> str:
        """Remove bracketed OpenCitations identifier annotations from a field."""
        return " ".join(_BRACKET_METADATA_PATTERN.sub("", value).split())

    @staticmethod
    def _extract_year(pub_date: str) -> str:
        """Extract a four-digit year from an OpenCitations publication date."""
        match = _YEAR_PREFIX_PATTERN.match(pub_date.strip())
        return match.group(1) if match else ""

    @staticmethod
    def _extract_count(payload: object) -> str:
        """Extract a count string from an Index count payload."""
        if not isinstance(payload, list) or not payload:
            return ""
        first = payload[0]
        if not isinstance(first, dict):
            return ""
        count = first.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            return str(count)
        if isinstance(count, str) and count.strip().isdigit():
            return count.strip()
        return ""

    @staticmethod
    def _build_descriptor(
        authors: list[str],
        year: str,
        venue: str,
        work_type: str,
    ) -> str:
        """Compose citation-style text for metadata-only records."""
        parts: list[str] = []
        if authors:
            parts.append("By " + ", ".join(authors))
        if venue:
            parts.append(f"in {venue}")
        if work_type:
            parts.append(f"[{work_type}]")
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar OpenCitations field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
