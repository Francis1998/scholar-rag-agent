"""Retraction-aware OpenAlex ingestion connector.

Scholarly RAG systems often miss retraction status (a common gap relative to
tools such as PaperQA and Elicit). OpenAlex exposes an ``is_retracted`` flag on
works, so this connector searches only retracted records via:

``GET https://api.openalex.org/works?filter=is_retracted:true&search=...``

and normalizes each hit into a :class:`Document` with explicit retraction
metadata for grounded literature review.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_PAGE_SIZE_CAP = 200
_DOI_PREFIX = "https://doi.org/"
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class RetractionWatchConnector:
    """Search OpenAlex for retracted works matching a free-text query."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional contact email for OpenAlex's polite pool. When
                omitted, ``OPENALEX_MAILTO`` (then ``UNPAYWALL_EMAIL``) is read
                from the environment when present.
        """
        self._mailto = (
            mailto
            or os.environ.get("OPENALEX_MAILTO", "").strip()
            or os.environ.get("UNPAYWALL_EMAIL", "").strip()
            or None
        )

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized documents for retracted OpenAlex works.

        Args:
            query: Free-text query used as OpenAlex ``search`` while filtering
                ``is_retracted:true``.
            max_results: Maximum number of retracted works to return.

        Returns:
            Normalized documents flagged as retracted. Blank queries,
            non-positive ``max_results``, unavailable API responses, and
            malformed payloads yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        params: dict[str, str | int] = {
            "filter": "is_retracted:true",
            "search": stripped,
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_results(payload, max_results)

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch OpenAlex works, returning an empty payload on API failure."""
        try:
            response = await client.get(OPENALEX_WORKS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex works search payload into documents."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        documents: list[Document] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            # Defense in depth: keep only explicitly retracted works.
            if item.get("is_retracted") is not True:
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one OpenAlex retracted work."""
        title = cls._as_str(item.get("title") or item.get("display_name")).strip()
        if not title:
            return None

        openalex_id = cls._as_str(item.get("id")).strip()
        doi = cls._normalize_doi(item.get("doi"))
        source = (
            openalex_id or (f"{_DOI_PREFIX}{doi}" if doi else "") or cls._landing_url(item) or title
        )
        authors = cls._extract_authors(item.get("authorships"))
        year = cls._extract_year(item)
        journal = cls._extract_journal(item)
        cited_by = cls._as_str(item.get("cited_by_count")).strip()
        abstract = cls._reconstruct_abstract(item.get("abstract_inverted_index"))
        text = abstract or cls._build_text(authors, year, journal, doi, cited_by)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "retraction_watch",
                "is_retracted": "true",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "journal": journal,
                "openalex_id": openalex_id,
                "cited_by_count": cited_by,
                "landing_url": cls._landing_url(item),
            },
        )

    @classmethod
    def _extract_authors(cls, value: object) -> list[str]:
        """Extract ordered author display names from OpenAlex authorships."""
        if not isinstance(value, list):
            return []
        authors: list[str] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            author = entry.get("author")
            if isinstance(author, dict):
                name = cls._as_str(author.get("display_name")).strip()
            else:
                name = ""
            if name:
                authors.append(name)
        return authors

    @classmethod
    def _extract_year(cls, item: dict[str, object]) -> str:
        """Resolve publication year from OpenAlex year or date fields."""
        year = cls._as_str(item.get("publication_year")).strip()
        if year.isdigit():
            return year
        published_date = cls._as_str(item.get("publication_date")).strip()
        match = _YEAR_PREFIX_PATTERN.match(published_date)
        return match.group(1) if match else ""

    @classmethod
    def _extract_journal(cls, item: dict[str, object]) -> str:
        """Return the primary location source display name when present."""
        location = item.get("primary_location")
        if not isinstance(location, dict):
            return ""
        source = location.get("source")
        if isinstance(source, dict):
            return cls._as_str(source.get("display_name")).strip()
        return ""

    @classmethod
    def _landing_url(cls, item: dict[str, object]) -> str:
        """Prefer primary landing page, then DOI URL, then OpenAlex id."""
        location = item.get("primary_location")
        if isinstance(location, dict):
            landing = cls._as_str(location.get("landing_page_url")).strip()
            if landing:
                return landing
        doi = cls._normalize_doi(item.get("doi"))
        if doi:
            return f"{_DOI_PREFIX}{doi}"
        return cls._as_str(item.get("id")).strip()

    @staticmethod
    def _normalize_doi(value: object) -> str:
        """Normalize an OpenAlex DOI field to a bare DOI string."""
        if not isinstance(value, str):
            return ""
        doi = value.strip()
        if doi.lower().startswith(_DOI_PREFIX):
            doi = doi[len(_DOI_PREFIX) :]
        return doi.strip()

    @staticmethod
    def _reconstruct_abstract(inverted_index: object) -> str:
        """Reconstruct abstract text from an OpenAlex inverted index."""
        if not isinstance(inverted_index, dict) or not inverted_index:
            return ""
        positioned: dict[int, str] = {}
        for word, positions in inverted_index.items():
            if not isinstance(word, str) or not isinstance(positions, list):
                continue
            for position in positions:
                if isinstance(position, int) and not isinstance(position, bool):
                    positioned[position] = word
        ordered_words = [positioned[index] for index in sorted(positioned)]
        return " ".join(ordered_words)

    @staticmethod
    def _build_text(
        authors: list[str],
        year: str,
        journal: str,
        doi: str,
        cited_by: str,
    ) -> str:
        """Compose searchable text when no abstract is available."""
        parts: list[str] = ["Retracted work."]
        if authors:
            parts.append("By " + ", ".join(authors))
        if journal:
            parts.append(f"in {journal}")
        if year:
            parts.append(f"({year})")
        if doi:
            parts.append(f"DOI {doi}")
        if cited_by:
            parts.append(f"cited_by_count={cited_by}")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar OpenAlex values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
