"""OpenAlex author works (citations blend) ingestion connector.

OpenAlex author profiles expose aggregate citation counts, but literature
workflows often need the underlying works with per-work ``cited_by_count``.
This connector resolves an author id or free-text author search, then fetches:

``GET https://api.openalex.org/works?filter=authorships.author.id:{id}``

Normalizes each work with citation metadata (``source_type=openalex_author_works``)
to fill the OpenAlex authors ↔ works citations blend gap. Distinct from
``openalex_authors.py`` (author profiles) and ``openalex.py`` (single-work fetch).

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_AUTHORS_URL = "https://api.openalex.org/authors"
_PAGE_SIZE_CAP = 200
_AUTHOR_ID_PATTERN = re.compile(r"^A\d+$", re.IGNORECASE)
_DOI_PREFIX = "https://doi.org/"
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class OpenAlexAuthorWorksConnector:
    """Resolve an OpenAlex author and normalize their works with citations."""

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
        """Return works for an OpenAlex author id or author name search.

        Blank queries, non-positive limits, failed requests, and malformed
        payloads yield an empty list.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        author_id = self._normalize_author_id(stripped)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            if author_id is None:
                author_id = await self._resolve_author_id(client, stripped)
            if not author_id:
                return []
            payload = await self._fetch_works(client, author_id, max_results)
        return self._parse_results(payload, max_results, author_id)

    async def _resolve_author_id(self, client: httpx.AsyncClient, query: str) -> str:
        """Resolve a free-text author query to the top OpenAlex author id."""
        params: dict[str, str | int] = {"search": query, "per-page": 1}
        if self._mailto:
            params["mailto"] = self._mailto
        payload = await self._fetch_payload(client, OPENALEX_AUTHORS_URL, params)
        if not isinstance(payload, dict):
            return ""
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return ""
        first = results[0]
        if not isinstance(first, dict):
            return ""
        return self._extract_author_id(first.get("id"))

    async def _fetch_works(
        self,
        client: httpx.AsyncClient,
        author_id: str,
        max_results: int,
    ) -> object:
        """Fetch works authored by ``author_id``."""
        params: dict[str, str | int] = {
            "filter": f"authorships.author.id:{author_id}",
            "per-page": min(max_results, _PAGE_SIZE_CAP),
            "sort": "cited_by_count:desc",
        }
        if self._mailto:
            params["mailto"] = self._mailto
        return await self._fetch_payload(client, OPENALEX_WORKS_URL, params)

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str | int],
    ) -> object:
        """Fetch an OpenAlex endpoint, returning an empty payload on failure."""
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
        max_results: int,
        author_id: str,
    ) -> list[Document]:
        """Parse an OpenAlex works payload into author-works documents."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        documents: list[Document] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item, author_id)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object], author_id: str) -> Document | None:
        """Build a document from one OpenAlex work with citation counts."""
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
                "source_type": "openalex_author_works",
                "author_id": author_id,
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "journal": journal,
                "openalex_id": openalex_id,
                "cited_by_count": cited_by,
                "landing_url": cls._landing_url(item),
            },
        )

    @staticmethod
    def _normalize_author_id(query: str) -> str | None:
        """Return a bare OpenAlex author id when ``query`` is author-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://openalex.org/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _AUTHOR_ID_PATTERN.fullmatch(candidate):
            return candidate.upper()
        return None

    @staticmethod
    def _extract_author_id(value: object) -> str:
        """Normalize an OpenAlex author id URL or bare id."""
        author_ref = OpenAlexAuthorWorksConnector._as_str(value).strip()
        if not author_ref:
            return ""
        if author_ref.lower().startswith("https://openalex.org/"):
            return author_ref.rsplit("/", maxsplit=1)[-1].upper()
        return author_ref.upper()

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
        """Return a bare DOI from an OpenAlex DOI URL or string."""
        doi_ref = OpenAlexAuthorWorksConnector._as_str(value).strip()
        if not doi_ref:
            return ""
        lower = doi_ref.lower()
        if lower.startswith("https://doi.org/"):
            return doi_ref[len("https://doi.org/") :]
        if lower.startswith("http://doi.org/"):
            return doi_ref[len("http://doi.org/") :]
        return doi_ref

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
        return " ".join(positioned[index] for index in sorted(positioned))

    @staticmethod
    def _build_text(
        authors: list[str],
        year: str,
        journal: str,
        doi: str,
        cited_by: str,
    ) -> str:
        """Compose searchable text when an abstract is unavailable."""
        parts = ["OpenAlex author work."]
        if authors:
            parts.append(f"Authors: {', '.join(authors)}.")
        if journal:
            parts.append(f"Journal: {journal}.")
        if year:
            parts.append(f"Year: {year}.")
        if cited_by:
            parts.append(f"Cited by count: {cited_by}.")
        if doi:
            parts.append(f"DOI: {doi}.")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar OpenAlex values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else str(value)
        return ""
