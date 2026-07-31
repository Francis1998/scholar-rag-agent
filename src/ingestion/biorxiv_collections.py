"""bioRxiv / medRxiv collection ingestion connector.

bioRxiv (https://www.biorxiv.org) and medRxiv (https://www.medrxiv.org) expose a
public details API whose JSON responses include a ``collection`` array of preprint
metadata. Unlike the general :class:`BioRxivConnector`, this connector targets
subject-category collections via the ``?category=`` query parameter on date-range
requests, or fetches a recent window when the query is free text.

Category-shaped queries (for example ``cell biology`` or ``cell_biology``) call:

``GET https://api.biorxiv.org/details/{server}/{interval}?category=...``

Free-text queries fetch recent posts and filter the returned ``collection`` client
side by title and abstract tokens. DOI-shaped queries resolve directly via the
DOI detail endpoint.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

BIORXIV_API_BASE = "https://api.biorxiv.org/details"
_SUPPORTED_SERVERS = frozenset({"biorxiv", "medrxiv"})
_PAGE_SIZE_CAP = 100
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_ ]{1,60}$", re.IGNORECASE)


class BioRxivCollectionsConnector:
    """Search bioRxiv / medRxiv category collections and normalize matches."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        server: str = "biorxiv",
    ) -> list[Document]:
        """Return normalized preprint documents from bioRxiv collections.

        Args:
            query: Subject category name, free-text query matched against title
                and abstract, or a DOI (``10.1101/...``) resolved via the DOI
                detail endpoint.
            max_results: Maximum number of preprints to return.
            server: Preprint server — ``biorxiv`` or ``medrxiv``.

        Returns:
            Normalized documents for the matching preprints. An empty list is
            returned when the query is blank, ``max_results`` is non-positive,
            or nothing matches.
        """
        if max_results <= 0 or not query.strip():
            return []

        normalized_server = server.strip().lower()
        if normalized_server not in _SUPPORTED_SERVERS:
            raise ValueError(
                f"Unsupported bioRxiv server '{server}'; expected one of "
                f"{sorted(_SUPPORTED_SERVERS)}"
            )

        stripped = query.strip()
        if _DOI_PATTERN.match(stripped):
            payload = await self._fetch_doi(normalized_server, stripped)
            return self._parse_collection(payload, stripped, max_results, normalized_server)

        if self._looks_like_category(stripped):
            payload = await self._fetch_category(normalized_server, stripped)
            return self._parse_collection(payload, stripped, max_results, normalized_server)

        window = min(max(max_results * 20, max_results), _PAGE_SIZE_CAP)
        payload = await self._fetch_recent(normalized_server, window)
        return self._parse_collection(payload, stripped, max_results, normalized_server)

    async def _fetch_recent(self, server: str, window: int) -> object:
        """Fetch the ``window`` most recent posts for a server."""
        url = f"{BIORXIV_API_BASE}/{server}/{window}"
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()

    async def _fetch_category(self, server: str, category: str) -> object:
        """Fetch recent posts filtered by subject category."""
        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=30)
        interval = f"{start_date.isoformat()}/{end_date.isoformat()}"
        url = f"{BIORXIV_API_BASE}/{server}/{interval}"
        category_param = category.strip().replace(" ", "_")
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, params={"category": category_param})
            response.raise_for_status()
        return response.json()

    async def _fetch_doi(self, server: str, doi: str) -> object:
        """Fetch a single preprint by DOI."""
        url = f"{BIORXIV_API_BASE}/{server}/{doi}/na"
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()

    @classmethod
    def _looks_like_category(cls, query: str) -> bool:
        """Return True when ``query`` resembles a bioRxiv subject category."""
        normalized = query.strip().lower()
        if "_" in normalized:
            return _CATEGORY_PATTERN.fullmatch(normalized.replace("_", " ")) is not None
        return False

    @classmethod
    def _parse_collection(
        cls,
        payload: object,
        query: str,
        max_results: int,
        server: str,
    ) -> list[Document]:
        """Parse a bioRxiv details payload into filtered documents."""
        if not isinstance(payload, dict):
            return []
        collection = payload.get("collection")
        if not isinstance(collection, list):
            return []

        is_doi_query = bool(_DOI_PATTERN.match(query))
        is_category_query = not is_doi_query and cls._looks_like_category(query)
        query_tokens = set() if is_doi_query or is_category_query else cls._tokens(query)
        category_tokens = cls._tokens(query.replace("_", " ")) if is_category_query else set()

        documents: list[Document] = []
        for item in collection:
            if not isinstance(item, dict):
                continue
            if is_category_query and not cls._matches_category(item, category_tokens):
                continue
            if not is_doi_query and not is_category_query and not cls._matches(item, query_tokens):
                continue
            document = cls._build_document(item, server)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _matches(cls, item: dict[str, object], query_tokens: set[str]) -> bool:
        """Return True when any query token appears in title or abstract."""
        if not query_tokens:
            return True
        haystack = " ".join(
            [
                cls._as_str(item.get("title")).lower(),
                cls._as_str(item.get("abstract")).lower(),
                cls._as_str(item.get("category")).lower(),
            ]
        )
        return any(token in haystack for token in query_tokens)

    @classmethod
    def _matches_category(cls, item: dict[str, object], category_tokens: set[str]) -> bool:
        """Return True when the item category overlaps category query tokens."""
        category = cls._as_str(item.get("category")).lower()
        if not category_tokens:
            return True
        category_tokens_in_item = set(cls._tokens(category))
        return bool(category_tokens & category_tokens_in_item) or all(
            token in category for token in category_tokens
        )

    @classmethod
    def _build_document(cls, item: dict[str, object], server: str) -> Document | None:
        """Build a document from one bioRxiv collection entry."""
        title = cls._as_str(item.get("title")).strip()
        if not title:
            return None
        doi = cls._as_str(item.get("doi")).strip()
        abstract = cls._as_str(item.get("abstract")).strip()
        authors = cls._as_str(item.get("authors")).strip()
        date = cls._as_str(item.get("date")).strip()
        year = cls._extract_year(date)
        category = cls._as_str(item.get("category")).strip()
        item_server = cls._as_str(item.get("server")).strip().lower() or server
        source = f"https://www.{item_server}.org/content/{doi}" if doi else title
        text = (
            " ".join(abstract.split())
            if abstract
            else cls._build_descriptor(authors, year, category)
        )
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "biorxiv_collections",
                "doi": doi,
                "year": year,
                "authors": authors,
                "category": category,
                "server": item_server,
            },
        )

    @staticmethod
    def _tokens(query: str) -> set[str]:
        """Return lowercase alphanumeric tokens from a query."""
        return set(_TOKEN_PATTERN.findall(query.lower()))

    @staticmethod
    def _extract_year(date: str) -> str:
        """Extract a four-digit year from a bioRxiv date string."""
        match = _YEAR_PREFIX_PATTERN.match(date.strip())
        return match.group(1) if match else ""

    @staticmethod
    def _build_descriptor(authors: str, year: str, category: str) -> str:
        """Compose a descriptor when no abstract is available."""
        parts: list[str] = []
        if authors:
            parts.append(f"Authors: {authors}.")
        if category:
            parts.append(f"Category: {category}.")
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar bioRxiv field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
