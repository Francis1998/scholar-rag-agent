"""bioRxiv / medRxiv preprint ingestion connector.

bioRxiv (https://www.biorxiv.org) and medRxiv (https://www.medrxiv.org) are the
Cold Spring Harbor Laboratory preprint servers for biology and health sciences.
They complement the existing sources (arXiv, Semantic Scholar, OpenAlex, PubMed,
Crossref, Europe PMC, DOAJ, DBLP, HAL, OpenAIRE, Zenodo, Figshare, CORE) with
early, unrefereed life-science and clinical research that may not yet appear in
publisher indexes.

The public content API does not offer a free-text search parameter; instead it
exposes recent posts (and DOI lookups) under
``https://api.biorxiv.org/details/{server}/{interval}``. This connector fetches
a recent window of posts for the chosen server (``biorxiv`` or ``medrxiv``),
filters them client-side by query terms against title and abstract, and
normalizes each match into a :class:`Document`. DOI-shaped queries are resolved
directly via the DOI detail endpoint.
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

BIORXIV_API_BASE = "https://api.biorxiv.org/details"
_SUPPORTED_SERVERS = frozenset({"biorxiv", "medrxiv"})
_PAGE_SIZE_CAP = 100
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class BioRxivConnector:
    """Search bioRxiv / medRxiv recent posts and normalize matches into documents."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        server: str = "biorxiv",
    ) -> list[Document]:
        """Return normalized preprint documents matching a query.

        Args:
            query: Free-text query matched against title and abstract, or a DOI
                (``10.1101/...``) resolved via the DOI detail endpoint.
            max_results: Maximum number of preprints to return.
            server: Preprint server — ``\"biorxiv\"`` or ``\"medrxiv\"``.

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
        else:
            # Fetch a recent window large enough to filter; API pages at 100.
            window = min(max(max_results * 20, max_results), _PAGE_SIZE_CAP)
            payload = await self._fetch_recent(normalized_server, window)

        return self._parse_collection(payload, stripped, max_results, normalized_server)

    async def _fetch_recent(self, server: str, window: int) -> object:
        """Fetch the ``window`` most recent posts for a server.

        Args:
            server: ``biorxiv`` or ``medrxiv``.
            window: Number of most-recent posts to request (capped at 100).

        Returns:
            Decoded JSON payload from the details endpoint.
        """
        url = f"{BIORXIV_API_BASE}/{server}/{window}"
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()

    async def _fetch_doi(self, server: str, doi: str) -> object:
        """Fetch a single preprint by DOI.

        Args:
            server: ``biorxiv`` or ``medrxiv``.
            doi: Preprint DOI.

        Returns:
            Decoded JSON payload from the DOI detail endpoint.
        """
        url = f"{BIORXIV_API_BASE}/{server}/{doi}/na"
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()

    @classmethod
    def _parse_collection(
        cls,
        payload: object,
        query: str,
        max_results: int,
        server: str,
    ) -> list[Document]:
        """Parse a bioRxiv details payload into filtered documents.

        Args:
            payload: Decoded API response.
            query: Original query (DOI or free text).
            max_results: Upper bound on documents returned.
            server: Normalized server name used for source URLs / metadata.

        Returns:
            Matching normalized documents.
        """
        if not isinstance(payload, dict):
            return []
        collection = payload.get("collection")
        if not isinstance(collection, list):
            return []

        is_doi_query = bool(_DOI_PATTERN.match(query))
        query_tokens = set() if is_doi_query else cls._tokens(query)

        documents: list[Document] = []
        for item in collection:
            if not isinstance(item, dict):
                continue
            if not is_doi_query and not cls._matches(item, query_tokens):
                continue
            document = cls._build_document(item, server)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object], server: str) -> Document | None:
        """Build a document from one bioRxiv / medRxiv collection entry.

        Args:
            item: A single preprint object from ``collection``.
            server: Normalized server name.

        Returns:
            Normalized document, or None when the entry carries no usable title.
        """
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
        text = " ".join(abstract.split()) if abstract else cls._build_descriptor(authors, year)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": item_server,
                "doi": doi,
                "year": year,
                "authors": authors,
                "category": category,
                "date": date,
            },
        )

    @staticmethod
    def _matches(item: dict[str, object], query_tokens: set[str]) -> bool:
        """Return True when every query token appears in title or abstract.

        Args:
            item: A preprint collection entry.
            query_tokens: Lowercase alphanumeric query tokens.

        Returns:
            Whether the entry matches the query.
        """
        if not query_tokens:
            return False
        haystack = " ".join(
            [
                BioRxivConnector._as_str(item.get("title")),
                BioRxivConnector._as_str(item.get("abstract")),
                BioRxivConnector._as_str(item.get("category")),
            ]
        ).lower()
        return all(token in haystack for token in query_tokens)

    @staticmethod
    def _tokens(query: str) -> set[str]:
        """Split a free-text query into lowercase alphanumeric tokens.

        Args:
            query: Free-text query.

        Returns:
            Set of tokens used for client-side filtering.
        """
        return set(re.findall(r"[a-z0-9]+", query.lower()))

    @staticmethod
    def _extract_year(date: str) -> str:
        """Extract a four-digit year from a ``YYYY-MM-DD`` (or similar) date.

        Args:
            date: Raw ``date`` field from the API.

        Returns:
            The year as a string, or empty when the date does not start with
            four digits.
        """
        match = _YEAR_PREFIX_PATTERN.match(date.strip())
        return match.group(1) if match else ""

    @staticmethod
    def _build_descriptor(authors: str, year: str) -> str:
        """Compose a citation-style descriptor when no abstract exists.

        Args:
            authors: Semicolon-separated author string from the API.
            year: Publication year, if any.

        Returns:
            A single-line descriptor, or an empty string when nothing is known.
        """
        parts: list[str] = []
        if authors:
            parts.append(f"By {authors}")
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar bioRxiv field value to a string.

        Args:
            value: A raw scalar field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
