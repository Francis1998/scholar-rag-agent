"""Semantic Scholar API ingestion connector."""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_PAPER_FIELDS = "title,abstract,year,authors,url,publicationDate"
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")
_PAGE_SIZE_CAP = 100


class SemanticScholarConnector:
    """Fetch and normalize Semantic Scholar paper records."""

    def __init__(self, api_key: str | None = None) -> None:
        """Create a connector with an optional API key."""
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Semantic Scholar documents matching a query.

        Args:
            query: Free-text Semantic Scholar query.
            max_results: Maximum number of papers to fetch (API caps ``limit``
                at 100).

        Returns:
            Normalized documents for the matching papers. An empty list is
            returned when the query is blank or ``max_results`` is non-positive.
        """
        if max_results <= 0 or not query.strip():
            return []

        headers = {"x-api-key": self._api_key} if self._api_key else None
        params: dict[str, str | int] = {
            "query": query.strip(),
            "limit": min(max_results, _PAGE_SIZE_CAP),
            "fields": _PAPER_FIELDS,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{S2_BASE_URL}/paper/search", params=params, headers=headers
            )
            response.raise_for_status()
        return self._parse_search_results(response.json(), max_results)

    async def fetch_paper(self, paper_id: str) -> Document:
        """Return one normalized Semantic Scholar paper by id."""
        headers = {"x-api-key": self._api_key} if self._api_key else None
        params = {"fields": _PAPER_FIELDS}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{S2_BASE_URL}/paper/{paper_id}", params=params, headers=headers
            )
            response.raise_for_status()
        return self._build_document(response.json(), fallback_source=paper_id)

    @classmethod
    def _parse_search_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Semantic Scholar ``paper/search`` JSON payload into documents.

        Args:
            payload: Decoded search response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each paper in ``data``.
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
            paper_id = item.get("paperId")
            fallback = paper_id if isinstance(paper_id, str) and paper_id else "semantic-scholar"
            documents.append(cls._build_document(item, fallback_source=fallback))
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, data: dict[str, object], *, fallback_source: str) -> Document:
        """Build a document from one Semantic Scholar paper object.

        Args:
            data: Decoded paper JSON.
            fallback_source: Source string used when ``url`` is absent.

        Returns:
            Normalized document.
        """
        title = str(data.get("title") or "Untitled Semantic Scholar paper")
        abstract = data.get("abstract")
        text = abstract.strip() if isinstance(abstract, str) else ""
        url = data.get("url")
        source = url.strip() if isinstance(url, str) and url.strip() else fallback_source
        return Document(
            document_id=stable_id(source, "doc"),
            title=title,
            text=text,
            source=source,
            metadata={
                "source_type": "semantic_scholar",
                "year": cls._resolve_year(data),
            },
        )

    @classmethod
    def _resolve_year(cls, data: dict[str, object]) -> str:
        """Resolve a publication year without turning a missing year into ``\"None\"``.

        Semantic Scholar may omit ``year`` (JSON ``null``) while still carrying a
        ``publicationDate`` such as ``2023-05-01``. Coercing a missing year with
        ``str(year)`` previously leaked the literal ``\"None\"`` into metadata;
        the year is now taken from an integer/digit ``year`` when present, else
        from the leading four digits of ``publicationDate``, else ``\"\"``.

        Args:
            data: Decoded paper JSON.

        Returns:
            The publication year as a string, or an empty string when absent.
        """
        year = data.get("year")
        if isinstance(year, int) and not isinstance(year, bool):
            return str(year)
        if isinstance(year, str) and year.strip().isdigit():
            return year.strip()
        publication_date = data.get("publicationDate")
        if isinstance(publication_date, str):
            match = _YEAR_PREFIX_PATTERN.match(publication_date.strip())
            if match:
                return match.group(1)
        return ""
