"""Crossref works type-filter ingestion connector.

Crossref works can be filtered by resource type (journal-article,
proceedings-article, book-chapter, ...). Popular scholarly RAG loaders often
expose type facets that the base Crossref connector lacks. This connector
searches ``/works`` with a ``filter=type:...`` clause and normalizes hits for
agentic synthesis with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import html
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

CROSSREF_BASE_URL = "https://api.crossref.org/works"
_JATS_TAG_PATTERN = re.compile(r"<[^>]+>")
_DEFAULT_TYPE = "journal-article"
_TYPE_SPLIT = re.compile(r"<<<TYPE>>>", re.IGNORECASE)
_ALLOWED_TYPES = frozenset(
    {
        "journal-article",
        "proceedings-article",
        "book-chapter",
        "book",
        "posted-content",
        "dataset",
        "dissertation",
        "peer-review",
        "report",
        "standard",
    }
)


class CrossrefTypesConnector:
    """Search Crossref works filtered by resource type."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional polite-pool contact email.
        """
        self._mailto = mailto

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return type-filtered Crossref works as documents.

        Args:
            query: Free-text query, optionally ``query<<<TYPE>>>journal-article``.
            max_results: Maximum number of works to return.

        Returns:
            Normalized documents. Blank queries and non-positive limits yield [].
        """
        text_query, work_type = self._split_query(query)
        if max_results <= 0 or not text_query:
            return []

        params: dict[str, str | int] = {
            "query": text_query,
            "rows": max_results,
            "filter": f"type:{work_type}",
        }
        if self._mailto:
            params["mailto"] = self._mailto

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(CROSSREF_BASE_URL, params=params)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return []

        return self._parse_works(payload, max_results, work_type)

    @classmethod
    def _split_query(cls, query: str) -> tuple[str, str]:
        """Split optional type sentinel from the free-text query."""
        parts = _TYPE_SPLIT.split(query.strip(), maxsplit=1)
        text_query = parts[0].strip()
        work_type = _DEFAULT_TYPE
        if len(parts) == 2:
            candidate = parts[1].strip().lower()
            if candidate in _ALLOWED_TYPES:
                work_type = candidate
        return text_query, work_type

    @classmethod
    def _parse_works(cls, payload: object, max_results: int, work_type: str) -> list[Document]:
        """Parse a Crossref works payload into type-tagged documents."""
        if not isinstance(payload, dict):
            return []
        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        items = message.get("items")
        if not isinstance(items, list):
            return []

        documents: list[Document] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item, work_type)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object], work_type: str) -> Document | None:
        """Build a document from one Crossref work item."""
        titles = item.get("title")
        title = ""
        if isinstance(titles, list) and titles:
            title = str(titles[0] or "").strip()
        if not title:
            return None
        abstract_raw = str(item.get("abstract") or "")
        abstract = html.unescape(_JATS_TAG_PATTERN.sub(" ", abstract_raw)).strip()
        doi = str(item.get("DOI") or "").strip()
        year = ""
        published = item.get("published-print") or item.get("published-online") or {}
        if isinstance(published, dict):
            date_parts = published.get("date-parts")
            if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list):
                year = str(date_parts[0][0]) if date_parts[0] else ""
        container = ""
        containers = item.get("container-title")
        if isinstance(containers, list) and containers:
            container = str(containers[0] or "").strip()
        item_type = str(item.get("type") or work_type).strip() or work_type
        source = f"https://doi.org/{doi}" if doi else title
        text_parts = [title + "."]
        if abstract:
            text_parts.append(abstract)
        text_parts.append(f"Crossref type: {item_type}.")
        if container:
            text_parts.append(f"Container: {container}.")
        if year:
            text_parts.append(f"Year: {year}.")
        if doi:
            text_parts.append(f"DOI: {doi}.")
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=" ".join(text_parts),
            source=source,
            metadata={
                "source_type": "crossref_types",
                "crossref_type": item_type,
                "doi": doi,
                "year": year,
                "container_title": container,
            },
        )
