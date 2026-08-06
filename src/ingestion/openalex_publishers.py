"""OpenAlex publishers ingestion connector.

OpenAlex (https://openalex.org) exposes a public publishers API that indexes
publishing organizations with hierarchy, country codes, and bibliometric
summaries. This connector searches the publishers endpoint and normalizes each
hit into a :class:`Document` for publisher-aware literature discovery.

Free-text queries use:

``GET https://api.openalex.org/publishers?search=...``

When the query is an OpenAlex publisher id (``P####``), the connector resolves
the publisher directly via:

``GET https://api.openalex.org/publishers/P####``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_PUBLISHERS_URL = "https://api.openalex.org/publishers"
_PAGE_SIZE_CAP = 200
_PUBLISHER_ID_PATTERN = re.compile(r"^P\d+$", re.IGNORECASE)


class OpenAlexPublishersConnector:
    """Search OpenAlex publishers and normalize matching records."""

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
        """Return normalized documents for matching OpenAlex publishers.

        Args:
            query: Free-text publisher search or an OpenAlex publisher id
                such as ``P4310319900``.
            max_results: Maximum number of publisher documents to return.

        Returns:
            Normalized publisher documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        publisher_id = self._normalize_publisher_id(stripped)
        if publisher_id is not None:
            return await self._search_by_publisher_id(publisher_id, max_results)

        params: dict[str, str | int] = {
            "search": stripped,
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, OPENALEX_PUBLISHERS_URL, params)
        return self._parse_publisher_results(payload, max_results)

    async def _search_by_publisher_id(
        self,
        publisher_id: str,
        max_results: int,
    ) -> list[Document]:
        """Resolve one OpenAlex publisher id directly."""
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(
                client,
                f"{OPENALEX_PUBLISHERS_URL}/{publisher_id}",
                params,
            )
        return self._parse_publisher_results(
            {"results": [payload]} if isinstance(payload, dict) else {},
            max_results,
        )

    @staticmethod
    def _normalize_publisher_id(query: str) -> str | None:
        """Return a bare OpenAlex publisher id when ``query`` is id-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://openalex.org/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _PUBLISHER_ID_PATTERN.fullmatch(candidate):
            return candidate.upper()
        return None

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str | int],
    ) -> object:
        """Fetch an OpenAlex endpoint, returning an empty payload on failure."""
        try:
            response = await client.get(url, params=params or None)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_publisher_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex publishers payload into documents."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            single = payload if payload.get("id") else None
            results = [single] if isinstance(single, dict) else []

        documents: list[Document] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            document = cls._build_publisher_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_publisher_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one OpenAlex publisher record."""
        display_name = cls._as_str(item.get("display_name")).strip()
        if not display_name:
            return None

        publisher_id = cls._extract_publisher_id(item.get("id"))
        country_codes = item.get("country_codes")
        countries = ""
        if isinstance(country_codes, list):
            countries = ", ".join(
                cls._as_str(value).strip() for value in country_codes if cls._as_str(value).strip()
            )
        hierarchy_level = cls._as_str(item.get("hierarchy_level")).strip()
        parent = cls._as_dict(item.get("parent_publisher"))
        parent_name = cls._as_str(parent.get("display_name")).strip()
        works_count = cls._as_str(item.get("works_count")).strip()
        cited_by_count = cls._as_str(item.get("cited_by_count")).strip()
        homepage = cls._as_str(item.get("homepage_url")).strip()
        summary_stats = cls._as_dict(item.get("summary_stats"))
        h_index = cls._as_str(summary_stats.get("h_index")).strip()
        alternate_titles = item.get("alternate_titles")
        aliases = ""
        if isinstance(alternate_titles, list):
            aliases = ", ".join(
                cls._as_str(value).strip()
                for value in alternate_titles[:8]
                if cls._as_str(value).strip()
            )

        source = cls._as_str(item.get("id")).strip() or (
            f"https://openalex.org/{publisher_id}" if publisher_id else display_name
        )
        text = cls._build_text(
            display_name,
            parent_name,
            countries,
            hierarchy_level,
            works_count,
            cited_by_count,
            h_index,
            aliases,
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openalex_publishers",
                "openalex_publisher_id": publisher_id,
                "country_codes": countries,
                "hierarchy_level": hierarchy_level,
                "parent_publisher": parent_name,
                "works_count": works_count,
                "cited_by_count": cited_by_count,
                "homepage_url": homepage,
                "h_index": h_index,
                "alternate_titles": aliases,
            },
        )

    @staticmethod
    def _extract_publisher_id(value: object) -> str:
        """Normalize an OpenAlex publisher id URL or bare id."""
        publisher_ref = OpenAlexPublishersConnector._as_str(value).strip()
        if not publisher_ref:
            return ""
        if publisher_ref.lower().startswith("https://openalex.org/"):
            return publisher_ref.rsplit("/", maxsplit=1)[-1].upper()
        return publisher_ref.upper()

    @staticmethod
    def _build_text(
        display_name: str,
        parent_name: str,
        countries: str,
        hierarchy_level: str,
        works_count: str,
        cited_by_count: str,
        h_index: str,
        aliases: str,
    ) -> str:
        """Compose searchable text for an OpenAlex publisher profile."""
        parts: list[str] = [f"OpenAlex publisher {display_name}."]
        if parent_name:
            parts.append(f"Parent publisher: {parent_name}.")
        if countries:
            parts.append(f"Country codes: {countries}.")
        if hierarchy_level:
            parts.append(f"Hierarchy level: {hierarchy_level}.")
        if aliases:
            parts.append(f"Also known as: {aliases}.")
        if works_count:
            parts.append(f"Works: {works_count}.")
        if cited_by_count:
            parts.append(f"Cited by count: {cited_by_count}.")
        if h_index:
            parts.append(f"h-index: {h_index}.")
        return " ".join(parts)

    @staticmethod
    def _as_dict(value: object) -> dict[str, object]:
        """Return a dict value or an empty dict."""
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar OpenAlex values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return ""
