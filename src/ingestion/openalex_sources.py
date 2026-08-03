"""OpenAlex sources (venues) ingestion connector.

OpenAlex (https://openalex.org) exposes a public sources API that indexes
journals, repositories, conferences, and other venues with ISSNs, host
organization links, and bibliometric summaries. This connector searches the
sources endpoint and normalizes each hit into a :class:`Document` for
venue-aware literature discovery.

Free-text queries use:

``GET https://api.openalex.org/sources?search=...``

When the query is an OpenAlex source id (``S####``), the connector resolves
the source directly via:

``GET https://api.openalex.org/sources/S####``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"
_PAGE_SIZE_CAP = 200
_SOURCE_ID_PATTERN = re.compile(r"^S\d+$", re.IGNORECASE)


class OpenAlexSourcesConnector:
    """Search OpenAlex sources/venues and normalize matching records."""

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
        """Return normalized documents for matching OpenAlex sources.

        Args:
            query: Free-text source/venue search or an OpenAlex source id
                such as ``S137773608``.
            max_results: Maximum number of source documents to return.

        Returns:
            Normalized source documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        source_id = self._normalize_source_id(stripped)
        if source_id is not None:
            return await self._search_by_source_id(source_id, max_results)

        params: dict[str, str | int] = {
            "search": stripped,
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, OPENALEX_SOURCES_URL, params)
        return self._parse_source_results(payload, max_results)

    async def _search_by_source_id(
        self,
        source_id: str,
        max_results: int,
    ) -> list[Document]:
        """Resolve one OpenAlex source id directly."""
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(
                client,
                f"{OPENALEX_SOURCES_URL}/{source_id}",
                params,
            )
        return self._parse_source_results(
            {"results": [payload]} if isinstance(payload, dict) else {},
            max_results,
        )

    @staticmethod
    def _normalize_source_id(query: str) -> str | None:
        """Return a bare OpenAlex source id when ``query`` is id-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://openalex.org/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _SOURCE_ID_PATTERN.fullmatch(candidate):
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
    def _parse_source_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex sources payload into documents."""
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
            document = cls._build_source_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_source_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one OpenAlex source/venue record."""
        display_name = cls._as_str(item.get("display_name")).strip()
        if not display_name:
            return None

        source_id = cls._extract_source_id(item.get("id"))
        source_type = cls._as_str(item.get("type")).strip()
        host_org = cls._as_str(item.get("host_organization_name")).strip()
        issn_l = cls._as_str(item.get("issn_l")).strip()
        is_oa = item.get("is_oa")
        works_count = cls._as_str(item.get("works_count")).strip()
        cited_by_count = cls._as_str(item.get("cited_by_count")).strip()
        homepage = cls._as_str(item.get("homepage_url")).strip()
        summary_stats = cls._as_dict(item.get("summary_stats"))
        h_index = cls._as_str(summary_stats.get("h_index")).strip()
        issn_list = item.get("issn")
        issn_joined = ""
        if isinstance(issn_list, list):
            issn_joined = ", ".join(
                cls._as_str(value).strip() for value in issn_list if cls._as_str(value).strip()
            )

        source = cls._as_str(item.get("id")).strip() or (
            f"https://openalex.org/{source_id}" if source_id else display_name
        )
        text = cls._build_text(
            display_name,
            source_type,
            host_org,
            issn_l or issn_joined,
            works_count,
            cited_by_count,
            h_index,
            is_oa is True,
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openalex_sources",
                "openalex_source_id": source_id,
                "type": source_type,
                "host_organization": host_org,
                "issn_l": issn_l,
                "issn": issn_joined,
                "is_oa": "true" if is_oa is True else "false" if is_oa is False else "",
                "works_count": works_count,
                "cited_by_count": cited_by_count,
                "homepage_url": homepage,
                "h_index": h_index,
            },
        )

    @staticmethod
    def _extract_source_id(value: object) -> str:
        """Normalize an OpenAlex source id URL or bare id."""
        source_ref = OpenAlexSourcesConnector._as_str(value).strip()
        if not source_ref:
            return ""
        if source_ref.lower().startswith("https://openalex.org/"):
            return source_ref.rsplit("/", maxsplit=1)[-1].upper()
        return source_ref.upper()

    @staticmethod
    def _build_text(
        display_name: str,
        source_type: str,
        host_org: str,
        issn: str,
        works_count: str,
        cited_by_count: str,
        h_index: str,
        is_oa: bool,
    ) -> str:
        """Compose searchable text for an OpenAlex source profile."""
        parts: list[str] = [f"OpenAlex source {display_name}."]
        if source_type:
            parts.append(f"Type: {source_type}.")
        if host_org:
            parts.append(f"Host organization: {host_org}.")
        if issn:
            parts.append(f"ISSN: {issn}.")
        if is_oa:
            parts.append("Open access venue.")
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
