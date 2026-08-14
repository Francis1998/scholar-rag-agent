"""DataCite DOI-prefix ingestion connector.

DataCite's public DOI endpoint supports a ``prefix`` parameter for records
registered under a specific DOI prefix:

``GET https://api.datacite.org/dois?prefix={prefix}&query=...``

A query shaped like ``10.xxxx`` is used directly as the prefix filter. Other
queries are sent as free text, optionally constrained by a configured default
prefix. This connector is distinct from general DataCite search, reports, and
related-identifier enrichment.

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re

import httpx

from ingestion.datacite import DataCiteConnector
from retrieval.models import Document

DATACITE_DOIS_URL = "https://api.datacite.org/dois"
_PAGE_SIZE_CAP = 100
_DOI_PREFIX_PATTERN = re.compile(r"^10\.\d{4,9}$")


class DataCiteDoisPrefixConnector(DataCiteConnector):
    """Search DataCite DOI records by prefix or prefix-scoped free text."""

    def __init__(self, default_prefix: str | None = None) -> None:
        """Create a connector with an optional default DOI prefix."""
        self._default_prefix = self._normalize_prefix(default_prefix or "")

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return DataCite DOI documents for a prefix or free-text query."""
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        query_prefix = self._normalize_prefix(stripped)
        selected_prefix = query_prefix or self._default_prefix
        params: dict[str, str | int] = {
            "page[size]": min(max_results, _PAGE_SIZE_CAP),
        }
        if query_prefix:
            params["prefix"] = query_prefix
        else:
            params["query"] = stripped
            if selected_prefix:
                params["prefix"] = selected_prefix

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)

        documents = DataCiteConnector._parse_results(payload, max_results)
        for document in documents:
            doi_prefix = self._extract_prefix(document.metadata.get("doi", ""))
            document.metadata["source_type"] = "datacite_dois_prefix"
            document.metadata["doi_prefix"] = doi_prefix or selected_prefix or ""
        return documents

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch DataCite DOIs, returning an empty payload on failure."""
        try:
            response = await client.get(DATACITE_DOIS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @staticmethod
    def _normalize_prefix(value: str) -> str | None:
        """Return a normalized DOI prefix when a value is prefix-shaped."""
        candidate = value.strip().lower()
        if _DOI_PREFIX_PATTERN.fullmatch(candidate):
            return candidate
        return None

    @classmethod
    def _extract_prefix(cls, doi: str) -> str:
        """Extract a valid registration prefix from a DOI."""
        candidate = doi.strip().split("/", maxsplit=1)[0]
        return cls._normalize_prefix(candidate) or ""
