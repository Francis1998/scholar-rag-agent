"""OpenAlex keywords ingestion connector.

OpenAlex (https://openalex.org) exposes a keywords entity that tags scholarly
works with human-readable research keywords and coverage statistics. This
connector searches the public keywords endpoint and normalizes each hit into a
:class:`Document` for keyword-aware literature discovery.

Free-text queries use:

``GET https://api.openalex.org/keywords?search=...``

When the query is an OpenAlex keyword slug (``machine-learning``) or keyword URL
(``https://openalex.org/keywords/machine-learning``), the connector resolves the
keyword directly via:

``GET https://api.openalex.org/keywords/{slug}``

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_KEYWORDS_URL = "https://api.openalex.org/keywords"
_PAGE_SIZE_CAP = 200
_KEYWORD_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", re.IGNORECASE)


class OpenAlexKeywordsConnector:
    """Search OpenAlex keywords and normalize matching records."""

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
        """Return normalized documents for matching OpenAlex keywords.

        Args:
            query: Free-text keyword search or an OpenAlex keyword slug / URL.
            max_results: Maximum number of keyword documents to return.

        Returns:
            Normalized keyword documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        keyword_slug = self._normalize_keyword_slug(stripped)
        if keyword_slug is not None:
            return await self._search_by_keyword_slug(keyword_slug, max_results)

        params: dict[str, str | int] = {
            "search": stripped,
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, OPENALEX_KEYWORDS_URL, params)
        return self._parse_keyword_results(payload, max_results)

    async def _search_by_keyword_slug(
        self,
        keyword_slug: str,
        max_results: int,
    ) -> list[Document]:
        """Resolve one OpenAlex keyword slug directly."""
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(
                client,
                f"{OPENALEX_KEYWORDS_URL}/{keyword_slug}",
                params,
            )
        return self._parse_keyword_results(
            {"results": [payload]} if isinstance(payload, dict) else {},
            max_results,
        )

    @staticmethod
    def _normalize_keyword_slug(query: str) -> str | None:
        """Return a bare OpenAlex keyword slug for URL / ``keywords/`` forms.

        Bare free-text tokens (even hyphenated ones) stay on the search endpoint
        so queries like ``machine learning`` and ``transformers`` return ranked
        keyword hits rather than a single direct lookup.
        """
        candidate = query.strip()
        lower = candidate.lower()
        if lower.startswith("https://openalex.org/keywords/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        elif lower.startswith("keywords/"):
            candidate = candidate.split("/", maxsplit=1)[-1]
        else:
            return None
        if " " in candidate:
            return None
        if _KEYWORD_SLUG_PATTERN.fullmatch(candidate):
            return candidate.lower()
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
    def _parse_keyword_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex keywords payload into documents."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            single = payload if payload.get("id") or payload.get("display_name") else None
            results = [single] if isinstance(single, dict) else []

        documents: list[Document] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            document = cls._build_keyword_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_keyword_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one OpenAlex keyword record."""
        display_name = cls._as_str(item.get("display_name")).strip()
        if not display_name:
            return None

        openalex_ref = cls._as_str(item.get("id")).strip()
        keyword_slug = cls._extract_keyword_slug(openalex_ref) or cls._slugify(display_name)
        works_count = cls._as_str(item.get("works_count")).strip()
        cited_by_count = cls._as_str(item.get("cited_by_count")).strip()
        works_api_url = cls._as_str(item.get("works_api_url")).strip()

        source = openalex_ref or (
            f"https://openalex.org/keywords/{keyword_slug}" if keyword_slug else display_name
        )
        text = cls._build_text(display_name, works_count, cited_by_count)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openalex_keywords",
                "openalex_keyword_id": keyword_slug,
                "openalex": openalex_ref,
                "works_count": works_count,
                "cited_by_count": cited_by_count,
                "works_api_url": works_api_url,
            },
        )

    @staticmethod
    def _extract_keyword_slug(value: object) -> str:
        """Normalize an OpenAlex keyword URL or bare slug."""
        ref = OpenAlexKeywordsConnector._as_str(value).strip()
        if not ref:
            return ""
        lower = ref.lower()
        if "/keywords/" in lower:
            return ref.rsplit("/", maxsplit=1)[-1].lower()
        if lower.startswith("keywords/"):
            return ref.split("/", maxsplit=1)[-1].lower()
        return ref.lower()

    @staticmethod
    def _slugify(display_name: str) -> str:
        """Best-effort slug from a display name when OpenAlex id is absent."""
        return "-".join(display_name.lower().split())

    @staticmethod
    def _build_text(display_name: str, works_count: str, cited_by_count: str) -> str:
        """Compose searchable text for an OpenAlex keyword profile."""
        parts: list[str] = [f"OpenAlex keyword {display_name}."]
        if works_count:
            parts.append(f"Works: {works_count}.")
        if cited_by_count:
            parts.append(f"Cited by count: {cited_by_count}.")
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
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return ""
