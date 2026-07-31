"""OpenAlex authors ingestion connector.

OpenAlex (https://openalex.org) exposes a public authors API that indexes
researcher profiles with display names, ORCID ids, affiliation history,
and bibliometric summaries. This connector searches the authors endpoint and
normalizes each hit into a :class:`Document` for author-centric literature
discovery.

Free-text queries use:

``GET https://api.openalex.org/authors?search=...``

When the query is an OpenAlex author id (``A####``), the connector resolves the
author directly via:

``GET https://api.openalex.org/authors/A####``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_AUTHORS_URL = "https://api.openalex.org/authors"
_PAGE_SIZE_CAP = 200
_AUTHOR_ID_PATTERN = re.compile(r"^A\d+$", re.IGNORECASE)


class OpenAlexAuthorsConnector:
    """Search OpenAlex authors and normalize matching records."""

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
        """Return normalized documents for matching OpenAlex authors.

        Args:
            query: Free-text author search or an OpenAlex author id such as
                ``A2208157607``.
            max_results: Maximum number of author documents to return.

        Returns:
            Normalized author documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        author_id = self._normalize_author_id(stripped)
        if author_id is not None:
            return await self._search_by_author_id(author_id, max_results)

        params: dict[str, str | int] = {
            "search": stripped,
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, OPENALEX_AUTHORS_URL, params)
        return self._parse_author_results(payload, max_results)

    async def _search_by_author_id(self, author_id: str, max_results: int) -> list[Document]:
        """Resolve one OpenAlex author id directly."""
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(
                client,
                f"{OPENALEX_AUTHORS_URL}/{author_id}",
                params,
            )
        return self._parse_author_results(
            {"results": [payload]} if isinstance(payload, dict) else {},
            max_results,
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
    def _parse_author_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex authors payload into documents."""
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
            document = cls._build_author_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_author_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one OpenAlex author record."""
        display_name = cls._as_str(item.get("display_name")).strip()
        if not display_name:
            return None

        author_id = cls._extract_author_id(item.get("id"))
        orcid = cls._extract_orcid(item.get("orcid"))
        works_count = cls._as_str(item.get("works_count")).strip()
        cited_by_count = cls._as_str(item.get("cited_by_count")).strip()
        institution = cls._primary_institution(item.get("last_known_institutions"))
        summary_stats = cls._as_dict(item.get("summary_stats"))
        h_index = cls._as_str(summary_stats.get("h_index")).strip()
        i10_index = cls._as_str(summary_stats.get("i10_index")).strip()

        source = cls._as_str(item.get("id")).strip() or (
            f"https://openalex.org/{author_id}" if author_id else display_name
        )
        text = cls._build_text(
            display_name,
            institution,
            works_count,
            cited_by_count,
            h_index,
            i10_index,
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openalex_authors",
                "author_id": author_id,
                "orcid": orcid,
                "works_count": works_count,
                "cited_by_count": cited_by_count,
                "institution": institution,
                "h_index": h_index,
                "i10_index": i10_index,
            },
        )

    @staticmethod
    def _extract_author_id(value: object) -> str:
        """Normalize an OpenAlex author id URL or bare id."""
        author_ref = OpenAlexAuthorsConnector._as_str(value).strip()
        if not author_ref:
            return ""
        if author_ref.lower().startswith("https://openalex.org/"):
            return author_ref.rsplit("/", maxsplit=1)[-1].upper()
        return author_ref.upper()

    @staticmethod
    def _extract_orcid(value: object) -> str:
        """Return a bare ORCID id from an OpenAlex ORCID URL or bare id."""
        orcid_ref = OpenAlexAuthorsConnector._as_str(value).strip()
        if not orcid_ref:
            return ""
        if orcid_ref.lower().startswith("https://orcid.org/"):
            return orcid_ref.rsplit("/", maxsplit=1)[-1]
        return orcid_ref

    @classmethod
    def _primary_institution(cls, value: object) -> str:
        """Return the display name of the first last-known institution."""
        if not isinstance(value, list):
            return ""
        for entry in value:
            if not isinstance(entry, dict):
                continue
            name = cls._as_str(entry.get("display_name")).strip()
            if name:
                return name
        return ""

    @staticmethod
    def _build_text(
        display_name: str,
        institution: str,
        works_count: str,
        cited_by_count: str,
        h_index: str,
        i10_index: str,
    ) -> str:
        """Compose searchable text for an OpenAlex author profile."""
        parts: list[str] = [f"OpenAlex author {display_name}."]
        if institution:
            parts.append(f"Affiliation: {institution}.")
        if works_count:
            parts.append(f"Works: {works_count}.")
        if cited_by_count:
            parts.append(f"Cited by count: {cited_by_count}.")
        if h_index:
            parts.append(f"h-index: {h_index}.")
        if i10_index:
            parts.append(f"i10-index: {i10_index}.")
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
        return ""
