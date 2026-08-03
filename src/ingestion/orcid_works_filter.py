"""ORCID public works deep-filter ingestion connector.

ORCID (https://orcid.org) exposes author-curated public work summaries under
``https://pub.orcid.org/v3.0``. This connector builds on the same public API as
``orcid.py`` but adds deep filters for publication year and work type so RAG
pipelines can narrow an author's public record (for example journal articles
from 2024) before normalizing documents.

Filter inputs may be supplied as constructor/search args or embedded in the
query string:

* ``year:2024`` / ``year=2024`` — keep works whose publication year matches
* work-type tokens such as ``journal-article``, ``preprint``, ``book-chapter``

ORCID iD queries (bare or URL) fetch ``{orcid}/works`` and apply filters
client-side. Keyword queries use ``expanded-search``, then fetch each
candidate's works and apply year/type (and remaining keyword token) filters.
"""

from __future__ import annotations

import re

import httpx

from ingestion.orcid import OrcidConnector
from retrieval.models import Document

ORCID_API_BASE = "https://pub.orcid.org/v3.0"
ORCID_EXPANDED_SEARCH_URL = f"{ORCID_API_BASE}/expanded-search/"
_PAGE_SIZE_CAP = 100

_YEAR_FILTER_PATTERN = re.compile(r"\byear\s*[:=]\s*(\d{4})\b", re.IGNORECASE)
_WORK_TYPE_PATTERN = re.compile(
    r"\b("
    r"journal-article|magazine-article|newsletter-article|newspaper-article|"
    r"book-chapter|book-review|conference-paper|conference-abstract|"
    r"conference-poster|dissertation-thesis|working-paper|online-resource|"
    r"lecture-speech|supervised-student-publication|research-technique|"
    r"research-tool|artistic-performance|data-set|dataset|preprint|"
    r"software|patent|report|other|book"
    r")\b",
    re.IGNORECASE,
)


class OrcidWorksFilterConnector:
    """Search ORCID public works with year and work-type deep filters."""

    def __init__(
        self,
        *,
        year: str | int | None = None,
        work_type: str | None = None,
    ) -> None:
        """Create a connector with optional default filters.

        Args:
            year: Optional default publication year filter (``YYYY``).
            work_type: Optional default ORCID work-type filter (for example
                ``journal-article`` or ``preprint``).
        """
        self._default_year = self._normalize_year(year)
        self._default_work_type = self._normalize_work_type(work_type)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        year: str | int | None = None,
        work_type: str | None = None,
    ) -> list[Document]:
        """Return ORCID work documents matching query plus year/type filters.

        Args:
            query: Free-text ORCID search, ORCID iD (bare or URL), and/or
                embedded filters such as ``year:2024 journal-article``.
            max_results: Maximum number of work summaries to return.
            year: Optional year filter overriding the constructor default and
                any ``year:YYYY`` token in ``query``.
            work_type: Optional work-type filter overriding the constructor
                default and any work-type token in ``query``.

        Returns:
            Normalized documents with ``metadata.source_type`` set to
            ``orcid_works_filter``. Blank queries, non-positive
            ``max_results``, and empty matches yield an empty list.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        filter_year, filter_work_type, remainder = self._parse_filters(
            stripped,
            year=year,
            work_type=work_type,
        )
        orcid_id = OrcidConnector._extract_orcid_id(stripped)
        if not orcid_id:
            orcid_id = OrcidConnector._extract_orcid_id(remainder)

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        ) as client:
            if orcid_id:
                payload = await self._fetch_works(client, orcid_id)
                return self._parse_works(
                    payload,
                    max_results,
                    orcid_id,
                    orcid_id,
                    year=filter_year,
                    work_type=filter_work_type,
                    filter_tokens=None,
                )

            search_query = remainder.strip()
            if not search_query:
                return []

            search_payload = await self._search_records(client, search_query, max_results)
            candidates = OrcidConnector._extract_candidates(search_payload)
            tokens = OrcidConnector._tokens(search_query)
            documents: list[Document] = []
            seen_document_ids: set[str] = set()
            for candidate_orcid, candidate_name in candidates:
                works_payload = await self._fetch_works(client, candidate_orcid)
                for document in self._parse_works(
                    works_payload,
                    max_results - len(documents),
                    candidate_orcid,
                    candidate_name or candidate_orcid,
                    year=filter_year,
                    work_type=filter_work_type,
                    filter_tokens=tokens or None,
                ):
                    if document.document_id in seen_document_ids:
                        continue
                    seen_document_ids.add(document.document_id)
                    documents.append(document)
                    if len(documents) >= max_results:
                        return documents
            return documents

    async def _search_records(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> object:
        """Fetch matching ORCID records via expanded search."""
        try:
            response = await client.get(
                ORCID_EXPANDED_SEARCH_URL,
                params={"q": query, "rows": min(max_results, _PAGE_SIZE_CAP)},
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    async def _fetch_works(self, client: httpx.AsyncClient, orcid_id: str) -> object:
        """Fetch public work summaries for one ORCID iD."""
        try:
            response = await client.get(f"{ORCID_API_BASE}/{orcid_id}/works")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_works(
        cls,
        payload: object,
        max_results: int,
        orcid_id: str,
        profile_name: str,
        *,
        year: str,
        work_type: str,
        filter_tokens: set[str] | None,
    ) -> list[Document]:
        """Parse ORCID works, applying year/type and optional keyword filters."""
        if max_results <= 0 or not isinstance(payload, dict):
            return []
        groups = payload.get("group")
        if not isinstance(groups, list):
            return []

        documents: list[Document] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            summaries = OrcidConnector._as_list(group.get("work-summary"))
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                if not cls._passes_filters(summary, year=year, work_type=work_type):
                    continue
                if filter_tokens and not OrcidConnector._matches(summary, filter_tokens):
                    continue
                document = cls._build_document(summary, orcid_id, profile_name)
                if document is not None:
                    documents.append(document)
                if len(documents) >= max_results:
                    return documents
        return documents

    @classmethod
    def _build_document(
        cls,
        summary: dict[str, object],
        orcid_id: str,
        profile_name: str,
    ) -> Document | None:
        """Build a filtered ORCID work document."""
        document = OrcidConnector._build_document(summary, orcid_id, profile_name)
        if document is None:
            return None
        metadata = dict(document.metadata)
        metadata["source_type"] = "orcid_works_filter"
        return Document(
            document_id=document.document_id,
            title=document.title,
            text=document.text,
            source=document.source,
            metadata=metadata,
        )

    @classmethod
    def _passes_filters(
        cls,
        summary: dict[str, object],
        *,
        year: str,
        work_type: str,
    ) -> bool:
        """Return True when the summary satisfies year and work-type filters."""
        if year:
            summary_year = OrcidConnector._extract_year(summary.get("publication-date"))
            if summary_year != year:
                return False
        if work_type:
            summary_type = OrcidConnector._as_str(summary.get("type")).strip().lower()
            if summary_type != work_type:
                return False
        return True

    def _parse_filters(
        self,
        query: str,
        *,
        year: str | int | None,
        work_type: str | None,
    ) -> tuple[str, str, str]:
        """Extract year/work-type filters and return the residual search text."""
        resolved_year = self._normalize_year(year) or self._default_year
        resolved_work_type = self._normalize_work_type(work_type) or self._default_work_type
        remainder = query

        year_match = _YEAR_FILTER_PATTERN.search(remainder)
        if year_match:
            if not resolved_year:
                resolved_year = year_match.group(1)
            remainder = remainder[: year_match.start()] + " " + remainder[year_match.end() :]

        type_match = _WORK_TYPE_PATTERN.search(remainder)
        if type_match:
            if not resolved_work_type:
                resolved_work_type = type_match.group(1).lower()
            remainder = remainder[: type_match.start()] + " " + remainder[type_match.end() :]

        # Prefer explicit args over query tokens (already applied above).
        # Strip ORCID iDs from remainder when they appear alongside filters so
        # keyword search is not polluted; direct ORCID resolution uses the
        # original query via ``_extract_orcid_id``.
        return resolved_year, resolved_work_type, " ".join(remainder.split())

    @staticmethod
    def _normalize_year(year: str | int | None) -> str:
        """Normalize a year filter to a four-digit string when valid."""
        if year is None:
            return ""
        if isinstance(year, int) and not isinstance(year, bool):
            text = str(year)
        else:
            text = str(year).strip()
        return text if text.isdigit() and len(text) == 4 else ""

    @staticmethod
    def _normalize_work_type(work_type: str | None) -> str:
        """Normalize a work-type filter to lowercase hyphenated ORCID form."""
        if work_type is None:
            return ""
        normalized = work_type.strip().lower().replace("_", "-").replace(" ", "-")
        if not normalized:
            return ""
        match = _WORK_TYPE_PATTERN.fullmatch(normalized)
        return match.group(1).lower() if match else normalized
