"""OpenAlex funders ingestion connector.

OpenAlex (https://openalex.org) exposes a public funders API that indexes
research funding organizations with country, identifier, grant, and bibliometric
summaries. This connector searches the funders endpoint and normalizes each hit
into a :class:`Document` for funding-aware literature discovery.

Free-text queries use:

``GET https://api.openalex.org/funders?search=...``

When the query is an OpenAlex funder id (``F####``), the connector resolves the
funder directly via:

``GET https://api.openalex.org/funders/F####``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_FUNDERS_URL = "https://api.openalex.org/funders"
_PAGE_SIZE_CAP = 200
_FUNDER_ID_PATTERN = re.compile(r"^F\d+$", re.IGNORECASE)


class OpenAlexFundersConnector:
    """Search OpenAlex funders and normalize matching records."""

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
        """Return normalized documents for matching OpenAlex funders.

        Args:
            query: Free-text funder search or an OpenAlex funder id such as
                ``F4320306076``.
            max_results: Maximum number of funder documents to return.

        Returns:
            Normalized funder documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        funder_id = self._normalize_funder_id(stripped)
        if funder_id is not None:
            return await self._search_by_funder_id(funder_id, max_results)

        params: dict[str, str | int] = {
            "search": stripped,
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, OPENALEX_FUNDERS_URL, params)
        return self._parse_funder_results(payload, max_results)

    async def _search_by_funder_id(
        self,
        funder_id: str,
        max_results: int,
    ) -> list[Document]:
        """Resolve one OpenAlex funder id directly."""
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(
                client,
                f"{OPENALEX_FUNDERS_URL}/{funder_id}",
                params,
            )
        return self._parse_funder_results(
            {"results": [payload]} if isinstance(payload, dict) else {},
            max_results,
        )

    @staticmethod
    def _normalize_funder_id(query: str) -> str | None:
        """Return a bare OpenAlex funder id when ``query`` is id-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://openalex.org/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _FUNDER_ID_PATTERN.fullmatch(candidate):
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
    def _parse_funder_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex funders payload into documents."""
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
            document = cls._build_funder_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_funder_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one OpenAlex funder record."""
        display_name = cls._as_str(item.get("display_name")).strip()
        if not display_name:
            return None

        ids = cls._as_dict(item.get("ids"))
        openalex_ref = (
            cls._as_str(ids.get("openalex")).strip() or cls._as_str(item.get("id")).strip()
        )
        funder_id = cls._extract_funder_id(openalex_ref)
        description = (
            cls._as_str(item.get("description")).strip() or cls._as_str(item.get("summary")).strip()
        )
        country = (
            cls._as_str(item.get("country")).strip()
            or cls._as_str(item.get("country_code")).strip()
        )
        grants_count = cls._as_str(item.get("grants_count")).strip()
        works_count = cls._as_str(item.get("works_count")).strip()
        cited_by_count = cls._as_str(item.get("cited_by_count")).strip()
        ror = cls._extract_ror(ids.get("ror") or item.get("ror"))
        wikidata = cls._extract_wikidata(ids.get("wikidata"))

        source = cls._as_str(item.get("id")).strip() or (
            f"https://openalex.org/{funder_id}" if funder_id else display_name
        )
        text = cls._build_text(
            display_name,
            description,
            country,
            grants_count,
            works_count,
            cited_by_count,
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openalex_funders",
                "openalex_funder_id": funder_id,
                "openalex": openalex_ref,
                "ror": ror,
                "wikidata": wikidata,
                "description": description,
                "country": country,
                "grants_count": grants_count,
                "works_count": works_count,
                "cited_by_count": cited_by_count,
            },
        )

    @staticmethod
    def _extract_funder_id(value: object) -> str:
        """Normalize an OpenAlex funder id URL or bare id."""
        funder_ref = OpenAlexFundersConnector._as_str(value).strip()
        if not funder_ref:
            return ""
        if funder_ref.lower().startswith("https://openalex.org/"):
            return funder_ref.rsplit("/", maxsplit=1)[-1].upper()
        return funder_ref.upper()

    @staticmethod
    def _extract_ror(value: object) -> str:
        """Return a bare ROR id from an OpenAlex ROR URL or bare id."""
        ror_ref = OpenAlexFundersConnector._as_str(value).strip()
        if not ror_ref:
            return ""
        if ror_ref.lower().startswith("https://ror.org/"):
            return ror_ref.rsplit("/", maxsplit=1)[-1]
        return ror_ref

    @classmethod
    def _extract_wikidata(cls, value: object) -> str:
        """Return a Wikidata Q-id from a URL or bare id."""
        wikidata_ref = cls._as_str(value).strip()
        if wikidata_ref.lower().startswith("https://www.wikidata.org/wiki/"):
            return wikidata_ref.rsplit("/", maxsplit=1)[-1]
        return wikidata_ref

    @staticmethod
    def _build_text(
        display_name: str,
        description: str,
        country: str,
        grants_count: str,
        works_count: str,
        cited_by_count: str,
    ) -> str:
        """Compose searchable text for an OpenAlex funder profile."""
        parts: list[str] = [f"OpenAlex funder {display_name}."]
        if description:
            parts.append(description)
        if country:
            parts.append(f"Country: {country}.")
        if grants_count:
            parts.append(f"Grants: {grants_count}.")
        if works_count:
            parts.append(f"Works: {works_count}.")
        if cited_by_count:
            parts.append(f"Cited by count: {cited_by_count}.")
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
