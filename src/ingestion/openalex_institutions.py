"""OpenAlex institutions ingestion connector.

OpenAlex (https://openalex.org) exposes a public institutions API that indexes
universities, hospitals, companies, and other research organizations with
display names, geographic metadata, ROR and Wikidata links, and bibliometric
summaries. This connector searches the institutions endpoint and normalizes
each hit into a :class:`Document` for affiliation-aware literature discovery.

Free-text queries use:

``GET https://api.openalex.org/institutions?search=...``

When the query is an OpenAlex institution id (``I####``), the connector resolves
the institution directly via:

``GET https://api.openalex.org/institutions/I####``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_INSTITUTIONS_URL = "https://api.openalex.org/institutions"
_PAGE_SIZE_CAP = 200
_INSTITUTION_ID_PATTERN = re.compile(r"^I\d+$", re.IGNORECASE)


class OpenAlexInstitutionsConnector:
    """Search OpenAlex institutions and normalize matching records."""

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
        """Return normalized documents for matching OpenAlex institutions.

        Args:
            query: Free-text institution search or an OpenAlex institution id
                such as ``I136199984``.
            max_results: Maximum number of institution documents to return.

        Returns:
            Normalized institution documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        institution_id = self._normalize_institution_id(stripped)
        if institution_id is not None:
            return await self._search_by_institution_id(institution_id, max_results)

        params: dict[str, str | int] = {
            "search": stripped,
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, OPENALEX_INSTITUTIONS_URL, params)
        return self._parse_institution_results(payload, max_results)

    async def _search_by_institution_id(
        self,
        institution_id: str,
        max_results: int,
    ) -> list[Document]:
        """Resolve one OpenAlex institution id directly."""
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(
                client,
                f"{OPENALEX_INSTITUTIONS_URL}/{institution_id}",
                params,
            )
        return self._parse_institution_results(
            {"results": [payload]} if isinstance(payload, dict) else {},
            max_results,
        )

    @staticmethod
    def _normalize_institution_id(query: str) -> str | None:
        """Return a bare OpenAlex institution id when ``query`` is id-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://openalex.org/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _INSTITUTION_ID_PATTERN.fullmatch(candidate):
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
    def _parse_institution_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex institutions payload into documents."""
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
            document = cls._build_institution_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_institution_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one OpenAlex institution record."""
        display_name = cls._as_str(item.get("display_name")).strip()
        if not display_name:
            return None

        institution_id = cls._extract_institution_id(item.get("id"))
        institution_type = cls._as_str(item.get("type")).strip()
        country_code = cls._as_str(item.get("country_code")).strip()
        works_count = cls._as_str(item.get("works_count")).strip()
        cited_by_count = cls._as_str(item.get("cited_by_count")).strip()
        ror = cls._extract_ror(item.get("ror"))
        wikidata = cls._extract_wikidata(item.get("ids"))
        geo = cls._as_dict(item.get("geo"))
        city = cls._as_str(geo.get("city")).strip()
        country = cls._as_str(geo.get("country")).strip()
        summary_stats = cls._as_dict(item.get("summary_stats"))
        h_index = cls._as_str(summary_stats.get("h_index")).strip()

        source = cls._as_str(item.get("id")).strip() or (
            f"https://openalex.org/{institution_id}" if institution_id else display_name
        )
        text = cls._build_text(
            display_name,
            institution_type,
            city,
            country,
            works_count,
            cited_by_count,
            h_index,
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openalex_institutions",
                "institution_id": institution_id,
                "type": institution_type,
                "country_code": country_code,
                "works_count": works_count,
                "cited_by_count": cited_by_count,
                "ror": ror,
                "wikidata": wikidata,
                "city": city,
                "country": country,
                "h_index": h_index,
            },
        )

    @staticmethod
    def _extract_institution_id(value: object) -> str:
        """Normalize an OpenAlex institution id URL or bare id."""
        institution_ref = OpenAlexInstitutionsConnector._as_str(value).strip()
        if not institution_ref:
            return ""
        if institution_ref.lower().startswith("https://openalex.org/"):
            return institution_ref.rsplit("/", maxsplit=1)[-1].upper()
        return institution_ref.upper()

    @staticmethod
    def _extract_ror(value: object) -> str:
        """Return a bare ROR id from an OpenAlex ROR URL or bare id."""
        ror_ref = OpenAlexInstitutionsConnector._as_str(value).strip()
        if not ror_ref:
            return ""
        if ror_ref.lower().startswith("https://ror.org/"):
            return ror_ref.rsplit("/", maxsplit=1)[-1]
        return ror_ref

    @classmethod
    def _extract_wikidata(cls, value: object) -> str:
        """Return a Wikidata Q-id from an OpenAlex ``ids`` object or bare id."""
        if isinstance(value, dict):
            wikidata_ref = cls._as_str(value.get("wikidata")).strip()
            if wikidata_ref:
                if wikidata_ref.lower().startswith("https://www.wikidata.org/wiki/"):
                    return wikidata_ref.rsplit("/", maxsplit=1)[-1]
                return wikidata_ref
        wikidata_ref = cls._as_str(value).strip()
        if wikidata_ref.lower().startswith("https://www.wikidata.org/wiki/"):
            return wikidata_ref.rsplit("/", maxsplit=1)[-1]
        return wikidata_ref

    @staticmethod
    def _build_text(
        display_name: str,
        institution_type: str,
        city: str,
        country: str,
        works_count: str,
        cited_by_count: str,
        h_index: str,
    ) -> str:
        """Compose searchable text for an OpenAlex institution profile."""
        parts: list[str] = [f"OpenAlex institution {display_name}."]
        if institution_type:
            parts.append(f"Type: {institution_type}.")
        location = ", ".join(part for part in (city, country) if part)
        if location:
            parts.append(f"Location: {location}.")
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
