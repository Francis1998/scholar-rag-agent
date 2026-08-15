"""OpenAlex sources host-organization ingestion connector.

OpenAlex sources expose venue identity together with a host organization
(publisher) link. This connector resolves host organizations and returns the
sources (venues) they host, including bibliometric summaries. It is distinct
from ``openalex_sources.py`` (general venue profiles) and
``openalex_sources_hierarchy.py`` (ancestry-path normalization).

Host-organization ids (``P####``) use:

``GET /sources?filter=host_organization:https://openalex.org/P####``

Free-text queries search publishers, then fetch sources for each resolved host.

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re

import httpx

from ingestion.openalex_sources import OpenAlexSourcesConnector
from retrieval.models import Document

OPENALEX_PUBLISHERS_URL = "https://api.openalex.org/publishers"
OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"
_PAGE_SIZE_CAP = 200
_HOST_ORG_ID_PATTERN = re.compile(r"^P\d+$", re.IGNORECASE)


class OpenAlexSourcesHostOrgConnector(OpenAlexSourcesConnector):
    """Search OpenAlex sources filtered by host organization."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return source documents for a host organization id or name."""
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        host_org_id = self._normalize_host_org_id(stripped)
        if host_org_id is not None:
            return await self._search_by_host_org(host_org_id, "", max_results)

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            publisher_payload = await self._fetch_payload(
                client,
                OPENALEX_PUBLISHERS_URL,
                {
                    "search": stripped,
                    "per-page": min(max_results, _PAGE_SIZE_CAP),
                    **({"mailto": self._mailto} if self._mailto else {}),
                },
            )
            publishers = self._extract_publishers(publisher_payload)
            documents: list[Document] = []
            seen_document_ids: set[str] = set()
            for publisher_id, publisher_name in publishers:
                for document in await self._search_by_host_org(
                    publisher_id,
                    publisher_name,
                    max_results - len(documents),
                    client=client,
                ):
                    if document.document_id in seen_document_ids:
                        continue
                    seen_document_ids.add(document.document_id)
                    documents.append(document)
                    if len(documents) >= max_results:
                        return documents
            return documents

    async def _search_by_host_org(
        self,
        host_org_id: str,
        host_org_name: str,
        max_results: int,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> list[Document]:
        """Fetch sources hosted by one OpenAlex organization id."""
        if max_results <= 0:
            return []

        params: dict[str, str | int] = {
            "filter": f"host_organization:https://openalex.org/{host_org_id}",
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        if client is None:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as owned:
                payload = await self._fetch_payload(owned, OPENALEX_SOURCES_URL, params)
        else:
            payload = await self._fetch_payload(client, OPENALEX_SOURCES_URL, params)

        return self._parse_host_org_results(payload, max_results, host_org_id, host_org_name)

    @staticmethod
    def _normalize_host_org_id(query: str) -> str | None:
        """Return a bare OpenAlex publisher/host id when ``query`` is id-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://openalex.org/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _HOST_ORG_ID_PATTERN.fullmatch(candidate):
            return candidate.upper()
        return None

    @classmethod
    def _extract_publishers(cls, payload: object) -> list[tuple[str, str]]:
        """Return publisher id/name pairs from an OpenAlex publishers payload."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        publishers: list[tuple[str, str]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            publisher_id = cls._extract_openalex_id(item.get("id"))
            display_name = cls._as_str(item.get("display_name")).strip()
            if publisher_id:
                publishers.append((publisher_id, display_name))
        return publishers

    @classmethod
    def _parse_host_org_results(
        cls,
        payload: object,
        max_results: int,
        host_org_id: str,
        host_org_name: str,
    ) -> list[Document]:
        """Parse OpenAlex sources into host-organization documents."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        documents: list[Document] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            document = cls._build_host_org_document(item, host_org_id, host_org_name)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_host_org_document(
        cls,
        item: dict[str, object],
        host_org_id: str,
        host_org_name: str,
    ) -> Document | None:
        """Build a source document enriched with host-organization context."""
        document = super()._build_source_document(item)
        if document is None:
            return None

        resolved_host_id = cls._extract_openalex_id(item.get("host_organization")) or host_org_id
        resolved_host_name = (
            cls._as_str(item.get("host_organization_name")).strip() or host_org_name
        )
        works_count = cls._as_str(item.get("works_count")).strip()
        display_name = cls._as_str(item.get("display_name")).strip()

        document.text = (
            f"OpenAlex source hosted by {resolved_host_name or resolved_host_id}: {document.text}"
        ).strip()
        document.metadata.update(
            {
                "source_type": "openalex_sources_host_org",
                "host_organization": resolved_host_id,
                "host_organization_name": resolved_host_name,
                "host_org_works_count": works_count,
            }
        )
        if display_name:
            document.metadata["source_display_name"] = display_name
        return document

    @staticmethod
    def _extract_openalex_id(value: object) -> str:
        """Normalize a bare or URL-shaped OpenAlex entity identifier."""
        ref = OpenAlexSourcesHostOrgConnector._as_str(value).strip()
        if not ref:
            return ""
        return ref.rsplit("/", maxsplit=1)[-1].upper()
