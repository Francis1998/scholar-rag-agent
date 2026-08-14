"""Crossref works type-and-license ingestion connector.

Crossref work records can be filtered by both resource type and license
metadata. Structured ``type|license-url`` queries use:

``GET /works?filter=type:{type},license.url:{url}``

Free-text queries use the configured default type together with
``has-license:true``. This connector is distinct from ``crossref_types.py``
(type-only) and ``crossref_works_license.py`` (license-only).

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re

import httpx

from ingestion.crossref_works_license import CrossrefWorksLicenseConnector
from retrieval.models import Document

_DEFAULT_TYPE = "journal-article"
_WORK_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$", re.IGNORECASE)


class CrossrefWorksTypeLicenseConnector(CrossrefWorksLicenseConnector):
    """Search Crossref works using a combined type and license filter."""

    def __init__(
        self,
        mailto: str | None = None,
        default_type: str = _DEFAULT_TYPE,
    ) -> None:
        """Create a connector with an optional polite-pool email and default type."""
        super().__init__(mailto=mailto)
        self._default_type = self._normalize_work_type(default_type) or _DEFAULT_TYPE

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return Crossref works matching type/license or free-text input.

        ``journal-article|https://creativecommons.org/licenses/by/4.0`` applies
        both exact filters without a text query. Other non-blank input is a
        free-text query constrained to the default type and licensed works.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        work_type, license_url = self._split_type_license(stripped)
        params: dict[str, str | int] = {"rows": max_results}
        if license_url:
            params["filter"] = f"type:{work_type},license.url:{license_url}"
        else:
            params["query"] = stripped
            params["filter"] = f"type:{work_type},has-license:true"
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_type_license_results(payload, max_results, work_type, license_url)

    def _split_type_license(self, query: str) -> tuple[str, str]:
        """Return the selected type and optional license URL from a query."""
        work_type, separator, license_query = query.partition("|")
        normalized_type = self._normalize_work_type(work_type)
        license_url = self._extract_license_url(license_query) if separator else None
        if normalized_type and license_url:
            return normalized_type, license_url
        return self._default_type, ""

    @staticmethod
    def _normalize_work_type(value: str) -> str | None:
        """Normalize a safe Crossref type slug."""
        candidate = value.strip().lower()
        if _WORK_TYPE_PATTERN.fullmatch(candidate):
            return candidate
        return None

    @classmethod
    def _parse_type_license_results(
        cls,
        payload: object,
        max_results: int,
        work_type: str,
        license_url: str,
    ) -> list[Document]:
        """Parse Crossref works into type-and-license documents."""
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
            document = super()._build_document(item, license_url)
            if document is not None:
                item_type = cls._as_str(item.get("type")).strip() or work_type
                document.metadata["source_type"] = "crossref_works_type_license"
                document.metadata["crossref_type"] = item_type
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents
