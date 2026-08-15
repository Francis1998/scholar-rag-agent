"""Crossref works ISSN-and-type ingestion connector.

Crossref work records can be filtered by both ISSN and resource type.
Structured ``issn|type`` queries use:

``GET /works?filter=issn:{issn},type:{type}``

Free-text queries use the configured default type together with
``has-issn:true``. This connector is distinct from ``crossref_types.py``
(type-only) and ``crossref_journals.py`` (journal metadata).

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re

import httpx

from ingestion.crossref_works_license import CrossrefWorksLicenseConnector
from retrieval.models import Document

_DEFAULT_TYPE = "journal-article"
_ISSN_PATTERN = re.compile(r"^\d{4}-?\d{3}[\dXx]$")
_WORK_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$", re.IGNORECASE)


class CrossrefWorksIssnTypeConnector(CrossrefWorksLicenseConnector):
    """Search Crossref works using a combined ISSN and type filter."""

    def __init__(
        self,
        mailto: str | None = None,
        default_type: str = _DEFAULT_TYPE,
    ) -> None:
        """Create a connector with an optional polite-pool email and default type."""
        super().__init__(mailto=mailto)
        self._default_type = self._normalize_work_type(default_type) or _DEFAULT_TYPE

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return Crossref works matching ISSN/type or free-text input.

        ``1532-4435|journal-article`` applies both exact filters without a text
        query. Other non-blank input is a free-text query constrained to the
        default type and ISSN-bearing works.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        issn, work_type = self._split_issn_type(stripped)
        params: dict[str, str | int] = {"rows": max_results}
        if issn:
            params["filter"] = f"issn:{issn},type:{work_type}"
        else:
            params["query"] = stripped
            params["filter"] = f"type:{work_type},has-issn:true"
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_issn_type_results(payload, max_results, issn, work_type)

    def _split_issn_type(self, query: str) -> tuple[str, str]:
        """Return the selected ISSN and work type from a query."""
        issn_part, separator, type_part = query.partition("|")
        normalized_issn = self._normalize_issn(issn_part)
        normalized_type = self._normalize_work_type(type_part) if separator else None
        if normalized_issn and normalized_type:
            return normalized_issn, normalized_type
        return "", self._default_type

    @staticmethod
    def _normalize_issn(value: str) -> str | None:
        """Normalize a hyphenated ISSN when ``value`` is ISSN-shaped."""
        candidate = value.strip()
        if not _ISSN_PATTERN.fullmatch(candidate):
            return None
        compact = candidate.replace("-", "").upper()
        return f"{compact[:4]}-{compact[4:]}"

    @staticmethod
    def _normalize_work_type(value: str) -> str | None:
        """Normalize a safe Crossref type slug."""
        candidate = value.strip().lower()
        if _WORK_TYPE_PATTERN.fullmatch(candidate):
            return candidate
        return None

    @classmethod
    def _parse_issn_type_results(
        cls,
        payload: object,
        max_results: int,
        issn: str,
        work_type: str,
    ) -> list[Document]:
        """Parse Crossref works into ISSN-and-type documents."""
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
            document = super()._build_document(item, "")
            if document is not None:
                item_type = cls._as_str(item.get("type")).strip() or work_type
                document.metadata["source_type"] = "crossref_works_issn_type"
                document.metadata["crossref_type"] = item_type
                document.metadata["issn"] = issn or cls._extract_primary_issn(item)
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _extract_primary_issn(cls, item: dict[str, object]) -> str:
        """Return the first ISSN declared on a Crossref work."""
        issn_types = item.get("issn-type")
        if isinstance(issn_types, list):
            for entry in issn_types:
                if not isinstance(entry, dict):
                    continue
                value = cls._as_str(entry.get("value")).strip()
                if value:
                    return value
        issns = item.get("ISSN")
        if isinstance(issns, list) and issns:
            return cls._as_str(issns[0]).strip()
        return ""
