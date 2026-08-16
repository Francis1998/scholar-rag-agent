"""Crossref works-by-ISBN ingestion connector.

Crossref work records can be scoped to books and other ISBN-bearing outputs:

``GET https://api.crossref.org/works?filter=isbn:{isbn}``

ISBNs embedded in otherwise free-form input are extracted and filtered
directly. General free text uses ``query`` with ``has-isbn:true``. This
connector is distinct from the general Crossref and ISSN/type connectors.

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re

import httpx

from ingestion.crossref_works_license import CrossrefWorksLicenseConnector
from retrieval.models import Document

_ISBN_PATTERN = re.compile(r"(?<![\dXx])(?:97[89][\s-]*)?\d(?:[\s-]*\d){8}[\s-]*[\dXx](?![\dXx])")


class CrossrefWorksIsbnConnector(CrossrefWorksLicenseConnector):
    """Search and normalize Crossref works carrying ISBN metadata."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return ISBN-filtered or ISBN-bearing Crossref works.

        ISBN-10 and ISBN-13 values may be bare, hyphenated, or embedded in free
        text. Other text searches are limited to records declaring an ISBN.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        isbn = self._extract_isbn(stripped)
        params: dict[str, str | int] = {"rows": max_results}
        if isbn:
            params["filter"] = f"isbn:{isbn}"
        else:
            params["query"] = stripped
            params["filter"] = "has-isbn:true"
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_isbn_results(payload, max_results, isbn)

    @classmethod
    def _extract_isbn(cls, query: str) -> str:
        """Extract and normalize the first ISBN-10 or ISBN-13 in ``query``."""
        for match in _ISBN_PATTERN.finditer(query):
            isbn = cls._normalize_isbn(match.group())
            if isbn:
                return isbn
        return ""

    @staticmethod
    def _normalize_isbn(value: str) -> str:
        """Collapse ISBN separators and reject malformed lengths."""
        compact = re.sub(r"[\s-]", "", value).upper()
        if (
            len(compact) == 10
            and compact[:9].isdigit()
            and (compact[-1].isdigit() or compact[-1] == "X")
        ):
            return compact
        if len(compact) == 13 and compact.isdigit() and compact.startswith(("978", "979")):
            return compact
        return ""

    @classmethod
    def _parse_isbn_results(
        cls,
        payload: object,
        max_results: int,
        selected_isbn: str,
    ) -> list[Document]:
        """Parse Crossref works and attach connector-specific ISBN metadata."""
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
                document.metadata["source_type"] = "crossref_works_isbn"
                document.metadata["isbn"] = cls._extract_item_isbn(item) or selected_isbn
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _extract_item_isbn(cls, item: dict[str, object]) -> str:
        """Return the first valid ISBN declared on a Crossref work."""
        values = item.get("ISBN")
        candidates = values if isinstance(values, list) else [values]
        for value in candidates:
            if isinstance(value, str):
                isbn = cls._normalize_isbn(value)
                if isbn:
                    return isbn
        return ""
