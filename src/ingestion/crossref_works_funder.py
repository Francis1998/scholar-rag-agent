"""Crossref works-by-funder ingestion connector.

Crossref works can be filtered by Open Funder Registry id, linking funded
outputs to funders acknowledged in Crossref metadata. This connector queries:

``GET https://api.crossref.org/works?filter=funder:{id}&query=...``

When the query is funder-id-shaped (``10.13039/...``, a bare registry id, or a
DOI URL), the connector applies ``filter=funder:{id}``. Free-text queries use
``query`` together with ``filter=has-funder:true`` so only funded works are
returned. Distinct from ``crossref_funder.py``, which resolves funder registry
entities rather than funded works.

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import html
import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

CROSSREF_WORKS_URL = "https://api.crossref.org/works"
_JATS_TAG_PATTERN = re.compile(r"<[^>]+>")
_FUNDER_DOI_PATTERN = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.13039/\d+)",
    re.IGNORECASE,
)
_BARE_FUNDER_ID_PATTERN = re.compile(r"^\d{4,}$")


class CrossrefWorksFunderConnector:
    """Search Crossref works filtered by funder acknowledgement."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional contact email for Crossref's polite pool. When
                omitted, ``CROSSREF_MAILTO`` (then ``OPENALEX_MAILTO``) is read
                from the environment when present.
        """
        self._mailto = (
            mailto
            or os.environ.get("CROSSREF_MAILTO", "").strip()
            or os.environ.get("OPENALEX_MAILTO", "").strip()
            or None
        )

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return funded Crossref works matching a free-text or funder-id query.

        Blank queries, non-positive limits, failed requests, and malformed
        payloads yield an empty list.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        funder_id = self._extract_funder_id(stripped)
        params: dict[str, str | int] = {"rows": max_results}
        if funder_id is not None:
            params["filter"] = f"funder:{funder_id}"
        else:
            params["query"] = stripped
            params["filter"] = "has-funder:true"
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_results(payload, max_results, funder_id or "")

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch Crossref works, returning an empty payload on failure."""
        try:
            response = await client.get(CROSSREF_WORKS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(
        cls,
        payload: object,
        max_results: int,
        funder_id: str,
    ) -> list[Document]:
        """Parse a Crossref works payload into funder-tagged documents."""
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
            document = cls._build_document(item, funder_id)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object], funder_id: str) -> Document | None:
        """Build a document from one Crossref work with funding metadata."""
        title = cls._first_string(item.get("title")).strip()
        doi = cls._as_str(item.get("DOI")).strip()
        if not title and not doi:
            return None
        if not title:
            title = f"Crossref work {doi}" if doi else "Untitled Crossref work"

        source = f"https://doi.org/{doi}" if doi else title
        abstract = cls._strip_jats(item.get("abstract"))
        year = cls._resolve_year(item)
        authors = cls._extract_authors(item.get("author"))
        funders = cls._extract_funders(item.get("funder"))
        resolved_funder = funder_id or (funders[0] if funders else "")
        text = abstract or cls._build_descriptor(
            authors=authors,
            year=year,
            doi=doi,
            funders=funders,
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "crossref_works_funder",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "funder_id": resolved_funder,
                "funders": ", ".join(funders),
            },
        )

    @classmethod
    def _extract_funder_id(cls, query: str) -> str | None:
        """Extract an Open Funder Registry id when the query is funder-shaped."""
        doi_match = _FUNDER_DOI_PATTERN.fullmatch(query.strip())
        if doi_match:
            return doi_match.group(1)
        if _BARE_FUNDER_ID_PATTERN.fullmatch(query):
            return query
        return None

    @classmethod
    def _extract_authors(cls, value: object) -> list[str]:
        """Extract ordered author names from Crossref ``author`` objects."""
        if not isinstance(value, list):
            return []
        names: list[str] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            given = cls._as_str(entry.get("given")).strip()
            family = cls._as_str(entry.get("family")).strip()
            name = " ".join(part for part in (given, family) if part)
            if not name:
                name = cls._as_str(entry.get("name")).strip()
            if name:
                names.append(name)
        return names

    @classmethod
    def _extract_funders(cls, value: object) -> list[str]:
        """Extract funder names or DOI ids from a Crossref ``funder`` list."""
        if not isinstance(value, list):
            return []
        funders: list[str] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            name = cls._as_str(entry.get("name")).strip()
            funder_doi = cls._as_str(entry.get("DOI")).strip()
            label = name or funder_doi
            if label:
                funders.append(label)
        return funders

    @staticmethod
    def _build_descriptor(
        *,
        authors: list[str],
        year: str,
        doi: str,
        funders: list[str],
    ) -> str:
        """Compose searchable text when Crossref omits an abstract."""
        parts = ["Crossref funded work."]
        if authors:
            parts.append(f"Authors: {', '.join(authors)}.")
        if year:
            parts.append(f"Year: {year}.")
        if funders:
            parts.append(f"Funders: {', '.join(funders)}.")
        if doi:
            parts.append(f"DOI: {doi}.")
        return " ".join(parts)

    @staticmethod
    def _first_string(value: object) -> str:
        """Return the first non-empty string in a Crossref list field."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str) and entry.strip():
                    return entry
        return ""

    @classmethod
    def _strip_jats(cls, abstract: object) -> str:
        """Strip JATS XML markup from a Crossref abstract."""
        if not isinstance(abstract, str) or not abstract.strip():
            return ""
        without_tags = _JATS_TAG_PATTERN.sub(" ", abstract)
        return " ".join(html.unescape(without_tags).split())

    @classmethod
    def _resolve_year(cls, item: dict[str, object]) -> str:
        """Resolve publication year from Crossref date fields."""
        for field in ("published", "issued", "published-print", "published-online"):
            year = cls._extract_year(item.get(field))
            if year:
                return year
        return ""

    @staticmethod
    def _extract_year(published: object) -> str:
        """Extract the year from a Crossref ``date-parts`` structure."""
        if not isinstance(published, dict):
            return ""
        date_parts = published.get("date-parts")
        if not isinstance(date_parts, list) or not date_parts:
            return ""
        first_date = date_parts[0]
        if not isinstance(first_date, list) or not first_date:
            return ""
        year = first_date[0]
        if isinstance(year, int) and not isinstance(year, bool):
            return str(year)
        return ""

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar Crossref fields to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
