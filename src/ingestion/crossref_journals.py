"""Crossref journals ingestion connector.

Crossref indexes scholarly journals (and other serials) via the journals
endpoint. This connector searches ``GET https://api.crossref.org/journals`` and
normalizes each hit into a :class:`Document` for venue-aware literature
discovery alongside Crossref works, members, and OpenAlex sources.

Free-text queries use:

``GET https://api.crossref.org/journals?query=...&rows={n}``

ISSN-shaped queries (for example ``1532-4435`` or ``15324435``) resolve a single
journal via:

``GET https://api.crossref.org/journals/{issn}``

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

CROSSREF_JOURNALS_URL = "https://api.crossref.org/journals"
_ISSN_PATTERN = re.compile(r"^\d{4}-?\d{3}[\dXx]$")


class CrossrefJournalsConnector:
    """Search Crossref journals and normalize matching serials into documents."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional polite-pool contact email. When omitted,
                ``CROSSREF_MAILTO`` (then ``OPENALEX_MAILTO``) is read from the
                environment when present.
        """
        self._mailto = (
            mailto
            or os.environ.get("CROSSREF_MAILTO", "").strip()
            or os.environ.get("OPENALEX_MAILTO", "").strip()
            or None
        )

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Crossref journal documents matching a query.

        Args:
            query: Free-text journal title search or an ISSN.
            max_results: Maximum number of journals to return for free-text
                search.

        Returns:
            Normalized journal documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        issn = self._normalize_issn(stripped)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            if issn is not None:
                payload = await self._fetch_journal(client, issn)
                return self._parse_single(payload)

            params: dict[str, str | int] = {"query": stripped, "rows": max_results}
            if self._mailto:
                params["mailto"] = self._mailto
            payload = await self._fetch_search(client, params)
        return self._parse_results(payload, max_results)

    async def _fetch_search(
        self,
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch a Crossref journals search payload, returning {} on failure."""
        try:
            response = await client.get(CROSSREF_JOURNALS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    async def _fetch_journal(self, client: httpx.AsyncClient, issn: str) -> object:
        """Fetch one journal by ISSN, returning {} on failure."""
        params: dict[str, str] = {}
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            response = await client.get(f"{CROSSREF_JOURNALS_URL}/{issn}", params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Crossref journal-list payload into documents."""
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
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _parse_single(cls, payload: object) -> list[Document]:
        """Parse a single Crossref journal payload."""
        if not isinstance(payload, dict):
            return []
        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        document = cls._build_document(message)
        return [document] if document is not None else []

    @classmethod
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one Crossref journal item."""
        title = cls._as_str(item.get("title")).strip()
        if not title:
            return None

        issns = cls._extract_issns(item)
        publisher = cls._as_str(item.get("publisher")).strip()
        subjects = cls._extract_subjects(item.get("subjects"))
        counts = cls._as_dict(item.get("counts"))
        total_dois = cls._as_str(counts.get("total-dois")).strip()
        current_dois = cls._as_str(counts.get("current-dois")).strip()
        primary_issn = issns[0] if issns else ""

        source = (
            f"{CROSSREF_JOURNALS_URL}/{primary_issn}"
            if primary_issn
            else f"{CROSSREF_JOURNALS_URL}"
        )
        text = cls._build_text(title, publisher, issns, subjects, total_dois, current_dois)

        return Document(
            document_id=stable_id(source if primary_issn else title, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source if primary_issn else title,
            metadata={
                "source_type": "crossref_journals",
                "issn": primary_issn,
                "issns": ",".join(issns),
                "publisher": publisher,
                "subjects": subjects,
                "total_dois": total_dois,
                "current_dois": current_dois,
            },
        )

    @staticmethod
    def _normalize_issn(query: str) -> str | None:
        """Return a hyphenated ISSN when ``query`` is ISSN-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://api.crossref.org/journals/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if not _ISSN_PATTERN.fullmatch(candidate):
            return None
        compact = candidate.replace("-", "").upper()
        return f"{compact[:4]}-{compact[4:]}"

    @classmethod
    def _extract_issns(cls, item: dict[str, object]) -> list[str]:
        """Collect ISSN values from ISSN and issn-type fields."""
        values: list[str] = []
        raw = item.get("ISSN")
        if isinstance(raw, list):
            for entry in raw:
                text = cls._as_str(entry).strip()
                if text and text not in values:
                    values.append(text)
        typed = item.get("issn-type")
        if isinstance(typed, list):
            for entry in typed:
                if not isinstance(entry, dict):
                    continue
                text = cls._as_str(entry.get("value")).strip()
                if text and text not in values:
                    values.append(text)
        return values

    @classmethod
    def _extract_subjects(cls, value: object) -> str:
        """Return a comma-joined subject list from Crossref subjects."""
        if not isinstance(value, list):
            return ""
        names: list[str] = []
        for entry in value:
            if isinstance(entry, dict):
                name = cls._as_str(entry.get("name")).strip()
            else:
                name = cls._as_str(entry).strip()
            if name and name not in names:
                names.append(name)
        return ", ".join(names)

    @staticmethod
    def _build_text(
        title: str,
        publisher: str,
        issns: list[str],
        subjects: str,
        total_dois: str,
        current_dois: str,
    ) -> str:
        """Compose searchable text for a Crossref journal profile."""
        parts: list[str] = [f"Crossref journal {title}."]
        if publisher:
            parts.append(f"Publisher: {publisher}.")
        if issns:
            parts.append(f"ISSN: {', '.join(issns)}.")
        if subjects:
            parts.append(f"Subjects: {subjects}.")
        if total_dois:
            parts.append(f"Total DOIs: {total_dois}.")
        if current_dois:
            parts.append(f"Current DOIs: {current_dois}.")
        return " ".join(parts)

    @staticmethod
    def _as_dict(value: object) -> dict[str, object]:
        """Return a dict value or an empty dict."""
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar Crossref values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return ""
