"""Crossref members ingestion connector.

Crossref members are scholarly publishers and content registrants indexed in the
Crossref REST API. Searching ``GET https://api.crossref.org/members`` returns
matching members with stable member ids, primary names, locations, DOI prefix
lists, and registration counts — useful context for publisher-aware RAG alongside
Crossref works, funders, and OpenAlex.

Free-text queries use:

``GET https://api.crossref.org/members?query=...&rows={n}``

Member-id-shaped queries (bare numeric ids such as ``78``) resolve a single
member via:

``GET https://api.crossref.org/members/{id}``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

CROSSREF_MEMBERS_URL = "https://api.crossref.org/members"
_BARE_MEMBER_ID_PATTERN = re.compile(r"^\d+$")


class CrossrefMembersConnector:
    """Search Crossref members and normalize registrants into documents."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional contact email added to requests so Crossref routes
                traffic to its faster, polite API pool. When omitted,
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
        """Return normalized Crossref member documents matching a query.

        Args:
            query: Free-text member name search or a bare Crossref member id.
            max_results: Maximum number of members to return for free-text
                search (ignored for single-id lookups beyond returning one).

        Returns:
            Normalized member documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        member_id = self._extract_member_id(stripped)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            if member_id is not None:
                payload = await self._fetch_member(client, member_id)
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
        """Fetch a Crossref members search payload, returning {} on API failure."""
        try:
            response = await client.get(CROSSREF_MEMBERS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    async def _fetch_member(self, client: httpx.AsyncClient, member_id: str) -> object:
        """Fetch one member by id, returning {} on failure."""
        params: dict[str, str] = {}
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            response = await client.get(f"{CROSSREF_MEMBERS_URL}/{member_id}", params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Crossref ``member-list`` JSON payload into documents."""
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
        """Parse a single Crossref ``member`` JSON payload into documents."""
        if not isinstance(payload, dict):
            return []
        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        document = cls._build_document(message)
        return [document] if document is not None else []

    @classmethod
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one Crossref member object."""
        name = cls._as_str(item.get("primary-name")).strip()
        if not name:
            return None

        member_id = cls._as_str(item.get("id")).strip()
        location = cls._as_str(item.get("location")).strip()
        prefixes = cls._extract_prefixes(item.get("prefixes"))
        alt_names = cls._extract_names(item.get("names"))
        total_dois = cls._as_count(cls._nested_count(item.get("counts"), "total-dois"))
        source = cls._resolve_source(member_id, name)
        text = cls._build_descriptor(
            name=name,
            location=location,
            prefixes=prefixes,
            alt_names=alt_names,
            total_dois=total_dois,
        )
        metadata = {
            "source_type": "crossref_members",
            "member_id": member_id,
            "location": location,
            "prefixes": ", ".join(prefixes[:12]),
            "alt_names": ", ".join(alt_names[:8]),
            "total_dois": total_dois,
        }
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(name.split()),
            text=text,
            source=source,
            metadata=metadata,
        )

    @classmethod
    def _extract_member_id(cls, query: str) -> str | None:
        """Extract a Crossref member id from a query when present."""
        if _BARE_MEMBER_ID_PATTERN.fullmatch(query.strip()):
            return query.strip()
        return None

    @staticmethod
    def _extract_prefixes(value: object) -> list[str]:
        """Extract ordered DOI prefixes from a Crossref member field."""
        if not isinstance(value, list):
            return []
        prefixes: list[str] = []
        for entry in value:
            if isinstance(entry, str) and entry.strip():
                prefixes.append(entry.strip())
        return prefixes

    @staticmethod
    def _extract_names(value: object) -> list[str]:
        """Extract ordered alternate member names."""
        if not isinstance(value, list):
            return []
        names: list[str] = []
        for entry in value:
            if isinstance(entry, str) and entry.strip():
                normalized = " ".join(entry.split())
                if normalized not in names:
                    names.append(normalized)
        return names

    @staticmethod
    def _nested_count(value: object, key: str) -> object:
        """Return a nested count field from a Crossref counts object."""
        if isinstance(value, dict):
            return value.get(key)
        return None

    @classmethod
    def _resolve_source(cls, member_id: str, name: str) -> str:
        """Resolve the canonical source URL for a member record."""
        if member_id:
            return f"https://api.crossref.org/members/{member_id}"
        return name

    @staticmethod
    def _build_descriptor(
        *,
        name: str,
        location: str,
        prefixes: list[str],
        alt_names: list[str],
        total_dois: str,
    ) -> str:
        """Compose searchable descriptor text for a Crossref member."""
        parts = [f"Crossref member: {name}"]
        if location:
            parts.append(f"Location: {location}")
        if alt_names:
            filtered = [entry for entry in alt_names if entry != name]
            if filtered:
                parts.append("Also known as: " + ", ".join(filtered[:8]))
        if prefixes:
            parts.append("DOI prefixes: " + ", ".join(prefixes[:8]))
        if total_dois:
            parts.append(f"Registered DOIs: {total_dois}")
        return ". ".join(parts) + "."

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar Crossref field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""

    @staticmethod
    def _as_count(value: object) -> str:
        """Coerce a Crossref count field to a digit string when present."""
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, str) and value.strip().isdigit():
            return value.strip()
        return ""
