"""Crossref Event Data ingestion connector.

Crossref Event Data (https://www.eventdata.crossref.org) aggregates altmetric-style
events — blog posts, social mentions, Wikipedia links, and similar signals that
reference registered scholarly content. The public Query API is queried as:

``GET https://api.eventdata.crossref.org/v1/events?rows={n}&query.bibliographic={query}``

for bibliographic text, or with ``obj-id`` when the query resolves to a DOI.
Each event is normalized into a :class:`Document` so downstream RAG pipelines can
surface attention and discourse around works alongside bibliographic metadata.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

CROSSREF_EVENTS_URL = "https://api.eventdata.crossref.org/v1/events"
_DOI_PREFIX = "https://doi.org/"

_DOI_PATTERN = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\s,;<>\"']+)",
    re.IGNORECASE,
)
_TRAILING_DOI_CHARS = ".,;:)]}"
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class CrossrefEventsConnector:
    """Search Crossref Event Data and normalize events into documents."""

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
        """Return normalized documents for Crossref Event Data matches.

        Args:
            query: Free-text bibliographic query or a DOI (with optional URL
                prefix). DOI-shaped queries use ``obj-id``; otherwise
                ``query.bibliographic`` is used.
            max_results: Maximum number of events to return.

        Returns:
            Normalized event documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        params: dict[str, str | int] = {"rows": max_results}
        if self._mailto:
            params["mailto"] = self._mailto

        doi = self._extract_doi(stripped)
        if doi:
            params["obj-id"] = doi
        else:
            params["query.bibliographic"] = stripped

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_results(payload, max_results)

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch Crossref Event Data, returning an empty payload on API failure."""
        try:
            response = await client.get(CROSSREF_EVENTS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an Event Data search payload into documents."""
        events = cls._extract_events(payload)
        documents: list[Document] = []
        for item in events:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _extract_events(cls, payload: object) -> list[object]:
        """Return the event list from an Event Data API payload."""
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []

        message = payload.get("message")
        if isinstance(message, list):
            return message
        if isinstance(message, dict):
            items = message.get("items")
            if isinstance(items, list):
                return items

        items = payload.get("items")
        if isinstance(items, list):
            return items
        return []

    @classmethod
    def _build_document(cls, event: dict[str, object]) -> Document | None:
        """Build a document from one Crossref Event Data record."""
        event_id = cls._as_str(event.get("id")).strip()
        subj_id = cls._as_str(event.get("subj_id")).strip()
        obj_id = cls._as_str(event.get("obj_id")).strip()
        relation_type = cls._as_str(event.get("relation_type_id")).strip()
        source_id = cls._as_str(event.get("source_id")).strip()
        occurred_at = cls._as_str(event.get("occurred_at")).strip()
        timestamp = cls._as_str(event.get("timestamp")).strip()
        evidence_record = cls._as_str(event.get("evidence-record")).strip()

        subj = event.get("subj")
        subj_meta = subj if isinstance(subj, dict) else {}
        obj = event.get("obj")
        obj_meta = obj if isinstance(obj, dict) else {}

        subj_title = cls._as_str(subj_meta.get("title")).strip()
        obj_doi = cls._normalize_doi(obj_id) or cls._normalize_doi(
            cls._as_str(obj_meta.get("pid")).strip()
        )
        subj_url = cls._as_str(subj_meta.get("url")).strip() or subj_id
        obj_url = cls._as_str(obj_meta.get("url")).strip() or obj_id

        title = subj_title or cls._fallback_title(relation_type, source_id, obj_doi)
        if not title:
            return None

        source = (
            evidence_record
            or subj_url
            or event_id
            or (f"{_DOI_PREFIX}{obj_doi}" if obj_doi else title)
        )
        year = cls._extract_year(occurred_at) or cls._extract_year(
            cls._as_str(subj_meta.get("issued")).strip()
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=cls._build_text(
                relation_type=relation_type,
                source_id=source_id,
                occurred_at=occurred_at,
                obj_doi=obj_doi,
                subj_url=subj_url,
                obj_url=obj_url,
            ),
            source=source,
            metadata={
                "source_type": "crossref_events",
                "event_id": event_id,
                "relation_type": relation_type,
                "source_id": source_id,
                "subj_id": subj_id,
                "obj_id": obj_id,
                "obj_doi": obj_doi,
                "subj_title": subj_title,
                "subj_url": subj_url,
                "obj_url": obj_url,
                "occurred_at": occurred_at,
                "timestamp": timestamp,
                "year": year,
                "evidence_record": evidence_record,
            },
        )

    @staticmethod
    def _extract_doi(query: str) -> str:
        """Extract a single DOI when the query is DOI-shaped."""
        match = _DOI_PATTERN.search(query)
        if match is None:
            return ""
        return match.group(1).strip().rstrip(_TRAILING_DOI_CHARS)

    @staticmethod
    def _normalize_doi(value: str) -> str:
        """Normalize a DOI URI or bare DOI to a bare DOI string."""
        if not value:
            return ""
        doi = value.strip()
        if doi.lower().startswith(_DOI_PREFIX):
            doi = doi[len(_DOI_PREFIX) :]
        return doi.strip()

    @staticmethod
    def _fallback_title(relation_type: str, source_id: str, obj_doi: str) -> str:
        """Synthesize a title when subject metadata omits one."""
        parts: list[str] = []
        if relation_type:
            parts.append(relation_type.replace("_", " "))
        if source_id:
            parts.append(f"on {source_id}")
        if obj_doi:
            parts.append(f"for DOI {obj_doi}")
        return " ".join(parts)

    @staticmethod
    def _extract_year(timestamp: str) -> str:
        """Extract a four-digit year from an ISO timestamp."""
        match = _YEAR_PREFIX_PATTERN.match(timestamp.strip())
        return match.group(1) if match else ""

    @staticmethod
    def _build_text(
        relation_type: str,
        source_id: str,
        occurred_at: str,
        obj_doi: str,
        subj_url: str,
        obj_url: str,
    ) -> str:
        """Compose searchable text summarizing one Crossref event."""
        parts: list[str] = ["Crossref Event Data mention."]
        if relation_type:
            parts.append(f"Relation: {relation_type.replace('_', ' ')}.")
        if source_id:
            parts.append(f"Source: {source_id}.")
        if occurred_at:
            parts.append(f"Occurred at {occurred_at}.")
        if obj_doi:
            parts.append(f"Object DOI {obj_doi}.")
        if subj_url:
            parts.append(f"Subject URL: {subj_url}.")
        if obj_url and obj_url != subj_url:
            parts.append(f"Object URL: {obj_url}.")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar Event Data values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
