"""DataCite Event Data ingestion connector.

DataCite Event Data exposes citation, reference, usage, and other links involving
registered DOIs. This connector queries the public JSON:API endpoint at
``https://api.datacite.org/events`` and normalizes each event into a
:class:`Document`. It is distinct from ``datacite.py``, which searches the DOI
registry's bibliographic metadata.

DOI-shaped queries use the API's ``doi`` filter; other text uses its general
``query`` filter. Prefer frontier models for downstream synthesis: GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

DATACITE_EVENTS_URL = "https://api.datacite.org/events"
_PAGE_SIZE_CAP = 1000
_DOI_PATTERN = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\s,;<>\"']+)",
    re.IGNORECASE,
)
_TRAILING_DOI_CHARS = ".,;:)]}"


class DataCiteEventsConnector:
    """Search DataCite Event Data and normalize DOI-related events."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized DataCite events matching text or a DOI.

        Blank queries, non-positive limits, failed requests, and malformed
        payloads yield an empty list.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        params: dict[str, str | int] = {
            "page[size]": min(max_results, _PAGE_SIZE_CAP),
        }
        doi = self._extract_doi(stripped)
        params["doi" if doi else "query"] = doi or stripped

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_results(payload, max_results)

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch DataCite events, returning an empty payload on failure."""
        try:
            response = await client.get(DATACITE_EVENTS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a DataCite events JSON:API payload."""
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, list):
            return []

        documents: list[Document] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build one document from a DataCite event resource."""
        attributes = cls._as_dict(item.get("attributes"))
        event_id = cls._as_str(item.get("id")).strip()
        subject_id = cls._as_str(attributes.get("subj-id")).strip()
        object_id = cls._as_str(attributes.get("obj-id")).strip()
        relation_type = cls._as_str(attributes.get("relation-type-id")).strip()
        source_id = cls._as_str(attributes.get("source-id")).strip()
        if not event_id and not subject_id and not object_id:
            return None

        total = cls._as_str(attributes.get("total")).strip()
        occurred_at = cls._as_str(attributes.get("occurred-at")).strip()
        timestamp = cls._as_str(attributes.get("timestamp")).strip()
        subtype = cls._as_str(attributes.get("subtype")).strip()
        citation_type = cls._as_str(attributes.get("citation-type")).strip()
        subject_doi = cls._normalize_doi(subject_id)
        object_doi = cls._normalize_doi(object_id)
        source = f"{DATACITE_EVENTS_URL}/{event_id}" if event_id else subject_id or object_id
        title = cls._build_title(relation_type, subject_doi or subject_id, object_doi or object_id)

        return Document(
            document_id=stable_id(source, "doc"),
            title=title,
            text=cls._build_text(
                relation_type=relation_type,
                source_id=source_id,
                subject_id=subject_id,
                object_id=object_id,
                total=total,
                occurred_at=occurred_at,
                subtype=subtype,
                citation_type=citation_type,
            ),
            source=source,
            metadata={
                "source_type": "datacite_events",
                "event_id": event_id,
                "subject_id": subject_id,
                "object_id": object_id,
                "subject_doi": subject_doi,
                "object_doi": object_doi,
                "relation_type": relation_type,
                "event_source": source_id,
                "total": total,
                "occurred_at": occurred_at,
                "timestamp": timestamp,
                "subtype": subtype,
                "citation_type": citation_type,
            },
        )

    @staticmethod
    def _build_title(relation_type: str, subject: str, object_: str) -> str:
        """Synthesize a readable title for an event."""
        relation = relation_type.replace("-", " ") or "related event"
        endpoints = " and ".join(value for value in (subject, object_) if value)
        return f"DataCite {relation}: {endpoints}" if endpoints else f"DataCite {relation}"

    @staticmethod
    def _build_text(
        *,
        relation_type: str,
        source_id: str,
        subject_id: str,
        object_id: str,
        total: str,
        occurred_at: str,
        subtype: str,
        citation_type: str,
    ) -> str:
        """Compose searchable text from event attributes."""
        parts = ["DataCite Event Data record."]
        if relation_type:
            parts.append(f"Relation: {relation_type}.")
        if source_id:
            parts.append(f"Event source: {source_id}.")
        if subject_id:
            parts.append(f"Subject: {subject_id}.")
        if object_id:
            parts.append(f"Object: {object_id}.")
        if total:
            parts.append(f"Total: {total}.")
        if occurred_at:
            parts.append(f"Occurred at {occurred_at}.")
        if subtype:
            parts.append(f"Subtype: {subtype}.")
        if citation_type:
            parts.append(f"Citation type: {citation_type}.")
        return " ".join(parts)

    @staticmethod
    def _extract_doi(query: str) -> str:
        """Extract a DOI only when the query consists of a DOI form."""
        match = _DOI_PATTERN.fullmatch(query.strip())
        return match.group(1).rstrip(_TRAILING_DOI_CHARS) if match else ""

    @staticmethod
    def _normalize_doi(value: str) -> str:
        """Return a bare DOI for DOI URLs and an empty string otherwise."""
        match = _DOI_PATTERN.fullmatch(value.strip())
        return match.group(1).rstrip(_TRAILING_DOI_CHARS) if match else ""

    @staticmethod
    def _as_dict(value: object) -> dict[str, object]:
        """Return a dict value or an empty dict."""
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar event values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else str(value)
        return ""
