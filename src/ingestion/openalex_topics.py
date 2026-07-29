"""OpenAlex topics ingestion connector.

OpenAlex (https://openalex.org) exposes a topics taxonomy that clusters
scholarly works into research themes with human-readable names,
descriptions, and coverage statistics. This connector searches the public
topics API and normalizes each hit into a :class:`Document` for taxonomy-aware
literature discovery.

Free-text queries use:

``GET https://api.openalex.org/topics?search=...``

When the query is an OpenAlex topic id (``T####``), the connector resolves
the topic directly and also queries representative works via:

``GET https://api.openalex.org/works?filter=topics.id:T####``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_TOPICS_URL = "https://api.openalex.org/topics"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_PAGE_SIZE_CAP = 200
_TOPIC_ID_PATTERN = re.compile(r"^T\d+$", re.IGNORECASE)


class OpenAlexTopicsConnector:
    """Search OpenAlex topics and normalize matching records."""

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
        """Return normalized documents for matching OpenAlex topics.

        Args:
            query: Free-text topic search or an OpenAlex topic id such as
                ``T11948``.
            max_results: Maximum number of topic documents to return.

        Returns:
            Normalized topic documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        topic_id = self._normalize_topic_id(stripped)
        if topic_id is not None:
            return await self._search_by_topic_id(topic_id, max_results)

        params: dict[str, str | int] = {
            "search": stripped,
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, OPENALEX_TOPICS_URL, params)
        return self._parse_topic_results(payload, max_results)

    async def _search_by_topic_id(self, topic_id: str, max_results: int) -> list[Document]:
        """Resolve one topic id and optionally enrich with filtered works."""
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            topic_payload = await self._fetch_payload(
                client,
                f"{OPENALEX_TOPICS_URL}/{topic_id}",
                params,
            )
            documents = self._parse_topic_results(
                {"results": [topic_payload]} if isinstance(topic_payload, dict) else {},
                max_results=1,
            )
            if not documents:
                return []

            works_params: dict[str, str | int] = {
                "filter": f"topics.id:{topic_id}",
                "per-page": min(max_results, _PAGE_SIZE_CAP),
            }
            if self._mailto:
                works_params["mailto"] = self._mailto
            works_payload = await self._fetch_payload(client, OPENALEX_WORKS_URL, works_params)
            work_titles = self._extract_work_titles(works_payload, max_results)
            if work_titles:
                document = documents[0]
                enriched_text = document.text
                sample = "; ".join(work_titles)
                if sample:
                    enriched_text = f"{enriched_text} Sample works: {sample}.".strip()
                documents[0] = Document(
                    document_id=document.document_id,
                    title=document.title,
                    text=enriched_text,
                    source=document.source,
                    metadata={
                        **document.metadata,
                        "sample_work_titles": sample,
                    },
                )
            return documents[:max_results]

    @staticmethod
    def _normalize_topic_id(query: str) -> str | None:
        """Return a bare OpenAlex topic id when ``query`` is topic-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://openalex.org/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _TOPIC_ID_PATTERN.fullmatch(candidate):
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
    def _parse_topic_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex topics payload into documents."""
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
            document = cls._build_topic_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_topic_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one OpenAlex topic record."""
        display_name = cls._as_str(item.get("display_name")).strip()
        if not display_name:
            return None

        description = cls._as_str(item.get("description")).strip()
        works_count = cls._as_str(item.get("works_count")).strip()
        topic_id = cls._extract_topic_id(item.get("id"))
        source = cls._as_str(item.get("id")).strip() or (
            f"https://openalex.org/{topic_id}" if topic_id else display_name
        )
        text = description or cls._build_text(display_name, works_count)

        subfield = cls._nested_display_name(item.get("subfield"))
        field = cls._nested_display_name(item.get("field"))
        domain = cls._nested_display_name(item.get("domain"))

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openalex_topics",
                "topic_id": topic_id,
                "description": description,
                "works_count": works_count,
                "subfield": subfield,
                "field": field,
                "domain": domain,
            },
        )

    @classmethod
    def _extract_work_titles(cls, payload: object, max_results: int) -> list[str]:
        """Return titles from an OpenAlex works payload."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        titles: list[str] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = cls._as_str(item.get("title") or item.get("display_name")).strip()
            if title:
                titles.append(title)
            if len(titles) >= max_results:
                break
        return titles

    @staticmethod
    def _extract_topic_id(value: object) -> str:
        """Normalize an OpenAlex topic id URL or bare id."""
        topic_ref = OpenAlexTopicsConnector._as_str(value).strip()
        if not topic_ref:
            return ""
        if topic_ref.lower().startswith("https://openalex.org/"):
            return topic_ref.rsplit("/", maxsplit=1)[-1].upper()
        return topic_ref.upper()

    @staticmethod
    def _nested_display_name(value: object) -> str:
        """Return ``display_name`` from an OpenAlex nested object."""
        if isinstance(value, dict):
            return OpenAlexTopicsConnector._as_str(value.get("display_name")).strip()
        return ""

    @staticmethod
    def _build_text(display_name: str, works_count: str) -> str:
        """Compose searchable text when a topic has no description."""
        if works_count:
            return f"OpenAlex topic {display_name} with {works_count} works."
        return f"OpenAlex topic {display_name}."

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar OpenAlex values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
