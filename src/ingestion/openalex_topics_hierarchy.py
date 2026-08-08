"""OpenAlex topics hierarchy ingestion connector.

OpenAlex topics nest under subfield → field → domain. Popular scholarly RAG
pipelines (paper-qa, llama-index OpenAlex loaders) often need the full taxonomy
path, not just the leaf topic. This connector searches topics and normalizes
hierarchy ancestry into :class:`Document` records for synthesis with GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_TOPICS_URL = "https://api.openalex.org/topics"
_PAGE_SIZE_CAP = 200
_TOPIC_ID_PATTERN = re.compile(r"^T\d+$", re.IGNORECASE)


class OpenAlexTopicsHierarchyConnector:
    """Search OpenAlex topics and normalize hierarchy ancestry."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional contact email for OpenAlex's polite pool.
        """
        self._mailto = (
            mailto
            or os.environ.get("OPENALEX_MAILTO", "").strip()
            or os.environ.get("UNPAYWALL_EMAIL", "").strip()
            or None
        )

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized documents with topic hierarchy paths.

        Args:
            query: Free-text topic search or an OpenAlex topic id ``T####``.
            max_results: Maximum number of topic documents to return.

        Returns:
            Normalized topic hierarchy documents.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        topic_id = self._normalize_topic_id(stripped)
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                if topic_id is not None:
                    response = await client.get(f"{OPENALEX_TOPICS_URL}/{topic_id}", params=params)
                    response.raise_for_status()
                    payload = {"results": [response.json()]}
                else:
                    params = {
                        "search": stripped,
                        "per-page": min(max_results, _PAGE_SIZE_CAP),
                        **({"mailto": self._mailto} if self._mailto else {}),
                    }
                    response = await client.get(OPENALEX_TOPICS_URL, params=params)
                    response.raise_for_status()
                    payload = response.json()
        except (httpx.HTTPError, ValueError):
            return []

        return self._parse_results(payload, max_results)

    @staticmethod
    def _normalize_topic_id(value: str) -> str | None:
        """Return a normalized topic id when the query is ``T####`` shaped."""
        candidate = value.strip()
        if candidate.lower().startswith("https://openalex.org/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _TOPIC_ID_PATTERN.fullmatch(candidate):
            return candidate.upper()
        return None

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse OpenAlex topics payloads into hierarchy documents."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            results = [payload] if isinstance(payload.get("id"), str) else []
        documents: list[Document] = []
        for item in results:
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
        """Build a hierarchy document from one OpenAlex topic record."""
        display_name = str(item.get("display_name") or "").strip()
        if not display_name:
            return None
        topic_id = cls._extract_id(item.get("id"))
        description = str(item.get("description") or "").strip()
        domain = cls._nested_name(item.get("domain"))
        field = cls._nested_name(item.get("field"))
        subfield = cls._nested_name(item.get("subfield"))
        hierarchy_path = " > ".join(
            part for part in (domain, field, subfield, display_name) if part
        )
        works_count = str(item.get("works_count") or "").strip()
        source = str(item.get("id") or "").strip() or (
            f"https://openalex.org/{topic_id}" if topic_id else display_name
        )
        text_parts = [f"OpenAlex topic hierarchy: {hierarchy_path}."]
        if description:
            text_parts.append(description)
        if works_count:
            text_parts.append(f"Works count: {works_count}.")
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=" ".join(text_parts),
            source=source,
            metadata={
                "source_type": "openalex_topics_hierarchy",
                "openalex_topic_id": topic_id,
                "domain": domain,
                "field": field,
                "subfield": subfield,
                "hierarchy_path": hierarchy_path,
                "works_count": works_count,
            },
        )

    @staticmethod
    def _extract_id(value: object) -> str:
        """Normalize an OpenAlex topic id URL or bare id."""
        ref = str(value or "").strip()
        if not ref:
            return ""
        if ref.lower().startswith("https://openalex.org/"):
            return ref.rsplit("/", maxsplit=1)[-1].upper()
        return ref.upper()

    @staticmethod
    def _nested_name(value: object) -> str:
        """Extract display_name from a nested OpenAlex taxonomy object."""
        if isinstance(value, dict):
            return str(value.get("display_name") or "").strip()
        return ""
