"""OpenAlex concepts ancestors hierarchy ingestion connector.

OpenAlex legacy concepts nest under ancestor concepts by ``level``. Literature
workflows often need the full concept ancestry path, not just the leaf concept.
This connector searches concepts and normalizes ancestor hierarchy into
:class:`Document` records. Distinct from ``openalex_concepts.py``, which focuses
on concept descriptions and sample works rather than ancestry paths.

Free-text queries use:

``GET https://api.openalex.org/concepts?search=...``

Concept-id shaped queries (``C####``) resolve:

``GET https://api.openalex.org/concepts/{id}``

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_CONCEPTS_URL = "https://api.openalex.org/concepts"
_PAGE_SIZE_CAP = 200
_CONCEPT_ID_PATTERN = re.compile(r"^C\d+$", re.IGNORECASE)


class OpenAlexConceptsAncestorsConnector:
    """Search OpenAlex concepts and normalize ancestor hierarchy paths."""

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
        """Return concept documents enriched with ancestor hierarchy paths.

        Blank queries, non-positive limits, failed requests, and malformed
        payloads yield an empty list.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        concept_id = self._normalize_concept_id(stripped)
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            if concept_id is not None:
                payload = await self._fetch_payload(
                    client,
                    f"{OPENALEX_CONCEPTS_URL}/{concept_id}",
                    params,
                )
                if isinstance(payload, dict) and payload.get("id"):
                    payload = {"results": [payload]}
            else:
                params = {
                    "search": stripped,
                    "per-page": min(max_results, _PAGE_SIZE_CAP),
                    **({"mailto": self._mailto} if self._mailto else {}),
                }
                payload = await self._fetch_payload(client, OPENALEX_CONCEPTS_URL, params)
        return self._parse_results(payload, max_results)

    @staticmethod
    def _normalize_concept_id(query: str) -> str | None:
        """Return a bare OpenAlex concept id when ``query`` is concept-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://openalex.org/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _CONCEPT_ID_PATTERN.fullmatch(candidate):
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
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex concepts payload into ancestor documents."""
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
        """Build a document from one OpenAlex concept with ancestor path."""
        display_name = cls._as_str(item.get("display_name")).strip()
        if not display_name:
            return None

        description = cls._as_str(item.get("description")).strip()
        works_count = cls._as_str(item.get("works_count")).strip()
        cited_by_count = cls._as_str(item.get("cited_by_count")).strip()
        level = cls._as_str(item.get("level")).strip()
        concept_id = cls._extract_concept_id(item.get("id"))
        ancestors = cls._extract_ancestors(item.get("ancestors"))
        ancestor_path = cls._build_ancestor_path(ancestors, display_name)
        source = cls._as_str(item.get("id")).strip() or (
            f"https://openalex.org/{concept_id}" if concept_id else display_name
        )
        text_parts = [f"OpenAlex concept ancestors: {ancestor_path}."]
        if description:
            text_parts.append(description)
        if works_count:
            text_parts.append(f"Works count: {works_count}.")
        if cited_by_count:
            text_parts.append(f"Cited by count: {cited_by_count}.")

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=" ".join(text_parts),
            source=source,
            metadata={
                "source_type": "openalex_concepts_ancestors",
                "concept_id": concept_id,
                "level": level,
                "works_count": works_count,
                "cited_by_count": cited_by_count,
                "ancestors": ", ".join(ancestors),
                "ancestor_path": ancestor_path,
            },
        )

    @classmethod
    def _extract_ancestors(cls, value: object) -> list[str]:
        """Return ancestor display names ordered by ascending level."""
        if not isinstance(value, list):
            return []
        ranked: list[tuple[int, str]] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            name = cls._as_str(entry.get("display_name")).strip()
            if not name:
                continue
            level_raw = entry.get("level")
            level = (
                level_raw if isinstance(level_raw, int) and not isinstance(level_raw, bool) else 99
            )
            ranked.append((level, name))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [name for _, name in ranked]

    @staticmethod
    def _build_ancestor_path(ancestors: list[str], display_name: str) -> str:
        """Compose ``ancestor > ... > leaf`` hierarchy path."""
        parts = [*ancestors, display_name]
        return " > ".join(part for part in parts if part)

    @staticmethod
    def _extract_concept_id(value: object) -> str:
        """Normalize an OpenAlex concept id URL or bare id."""
        concept_ref = OpenAlexConceptsAncestorsConnector._as_str(value).strip()
        if not concept_ref:
            return ""
        if concept_ref.lower().startswith("https://openalex.org/"):
            return concept_ref.rsplit("/", maxsplit=1)[-1].upper()
        return concept_ref.upper()

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar OpenAlex values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
