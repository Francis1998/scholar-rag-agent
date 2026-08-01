"""OpenAlex concepts ingestion connector.

OpenAlex (https://openalex.org) exposes a legacy concepts taxonomy that clusters
scholarly works into research themes with human-readable names, descriptions,
Wikidata links, and coverage statistics. This connector searches the public
concepts API and normalizes each hit into a :class:`Document` for concept-aware
literature discovery.

Free-text queries use:

``GET https://api.openalex.org/concepts?search=...``

When the query is an OpenAlex concept id (``C####``), the connector resolves the
concept directly and also queries representative works via:

``GET https://api.openalex.org/works?filter=concepts.id:{id}``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_CONCEPTS_URL = "https://api.openalex.org/concepts"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_PAGE_SIZE_CAP = 200
_CONCEPT_ID_PATTERN = re.compile(r"^C\d+$", re.IGNORECASE)


class OpenAlexConceptsConnector:
    """Search OpenAlex concepts and normalize matching records."""

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
        """Return normalized documents for matching OpenAlex concepts.

        Args:
            query: Free-text concept search or an OpenAlex concept id such as
                ``C119857082``.
            max_results: Maximum number of concept documents to return.

        Returns:
            Normalized concept documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        concept_id = self._normalize_concept_id(stripped)
        if concept_id is not None:
            return await self._search_by_concept_id(concept_id, max_results)

        params: dict[str, str | int] = {
            "search": stripped,
            "per-page": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, OPENALEX_CONCEPTS_URL, params)
        return self._parse_concept_results(payload, max_results)

    async def _search_by_concept_id(self, concept_id: str, max_results: int) -> list[Document]:
        """Resolve one concept id and optionally enrich with filtered works."""
        params: dict[str, str | int] = {}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            concept_payload = await self._fetch_payload(
                client,
                f"{OPENALEX_CONCEPTS_URL}/{concept_id}",
                params,
            )
            documents = self._parse_concept_results(
                {"results": [concept_payload]} if isinstance(concept_payload, dict) else {},
                max_results=1,
            )
            if not documents:
                return []

            numeric_id = concept_id[1:] if concept_id.startswith("C") else concept_id
            works_params: dict[str, str | int] = {
                "filter": f"concepts.id:{numeric_id}",
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
    def _parse_concept_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAlex concepts payload into documents."""
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
            document = cls._build_concept_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_concept_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one OpenAlex concept record."""
        display_name = cls._as_str(item.get("display_name")).strip()
        if not display_name:
            return None

        description = cls._as_str(item.get("description")).strip()
        works_count = cls._as_str(item.get("works_count")).strip()
        cited_by_count = cls._as_str(item.get("cited_by_count")).strip()
        level = cls._as_str(item.get("level")).strip()
        concept_id = cls._extract_concept_id(item.get("id"))
        wikidata = cls._extract_wikidata(item.get("wikidata"))
        source = cls._as_str(item.get("id")).strip() or (
            f"https://openalex.org/{concept_id}" if concept_id else display_name
        )
        text = description or cls._build_text(display_name, works_count, cited_by_count)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(display_name.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openalex_concepts",
                "concept_id": concept_id,
                "description": description,
                "works_count": works_count,
                "cited_by_count": cited_by_count,
                "level": level,
                "wikidata": wikidata,
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
    def _extract_concept_id(value: object) -> str:
        """Normalize an OpenAlex concept id URL or bare id."""
        concept_ref = OpenAlexConceptsConnector._as_str(value).strip()
        if not concept_ref:
            return ""
        if concept_ref.lower().startswith("https://openalex.org/"):
            return concept_ref.rsplit("/", maxsplit=1)[-1].upper()
        return concept_ref.upper()

    @staticmethod
    def _extract_wikidata(value: object) -> str:
        """Return a bare Wikidata Q-id from an OpenAlex Wikidata URL or bare id."""
        wikidata_ref = OpenAlexConceptsConnector._as_str(value).strip()
        if not wikidata_ref:
            return ""
        if wikidata_ref.lower().startswith("https://www.wikidata.org/wiki/"):
            return wikidata_ref.rsplit("/", maxsplit=1)[-1]
        return wikidata_ref

    @staticmethod
    def _build_text(display_name: str, works_count: str, cited_by_count: str) -> str:
        """Compose searchable text when a concept has no description."""
        parts = [f"OpenAlex concept {display_name}"]
        if works_count:
            parts.append(f"with {works_count} works")
        if cited_by_count:
            parts.append(f"cited by count {cited_by_count}")
        return " ".join(parts) + "."

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar OpenAlex values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
