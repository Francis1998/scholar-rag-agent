"""OpenAIRE funded-projects ingestion connector.

OpenAIRE (https://www.openaire.eu) aggregates funded research projects across
European and international funder registries. This connector is projects-only:
it queries the public ``search/projects`` endpoint and normalizes each project
(title, summary abstract, grant code, funder, dates) into a :class:`Document`.
It complements ``openaire.py``, which indexes research products rather than
grant records.

Free-text queries use keyword search:

``GET https://api.openaire.eu/search/projects?format=json&keywords=...``

Title-shaped queries use the supported ``name`` parameter:

``GET https://api.openaire.eu/search/projects?format=json&name=...``

Grant-shaped queries resolve via ``grantID`` or ``openaireProjectID`` when the
query carries a funder-scoped project identifier.
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENAIRE_PROJECTS_URL = "https://api.openaire.eu/search/projects"
_PAGE_SIZE_CAP = 100
_OPENAIRE_PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9_]+::.+$", re.IGNORECASE)
_GRANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,}$")


class OpenaireProjectsConnector:
    """Search OpenAIRE projects and normalize matching grant records."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized documents for matching OpenAIRE projects.

        Args:
            query: Free-text keywords, a project title phrase, or a grant /
                OpenAIRE project identifier.
            max_results: Maximum number of project documents to return.

        Returns:
            Normalized project documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        params = self._build_search_params(stripped, max_results)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_results(payload, max_results)

    @classmethod
    def _build_search_params(cls, query: str, max_results: int) -> dict[str, str | int]:
        """Build OpenAIRE project search parameters for a query shape."""
        size = min(max_results, _PAGE_SIZE_CAP)
        if _OPENAIRE_PROJECT_ID_PATTERN.fullmatch(query):
            return {
                "format": "json",
                "openaireProjectID": query,
                "page": 1,
                "size": size,
            }
        if _GRANT_ID_PATTERN.fullmatch(query) and " " not in query:
            return {
                "format": "json",
                "grantID": query,
                "page": 1,
                "size": size,
            }
        if len(query.split()) >= 3:
            return {
                "format": "json",
                "name": query,
                "page": 1,
                "size": size,
            }
        return {
            "format": "json",
            "keywords": query,
            "page": 1,
            "size": size,
        }

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch the OpenAIRE projects endpoint, returning {} on failure."""
        try:
            response = await client.get(OPENAIRE_PROJECTS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAIRE projects JSON payload into documents."""
        if not isinstance(payload, dict):
            return []
        response = payload.get("response")
        if not isinstance(response, dict):
            return []
        results = response.get("results")
        if not isinstance(results, dict):
            return []
        result_items = results.get("result")
        if not isinstance(result_items, list):
            return []

        documents: list[Document] = []
        for item in result_items:
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
        """Build a document from one OpenAIRE project search result."""
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            return None
        entity = metadata.get("oaf:entity")
        if not isinstance(entity, dict):
            return None
        project = entity.get("oaf:project")
        if not isinstance(project, dict):
            return None

        title = cls._unwrap_text(project.get("title")).strip()
        if not title:
            return None

        abstract = cls._unwrap_text(project.get("summary")).strip()
        keywords = cls._unwrap_text(project.get("keywords")).strip()
        code = cls._unwrap_text(project.get("code")).strip()
        original_id = cls._unwrap_text(project.get("originalId")).strip()
        start_date = cls._unwrap_text(project.get("startdate")).strip()
        end_date = cls._unwrap_text(project.get("enddate")).strip()
        funded_amount = cls._unwrap_text(project.get("fundedamount")).strip()
        funder = cls._extract_funder_name(project.get("fundingtree"))

        project_id = original_id or code
        source = project_id or title
        text = abstract or cls._build_descriptor(title, keywords, funder, start_date, end_date)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openaire_projects",
                "project_id": project_id,
                "code": code,
                "abstract": abstract,
                "keywords": keywords,
                "funder": funder,
                "start_date": start_date,
                "end_date": end_date,
                "funded_amount": funded_amount,
            },
        )

    @classmethod
    def _extract_funder_name(cls, fundingtree: object) -> str:
        """Return the primary funder name from an OpenAIRE funding tree."""
        if not isinstance(fundingtree, dict):
            return ""
        funder = fundingtree.get("funder")
        if not isinstance(funder, dict):
            return ""
        return cls._unwrap_text(funder.get("name")).strip()

    @classmethod
    def _unwrap_text(cls, value: object) -> str:
        """Unwrap OpenAIRE ``{\"$\": \"...\"}`` text nodes and coerce scalars."""
        if isinstance(value, dict):
            wrapped = value.get("$")
            if isinstance(wrapped, str):
                return wrapped
            if isinstance(wrapped, int) and not isinstance(wrapped, bool):
                return str(wrapped)
            if isinstance(wrapped, float) and wrapped.is_integer():
                return str(int(wrapped))
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return ""

    @staticmethod
    def _build_descriptor(
        title: str,
        keywords: str,
        funder: str,
        start_date: str,
        end_date: str,
    ) -> str:
        """Compose searchable descriptor text when a project has no summary."""
        parts = [f"OpenAIRE project: {title}."]
        if funder:
            parts.append(f"Funder: {funder}.")
        if keywords:
            parts.append(f"Keywords: {keywords}.")
        if start_date or end_date:
            parts.append(f"Dates: {start_date or '?'} to {end_date or '?'}.")
        return " ".join(parts)
