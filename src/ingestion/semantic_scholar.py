"""Semantic Scholar API ingestion connector."""

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarConnector:
    """Fetch and normalize Semantic Scholar paper records."""

    def __init__(self, api_key: str | None = None) -> None:
        """Create a connector with an optional API key."""
        self._api_key = api_key

    async def fetch_paper(self, paper_id: str) -> Document:
        """Return one normalized Semantic Scholar paper by id."""
        headers = {"x-api-key": self._api_key} if self._api_key else None
        params = {"fields": "title,abstract,year,authors,url"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{S2_BASE_URL}/paper/{paper_id}", params=params, headers=headers
            )
            response.raise_for_status()
        data = response.json()
        title = str(data.get("title") or "Untitled Semantic Scholar paper")
        abstract = str(data.get("abstract") or "")
        source = str(data.get("url") or paper_id)
        return Document(
            document_id=stable_id(source, "doc"),
            title=title,
            text=abstract,
            source=source,
            metadata={"source_type": "semantic_scholar", "year": str(data.get("year") or "")},
        )
