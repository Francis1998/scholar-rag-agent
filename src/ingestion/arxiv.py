"""arXiv API ingestion connector."""

import httpx
from defusedxml import ElementTree

from ingestion.chunking import stable_id
from retrieval.models import Document

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"


class ArxivConnector:
    """Fetch and normalize arXiv metadata through the public API."""

    async def fetch(self, arxiv_id_or_query: str, max_results: int = 1) -> list[Document]:
        """Return normalized arXiv documents for an id or query."""
        params: dict[str, str | int] = {
            "search_query": arxiv_id_or_query,
            "start": 0,
            "max_results": max_results,
        }
        if arxiv_id_or_query.replace(".", "").isdigit() or "/" in arxiv_id_or_query:
            params = {"id_list": arxiv_id_or_query, "max_results": max_results}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        documents: list[Document] = []
        for entry in root.findall(f"{ATOM_NAMESPACE}entry"):
            title = (entry.findtext(f"{ATOM_NAMESPACE}title") or "Untitled arXiv paper").strip()
            summary = (entry.findtext(f"{ATOM_NAMESPACE}summary") or "").strip()
            entry_id = (entry.findtext(f"{ATOM_NAMESPACE}id") or title).strip()
            documents.append(
                Document(
                    document_id=stable_id(entry_id, "doc"),
                    title=" ".join(title.split()),
                    text=" ".join(summary.split()),
                    source=entry_id,
                    metadata={"source_type": "arxiv"},
                )
            )
        return documents
