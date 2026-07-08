"""arXiv API ingestion connector."""

import re

import httpx
from defusedxml import ElementTree

from ingestion.chunking import stable_id
from retrieval.models import Document

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"

# New-style arXiv identifiers are ``YYMM.NNNNN`` with an optional ``vN`` version
# suffix (e.g. ``2301.00001`` or ``2301.00001v2``). Old-style identifiers carry
# an archive prefix and a slash (e.g. ``math/0309136``) and are detected
# separately by the presence of ``/``.
_ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


class ArxivConnector:
    """Fetch and normalize arXiv metadata through the public API."""

    async def fetch(self, arxiv_id_or_query: str, max_results: int = 1) -> list[Document]:
        """Return normalized arXiv documents for an id or query."""
        identifier = arxiv_id_or_query.strip()
        params: dict[str, str | int] = {
            "search_query": arxiv_id_or_query,
            "start": 0,
            "max_results": max_results,
        }
        if _ARXIV_ID_PATTERN.match(identifier) or "/" in identifier:
            params = {"id_list": identifier, "max_results": max_results}
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
