"""Semantic Scholar bulk paper-batch ingestion connector.

Semantic Scholar's Graph API exposes a batch endpoint that resolves many paper
ids in one request — a common gap vs single-id fetchers used by paper-qa and
llama-index scholarly loaders. This connector accepts comma/whitespace/newline
separated paper ids (or DOI:/ARXIV: prefixed ids), posts them to the bulk
endpoint, and normalizes each hit into a :class:`Document` for agentic RAG
pipelines synthesizing with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_PAPER_FIELDS = "title,abstract,year,authors,url,externalIds"
_MAX_IDS = 100
_ID_SPLIT = re.compile(r"[\s,;]+")


class SemanticScholarBulkConnector:
    """Fetch and normalize Semantic Scholar papers via the bulk batch API."""

    def __init__(self, api_key: str | None = None) -> None:
        """Create a connector with an optional API key.

        Args:
            api_key: Optional Semantic Scholar API key.
        """
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized documents for a batch of paper ids.

        Args:
            query: Comma/whitespace separated paper ids (or DOI:/ARXIV: ids).
            max_results: Maximum number of documents to return.

        Returns:
            Normalized paper documents. Blank queries, empty id lists, and
            non-positive ``max_results`` yield an empty list.
        """
        ids = self._parse_ids(query)
        if max_results <= 0 or not ids:
            return []

        ids = ids[: min(max_results, _MAX_IDS)]
        headers = {"x-api-key": self._api_key} if self._api_key else None
        params = {"fields": _PAPER_FIELDS}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    S2_BULK_URL,
                    params=params,
                    headers=headers,
                    json={"ids": ids},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return []

        return self._parse_batch(payload, max_results)

    @staticmethod
    def _parse_ids(query: str) -> list[str]:
        """Split a query into unique paper ids preserving order."""
        parts = [part.strip() for part in _ID_SPLIT.split(query.strip()) if part.strip()]
        seen: set[str] = set()
        ordered: list[str] = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                ordered.append(part)
        return ordered

    @classmethod
    def _parse_batch(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Semantic Scholar batch payload into documents."""
        if not isinstance(payload, list):
            return []
        documents: list[Document] = []
        for item in payload:
            if not isinstance(item, dict) or item.get("paperId") in (None, ""):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one Semantic Scholar paper record."""
        title = str(item.get("title") or "").strip()
        if not title:
            return None
        paper_id = str(item.get("paperId") or "").strip()
        abstract = str(item.get("abstract") or "").strip()
        year = item.get("year")
        year_str = str(year) if isinstance(year, int) else str(year or "").strip()
        authors_raw = item.get("authors")
        authors = ""
        if isinstance(authors_raw, list):
            names = [
                str(author.get("name") or "").strip()
                for author in authors_raw
                if isinstance(author, dict) and str(author.get("name") or "").strip()
            ]
            authors = ", ".join(names[:12])
        external = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        doi = str(external.get("DOI") or "").strip() if isinstance(external, dict) else ""
        url = str(item.get("url") or "").strip() or (
            f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else title
        )
        text_parts = [title + "."]
        if abstract:
            text_parts.append(abstract)
        if authors:
            text_parts.append(f"Authors: {authors}.")
        if year_str:
            text_parts.append(f"Year: {year_str}.")
        if doi:
            text_parts.append(f"DOI: {doi}.")
        return Document(
            document_id=stable_id(url, "doc"),
            title=" ".join(title.split()),
            text=" ".join(text_parts),
            source=url,
            metadata={
                "source_type": "semantic_scholar_bulk",
                "semantic_scholar_id": paper_id,
                "doi": doi,
                "authors": authors,
                "year": year_str,
            },
        )
