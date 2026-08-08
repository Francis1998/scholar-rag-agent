"""arXiv HTML abs-page abstract enrichment connector.

The public arXiv Atom API returns abstracts, but abs HTML pages
(``https://arxiv.org/abs/{id}``) can carry a cleaner citation abstract via
``citation_abstract`` meta tags or the ``blockquote.abstract`` block. This
connector enriches papers by fetching that HTML abstract text.

- When the query looks like an arXiv id, fetch that abs page directly.
- For free-text queries, search the arXiv API then enrich the first N hits
  with HTML abs abstracts when available.
"""

from __future__ import annotations

import html
import re

import httpx
from defusedxml import ElementTree

from ingestion.chunking import stable_id
from retrieval.models import Document

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_ABS_URL = "https://arxiv.org/abs"
ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"

_ARXIV_ID_PATTERN = re.compile(
    r"^(?:arxiv:)?(?:https?://(?:www\.)?arxiv\.org/abs/)?(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+/\d{7})(?:\.pdf)?$",
    re.IGNORECASE,
)
_META_ABSTRACT_PATTERN = re.compile(
    r'<meta[^>]+name=["\']citation_abstract["\'][^>]+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
_META_ABSTRACT_PATTERN_ALT = re.compile(
    r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']citation_abstract["\']',
    re.IGNORECASE | re.DOTALL,
)
_BLOCKQUOTE_ABSTRACT_PATTERN = re.compile(
    r'<blockquote[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</blockquote>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_META_TITLE_PATTERN = re.compile(
    r'<meta[^>]+name=["\']citation_title["\'][^>]+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)


class ArxivHtmlAbstractConnector:
    """Fetch arXiv abs HTML abstracts and normalize enriched documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return documents enriched with abs-page HTML abstract text.

        Args:
            query: An arXiv id (``2301.00001``, ``arxiv:2301.00001``, or abs
                URL) or free-text search query.
            max_results: Maximum number of documents to return.

        Returns:
            Normalized documents. Blank queries, non-positive ``max_results``,
            and HTTP/parse failures yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        arxiv_id = self._normalize_arxiv_id(stripped)
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                if arxiv_id is not None:
                    document = await self._fetch_abs_document(client, arxiv_id)
                    return [document] if document is not None else []

                papers = await self._search_arxiv_api(client, stripped, max_results)
                documents: list[Document] = []
                for paper in papers:
                    document = await self._enrich_paper(client, paper)
                    if document is not None:
                        documents.append(document)
                    if len(documents) >= max_results:
                        break
                return documents
        except (httpx.HTTPError, ValueError, ElementTree.ParseError):
            return []

    async def _search_arxiv_api(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> list[dict[str, str]]:
        """Search the arXiv Atom API and return lightweight paper records."""
        response = await client.get(
            ARXIV_API_URL,
            params={
                "search_query": query,
                "start": 0,
                "max_results": max_results,
            },
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        papers: list[dict[str, str]] = []
        for entry in root.findall(f"{ATOM_NAMESPACE}entry"):
            title = (entry.findtext(f"{ATOM_NAMESPACE}title") or "Untitled arXiv paper").strip()
            summary = (entry.findtext(f"{ATOM_NAMESPACE}summary") or "").strip()
            entry_id = (entry.findtext(f"{ATOM_NAMESPACE}id") or "").strip()
            arxiv_id = self._normalize_arxiv_id(entry_id) or ""
            papers.append(
                {
                    "title": " ".join(title.split()),
                    "summary": " ".join(summary.split()),
                    "entry_id": entry_id,
                    "arxiv_id": arxiv_id,
                }
            )
        return papers

    async def _enrich_paper(
        self,
        client: httpx.AsyncClient,
        paper: dict[str, str],
    ) -> Document | None:
        """Enrich one API hit with HTML abs abstract text when available."""
        arxiv_id = paper.get("arxiv_id", "").strip()
        title = paper.get("title", "").strip() or "Untitled arXiv paper"
        api_summary = paper.get("summary", "").strip()
        entry_id = paper.get("entry_id", "").strip()

        html_abstract = ""
        if arxiv_id:
            html_abstract = await self._fetch_html_abstract(client, arxiv_id)

        abstract = html_abstract or api_summary
        if not abstract and not title:
            return None

        source = entry_id or (f"{ARXIV_ABS_URL}/{arxiv_id}" if arxiv_id else title)
        return Document(
            document_id=stable_id(source, "doc"),
            title=title,
            text=abstract or title,
            source=source,
            metadata={
                "source_type": "arxiv_html_abstract",
                "arxiv_id": arxiv_id,
                "abstract_source": "html_abs" if html_abstract else "atom_api",
            },
        )

    async def _fetch_abs_document(
        self,
        client: httpx.AsyncClient,
        arxiv_id: str,
    ) -> Document | None:
        """Fetch one abs HTML page and build a document."""
        url = f"{ARXIV_ABS_URL}/{arxiv_id}"
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError:
            return None

        abstract = self._extract_html_abstract(response.text)
        if not abstract:
            return None

        title = self._extract_html_title(response.text) or arxiv_id
        return Document(
            document_id=stable_id(url, "doc"),
            title=" ".join(title.split()),
            text=abstract,
            source=url,
            metadata={
                "source_type": "arxiv_html_abstract",
                "arxiv_id": arxiv_id,
                "abstract_source": "html_abs",
            },
        )

    async def _fetch_html_abstract(
        self,
        client: httpx.AsyncClient,
        arxiv_id: str,
    ) -> str:
        """Fetch abs HTML and return extracted abstract text, or empty."""
        try:
            response = await client.get(f"{ARXIV_ABS_URL}/{arxiv_id}")
            response.raise_for_status()
        except httpx.HTTPError:
            return ""
        return self._extract_html_abstract(response.text)

    @classmethod
    def _extract_html_abstract(cls, page_html: str) -> str:
        """Extract abstract text from an arXiv abs HTML page."""
        for pattern in (_META_ABSTRACT_PATTERN, _META_ABSTRACT_PATTERN_ALT):
            match = pattern.search(page_html)
            if match:
                cleaned = cls._clean_abstract(html.unescape(match.group(1)))
                if cleaned:
                    return cleaned

        match = _BLOCKQUOTE_ABSTRACT_PATTERN.search(page_html)
        if match:
            cleaned = cls._clean_abstract(html.unescape(_TAG_PATTERN.sub(" ", match.group(1))))
            if cleaned:
                return cleaned
        return ""

    @staticmethod
    def _extract_html_title(page_html: str) -> str:
        """Extract citation title from abs HTML when present."""
        match = _META_TITLE_PATTERN.search(page_html)
        if not match:
            return ""
        return html.unescape(match.group(1)).strip()

    @staticmethod
    def _clean_abstract(text: str) -> str:
        """Normalize whitespace and strip leading Abstract: descriptors."""
        cleaned = " ".join(text.split()).strip()
        if cleaned.lower().startswith("abstract:"):
            cleaned = cleaned[9:].strip()
        return cleaned

    @staticmethod
    def _normalize_arxiv_id(query: str) -> str | None:
        """Return a bare arXiv id when ``query`` is id-shaped."""
        candidate = query.strip()
        match = _ARXIV_ID_PATTERN.fullmatch(candidate)
        if match:
            return match.group(1)
        # Atom entry ids look like http://arxiv.org/abs/1234.5678
        if "arxiv.org/abs/" in candidate.lower():
            tail = candidate.rsplit("/", maxsplit=1)[-1].strip()
            match = _ARXIV_ID_PATTERN.fullmatch(tail)
            if match:
                return match.group(1)
            if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", tail, re.IGNORECASE):
                return tail
        return None
