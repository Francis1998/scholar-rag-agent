"""CORE open-access research ingestion connector.

CORE (https://core.ac.uk) aggregates open-access research outputs from
repositories and journals worldwide, offering one of the largest full-text
open-access indexes available via a public REST API. It complements the
existing sources (arXiv, Semantic Scholar, OpenAlex, PubMed, Crossref,
Europe PMC, DOAJ, DBLP, HAL, OpenAIRE, Zenodo, Figshare) with broad
cross-repository coverage of OA works that may not surface in a single
publisher or preprint index.

Its public ``search/works`` endpoint takes a free-text ``q`` query and returns
hits under ``results``, each carrying ``title``, ``abstract``, ``doi``,
``yearPublished``, ``authors`` (each with a ``name``), ``downloadUrl``, and
``links`` (including a ``display`` landing page). One request can ingest
several works for a topic. Search metadata is available without authentication;
an optional API key is sent as a Bearer token when configured.
"""

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works"
_PAGE_SIZE_CAP = 100
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class CoreConnector:
    """Search CORE and normalize matching works into documents."""

    def __init__(self, api_key: str | None = None) -> None:
        """Create a connector.

        Args:
            api_key: Optional CORE API key sent as a Bearer token for higher
                rate limits and full-text access when available.
        """
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized CORE documents matching a query.

        Args:
            query: Free-text CORE query.
            max_results: Maximum number of works to fetch (CORE caps ``limit``
                at 100).

        Returns:
            Normalized documents for the matching works. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "q": query.strip(),
            "limit": min(max_results, _PAGE_SIZE_CAP),
        }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(CORE_SEARCH_URL, params=params, headers=headers)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a CORE ``search/works`` JSON payload into documents.

        Args:
            payload: Decoded CORE response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each work in ``results``.
        """
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            return []

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
    def _build_document(cls, work: dict[str, object]) -> Document | None:
        """Build a document from one CORE work object.

        Args:
            work: A single work from ``results``.

        Returns:
            Normalized document, or None when the work carries no usable title.
        """
        title = cls._as_str(work.get("title")).strip()
        if not title:
            return None
        authors = cls._extract_authors(work.get("authors"))
        abstract = cls._as_str(work.get("abstract")).strip()
        year = cls._extract_year(work.get("yearPublished"))
        doi = cls._as_str(work.get("doi")).strip()
        source = cls._resolve_source(work, doi, title)
        text = " ".join(abstract.split()) if abstract else cls._build_descriptor(authors, year)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "core",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
            },
        )

    @staticmethod
    def _extract_authors(authors: object) -> list[str]:
        """Extract ordered author names from a CORE ``authors`` list.

        Each author is an object carrying a ``name``; bare string entries are
        also tolerated.

        Args:
            authors: The ``authors`` value.

        Returns:
            Ordered author names, empty when none are present.
        """
        if not isinstance(authors, list):
            return []
        names: list[str] = []
        for entry in authors:
            if isinstance(entry, str):
                name = entry.strip()
            elif isinstance(entry, dict):
                name = CoreConnector._as_str(entry.get("name")).strip()
            else:
                name = ""
            if name:
                names.append(name)
        return names

    @staticmethod
    def _extract_year(year_published: object) -> str:
        """Extract the publication year from CORE ``yearPublished``.

        Args:
            year_published: The raw ``yearPublished`` field (typically an int).

        Returns:
            The year as a string, or an empty string when absent/invalid.
        """
        if isinstance(year_published, int) and not isinstance(year_published, bool):
            return str(year_published)
        if isinstance(year_published, str):
            match = _YEAR_PREFIX_PATTERN.match(year_published.strip())
            return match.group(1) if match else ""
        return ""

    @classmethod
    def _resolve_source(cls, work: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for a CORE work.

        A ``links`` entry with ``type == \"display\"`` is preferred, then
        ``downloadUrl``, then a DOI link, and finally the title as an anchor of
        last resort.

        Args:
            work: The work object.
            doi: The normalized DOI, if any.
            title: The title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        links = work.get("links")
        if isinstance(links, list):
            for entry in links:
                if not isinstance(entry, dict):
                    continue
                if cls._as_str(entry.get("type")).strip().lower() != "display":
                    continue
                url = cls._as_str(entry.get("url")).strip()
                if url:
                    return url
        download_url = cls._as_str(work.get("downloadUrl")).strip()
        if download_url:
            return download_url
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _build_descriptor(authors: list[str], year: str) -> str:
        """Compose a citation-style descriptor used when no abstract exists.

        Args:
            authors: Ordered author names.
            year: Publication year, if any.

        Returns:
            A single-line descriptor, or an empty string when nothing is known.
        """
        parts: list[str] = []
        if authors:
            parts.append("By " + ", ".join(authors))
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar CORE field value to a string.

        Args:
            value: A raw scalar CORE field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
