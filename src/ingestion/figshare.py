"""Figshare open-research repository ingestion connector.

Figshare (https://figshare.com) is a large general-purpose open-research
repository that mints DOIs for figures, datasets, media, papers, posters,
presentations, theses, code, and other research outputs across every
discipline. It complements the existing sources (arXiv, Semantic Scholar,
OpenAlex, PubMed, Crossref, Europe PMC, DOAJ, DBLP, HAL, OpenAIRE, Zenodo)
with strong coverage of institutional and self-deposited research outputs
that may never reach a traditional publisher.

Its public REST search endpoint accepts a JSON body with a free-text
``search_for`` query and a ``page_size``, and returns a JSON array of article
objects carrying ``title``, ``doi``, ``published_date``, ``url_public_html``,
and optionally an HTML ``description``. The HTML description is reduced to
plain text when present. One request can ingest several outputs for a topic,
and the endpoint is unauthenticated.
"""

import html
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

FIGSHARE_SEARCH_URL = "https://api.figshare.com/v2/articles/search"
_PAGE_SIZE_CAP = 100
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


class FigshareConnector:
    """Search Figshare and normalize matching articles into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Figshare documents matching a query.

        Args:
            query: Free-text Figshare query.
            max_results: Maximum number of articles to fetch (Figshare caps
                ``page_size`` at 100).

        Returns:
            Normalized documents for the matching articles. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        body: dict[str, str | int] = {
            "search_for": query.strip(),
            "page_size": min(max_results, _PAGE_SIZE_CAP),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(FIGSHARE_SEARCH_URL, json=body)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Figshare ``articles/search`` JSON payload into documents.

        Args:
            payload: Decoded Figshare response (a JSON array of articles).
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each article in the payload.
        """
        if not isinstance(payload, list):
            return []

        documents: list[Document] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, article: dict[str, object]) -> Document | None:
        """Build a document from one Figshare article.

        Args:
            article: A single article object from the search response array.

        Returns:
            Normalized document, or None when the article carries no usable title.
        """
        title = cls._as_str(article.get("title")).strip()
        if not title:
            return None
        doi = cls._as_str(article.get("doi")).strip()
        year = cls._as_str(article.get("published_date")).strip()[:4]
        abstract = cls._strip_html(article.get("description"))
        source = cls._resolve_source(article, doi, title)
        text = abstract if abstract else cls._build_descriptor(year)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "figshare",
                "doi": doi,
                "year": year,
            },
        )

    @classmethod
    def _resolve_source(cls, article: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for a Figshare article.

        ``url_public_html`` is preferred, then a DOI link, and finally the title
        as an anchor of last resort.

        Args:
            article: The article object.
            doi: The normalized DOI, if any.
            title: The title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        url = cls._as_str(article.get("url_public_html")).strip()
        if url:
            return url
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _build_descriptor(year: str) -> str:
        """Compose a year descriptor used when no description exists.

        Args:
            year: Publication year, if any.

        Returns:
            A single-line descriptor, or an empty string when nothing is known.
        """
        if year:
            return f"({year})"
        return ""

    @classmethod
    def _strip_html(cls, description: object) -> str:
        """Reduce an HTML Figshare description to collapsed plain text.

        Figshare may store descriptions as HTML (for example ``<p>...</p>``)
        with entity-encoded characters. Tags are removed, entities are decoded,
        and surrounding whitespace is collapsed so the stored text is readable
        prose rather than leaking raw markup.

        Args:
            description: The raw ``description`` field.

        Returns:
            Plain-text description, or an empty string when unavailable.
        """
        if not isinstance(description, str) or not description.strip():
            return ""
        without_tags = _HTML_TAG_PATTERN.sub(" ", description)
        return " ".join(html.unescape(without_tags).split())

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar Figshare field value to a string.

        Args:
            value: A raw scalar Figshare field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
