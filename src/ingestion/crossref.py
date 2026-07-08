"""Crossref REST API ingestion connector.

Crossref (https://www.crossref.org) is the largest DOI registration agency and
a broad, cross-disciplinary index of scholarly metadata, complementing arXiv,
Semantic Scholar, OpenAlex, and PubMed. Like PubMed, it is queried by keyword:
its ``works`` endpoint returns a list of matching works, each with a title, an
optional abstract (encoded as JATS XML), a DOI, and a publication year. This
connector runs a single ``works?query=`` request and normalizes every hit into
a :class:`Document`, so one call can ingest several papers for a topic.
"""

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

CROSSREF_BASE_URL = "https://api.crossref.org/works"

_JATS_TAG_PATTERN = re.compile(r"<[^>]+>")


class CrossrefConnector:
    """Search Crossref and normalize matching works into documents."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional contact email added to requests so Crossref routes
                traffic to its faster, polite API pool.
        """
        self._mailto = mailto

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Crossref documents matching a query.

        Args:
            query: Free-text Crossref query.
            max_results: Maximum number of works to fetch.

        Returns:
            Normalized documents for the matching works. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {"query": query, "rows": max_results}
        if self._mailto:
            params["mailto"] = self._mailto

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(CROSSREF_BASE_URL, params=params)
            response.raise_for_status()

        return self._parse_works(response.json(), max_results)

    @classmethod
    def _parse_works(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Crossref ``works`` JSON payload into documents.

        Args:
            payload: Decoded Crossref response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each work item in the payload.
        """
        if not isinstance(payload, dict):
            return []
        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        items = message.get("items")
        if not isinstance(items, list):
            return []

        documents: list[Document] = []
        for item in items:
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
        """Build a document from one Crossref work item.

        Args:
            item: A single work object from ``message.items``.

        Returns:
            Normalized document, or None when the item carries no usable title
            or DOI to anchor it.
        """
        title = cls._first_string(item.get("title")) or "Untitled Crossref work"
        doi = item.get("DOI")
        doi_str = doi.strip() if isinstance(doi, str) else ""
        source = f"https://doi.org/{doi_str}" if doi_str else title
        abstract = cls._strip_jats(item.get("abstract"))
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=abstract,
            source=source,
            metadata={
                "source_type": "crossref",
                "doi": doi_str,
                "year": cls._extract_year(item.get("published")),
            },
        )

    @staticmethod
    def _first_string(value: object) -> str:
        """Return the first non-empty string in a Crossref list field.

        Crossref encodes single-valued fields such as ``title`` as a list of
        strings. The first non-empty entry is used.

        Args:
            value: A Crossref field expected to be a list of strings.

        Returns:
            The first non-empty string, or an empty string when none is present.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str) and entry.strip():
                    return entry
        return ""

    @classmethod
    def _strip_jats(cls, abstract: object) -> str:
        """Strip JATS XML markup from a Crossref abstract.

        Crossref abstracts are stored as JATS XML (for example
        ``<jats:p>...</jats:p>``). Tags are removed and surrounding whitespace is
        collapsed so the stored text is plain, readable prose.

        Args:
            abstract: The raw ``abstract`` field.

        Returns:
            Plain-text abstract, or an empty string when unavailable.
        """
        if not isinstance(abstract, str) or not abstract.strip():
            return ""
        without_tags = _JATS_TAG_PATTERN.sub(" ", abstract)
        return " ".join(without_tags.split())

    @staticmethod
    def _extract_year(published: object) -> str:
        """Extract the publication year from a Crossref ``published`` field.

        Crossref encodes dates as ``{"date-parts": [[year, month, day]]}`` with
        only the year guaranteed present. The leading year integer is returned as
        a string.

        Args:
            published: The ``published`` (or equivalent) date field.

        Returns:
            The publication year as a string, or an empty string when absent.
        """
        if not isinstance(published, dict):
            return ""
        date_parts = published.get("date-parts")
        if not isinstance(date_parts, list) or not date_parts:
            return ""
        first_date = date_parts[0]
        if not isinstance(first_date, list) or not first_date:
            return ""
        year = first_date[0]
        if isinstance(year, int) and not isinstance(year, bool):
            return str(year)
        return ""
