"""DBLP computer-science bibliography ingestion connector.

DBLP (https://dblp.org) is the authoritative open bibliography of computer
science, indexing conference and journal publications that the biomedical- and
general-science-leaning sources (arXiv, Semantic Scholar, OpenAlex, PubMed,
Crossref, Europe PMC, DOAJ) cover unevenly. Its public ``search/publ`` endpoint
takes a free-text query and returns each hit inside an ``info`` object carrying
the title, authors, venue, publication year, DOI, and electronic-edition link.

DBLP is a *bibliographic* index and does not expose abstracts, so this connector
synthesises a concise citation-style descriptor (authors, venue, year) as the
document text \u2014 giving sparse and entity-relationship retrieval real author and
venue signal \u2014 while the structured fields are preserved in the document
metadata. One ``search`` request can ingest several publications for a topic, and
the endpoint is unauthenticated.
"""

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

DBLP_BASE_URL = "https://dblp.org/search/publ/api"


class DblpConnector:
    """Search DBLP and normalize matching publications into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized DBLP documents matching a query.

        Args:
            query: Free-text DBLP query.
            max_results: Maximum number of publications to fetch (DBLP caps the
                ``h`` hit count at 1000).

        Returns:
            Normalized documents for the matching publications. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "q": query.strip(),
            "format": "json",
            "h": min(max_results, 1000),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(DBLP_BASE_URL, params=params)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a DBLP ``search/publ`` JSON payload into documents.

        Args:
            payload: Decoded DBLP response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each hit in the payload.
        """
        if not isinstance(payload, dict):
            return []
        result = payload.get("result")
        if not isinstance(result, dict):
            return []
        hits = result.get("hits")
        if not isinstance(hits, dict):
            return []
        hit_value = hits.get("hit")
        # DBLP returns ``hit`` as a list, but collapses it to a single object
        # when exactly one publication matches; normalize both shapes.
        if isinstance(hit_value, dict):
            hit_items: list[object] = [hit_value]
        elif isinstance(hit_value, list):
            hit_items = hit_value
        else:
            return []

        documents: list[Document] = []
        for hit in hit_items:
            if not isinstance(hit, dict):
                continue
            info = hit.get("info")
            if not isinstance(info, dict):
                continue
            document = cls._build_document(info)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, info: dict[str, object]) -> Document | None:
        """Build a document from one DBLP ``info`` object.

        Args:
            info: A single publication ``info`` object from ``hits.hit``.

        Returns:
            Normalized document, or None when the entry carries no usable title.
        """
        title = cls._as_str(info.get("title")).strip()
        if not title:
            return None
        authors = cls._extract_authors(info.get("authors"))
        venue = cls._as_str(info.get("venue")).strip()
        year = cls._as_str(info.get("year")).strip()
        doi = cls._as_str(info.get("doi")).strip()
        source = cls._resolve_source(info, doi, title)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=cls._build_descriptor(authors, venue, year),
            source=source,
            metadata={
                "source_type": "dblp",
                "doi": doi,
                "year": year,
                "venue": venue,
                "authors": ", ".join(authors),
            },
        )

    @staticmethod
    def _build_descriptor(authors: list[str], venue: str, year: str) -> str:
        """Compose a citation-style descriptor used as the document text.

        DBLP exposes no abstract, so a short factual descriptor built from the
        authors, venue, and year provides retrievable content without inventing
        prose that the record does not contain.

        Args:
            authors: Ordered author names.
            venue: Publication venue, if any.
            year: Publication year, if any.

        Returns:
            A single-line descriptor, or an empty string when nothing is known.
        """
        parts: list[str] = []
        if authors:
            parts.append("By " + ", ".join(authors))
        if venue:
            parts.append("In " + venue)
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @staticmethod
    def _extract_authors(authors: object) -> list[str]:
        """Extract ordered author names from a DBLP ``authors`` field.

        DBLP nests authors as ``{"author": [...]}`` where each entry is an object
        with a ``text`` name (a single author collapses to one object rather than
        a list). Both shapes, and bare string entries, are supported.

        Args:
            authors: The ``info.authors`` value.

        Returns:
            Ordered author names, empty when none are present.
        """
        if not isinstance(authors, dict):
            return []
        author_value = authors.get("author")
        if isinstance(author_value, dict):
            entries: list[object] = [author_value]
        elif isinstance(author_value, list):
            entries = author_value
        else:
            return []
        names: list[str] = []
        for entry in entries:
            if isinstance(entry, str):
                name = entry.strip()
            elif isinstance(entry, dict):
                name = DblpConnector._as_str(entry.get("text")).strip()
            else:
                name = ""
            if name:
                names.append(name)
        return names

    @staticmethod
    def _resolve_source(info: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for a DBLP publication.

        The electronic-edition link (``ee``) points at the publisher/open copy
        and is preferred; a DOI link, the DBLP record page (``url``), and finally
        the title are used as ordered fallbacks.

        Args:
            info: The publication ``info`` object.
            doi: The normalized DOI, if any.
            title: The publication title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        electronic_edition = DblpConnector._as_str(info.get("ee")).strip()
        if electronic_edition:
            return electronic_edition
        if doi:
            return f"https://doi.org/{doi}"
        record_url = DblpConnector._as_str(info.get("url")).strip()
        if record_url:
            return record_url
        return title

    @staticmethod
    def _as_str(value: object) -> str:
        """Return a string field verbatim, coercing integers to their text form.

        DBLP encodes some fields (for example ``year``) as either a JSON string
        or a number depending on the record, so integers are coerced to their
        decimal string form and other non-string values yield an empty string.

        Args:
            value: A raw DBLP field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
