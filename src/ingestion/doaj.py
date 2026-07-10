"""DOAJ (Directory of Open Access Journals) ingestion connector.

DOAJ (https://doaj.org) is a community-curated index of peer-reviewed open
access journals and their articles, complementing arXiv, Semantic Scholar,
OpenAlex, PubMed, Crossref, and Europe PMC with a source that guarantees a
freely readable full text. Its public ``search/articles`` endpoint takes a
free-text query in the URL path and returns each hit inside a ``bibjson``
object carrying the title, abstract, publication year, DOI, and full-text link.
This connector runs one ``search`` request and normalizes every hit into a
:class:`Document`, so one call can ingest several open access papers for a
topic. The endpoint is unauthenticated.
"""

from urllib.parse import quote

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

DOAJ_BASE_URL = "https://doaj.org/api/search/articles"


class DoajConnector:
    """Search DOAJ and normalize matching open access articles into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized DOAJ documents matching a query.

        Args:
            query: Free-text DOAJ query.
            max_results: Maximum number of articles to fetch (DOAJ caps
                ``pageSize`` at 100).

        Returns:
            Normalized documents for the matching articles. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        # The search query is a URL *path* parameter; encode it so slashes and
        # spaces in the query cannot break out of the path segment.
        url = f"{DOAJ_BASE_URL}/{quote(query.strip(), safe='')}"
        params: dict[str, str | int] = {"pageSize": min(max_results, 100), "page": 1}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a DOAJ ``search/articles`` JSON payload into documents.

        Args:
            payload: Decoded DOAJ response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each result in the payload.
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
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one DOAJ result item.

        Args:
            item: A single result object from ``results``.

        Returns:
            Normalized document, or None when the item carries no ``bibjson``.
        """
        bibjson = item.get("bibjson")
        if not isinstance(bibjson, dict):
            return None
        title = cls._as_str(bibjson.get("title")).strip() or "Untitled DOAJ article"
        abstract = cls._as_str(bibjson.get("abstract")).strip()
        doi = cls._extract_doi(bibjson.get("identifier"))
        article_id = cls._as_str(item.get("id")).strip()
        source = cls._resolve_source(bibjson.get("link"), doi, article_id, title)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=" ".join(abstract.split()),
            source=source,
            metadata={
                "source_type": "doaj",
                "doi": doi,
                "year": cls._as_str(bibjson.get("year")).strip(),
            },
        )

    @staticmethod
    def _extract_doi(identifiers: object) -> str:
        """Extract the DOI from a DOAJ ``bibjson.identifier`` list.

        Args:
            identifiers: The ``bibjson.identifier`` value (a list of typed id
                objects) or any other shape.

        Returns:
            The DOI string when a ``doi`` identifier is present, else an empty
            string. The identifier ``type`` is matched case-insensitively
            because DOAJ historically emitted ``DOI`` before normalising it to
            ``doi``.
        """
        if not isinstance(identifiers, list):
            return ""
        for identifier in identifiers:
            if not isinstance(identifier, dict):
                continue
            if DoajConnector._as_str(identifier.get("type")).strip().lower() == "doi":
                return DoajConnector._as_str(identifier.get("id")).strip()
        return ""

    @staticmethod
    def _resolve_source(links: object, doi: str, article_id: str, title: str) -> str:
        """Resolve the canonical source URL for a DOAJ article.

        A DOAJ article always advertises a freely readable full-text link; that
        URL is preferred. A DOI link, the DOAJ article page, and finally the
        title are used as ordered fallbacks.

        Args:
            links: The ``bibjson.link`` value (a list of typed link objects).
            doi: The normalized DOI, if any.
            article_id: The DOAJ internal article id, if any.
            title: The article title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        if isinstance(links, list):
            fulltext_url = ""
            first_url = ""
            for link in links:
                if not isinstance(link, dict):
                    continue
                url = DoajConnector._as_str(link.get("url")).strip()
                if not url:
                    continue
                if not first_url:
                    first_url = url
                if DoajConnector._as_str(link.get("type")).strip().lower() == "fulltext":
                    fulltext_url = url
                    break
            resolved = fulltext_url or first_url
            if resolved:
                return resolved
        if doi:
            return f"https://doi.org/{doi}"
        if article_id:
            return f"https://doaj.org/article/{article_id}"
        return title

    @staticmethod
    def _as_str(value: object) -> str:
        """Return a string field verbatim, coercing integers to their text form.

        DOAJ encodes some fields (for example ``year``) as either a JSON string
        or a number depending on the record, so integers are coerced to their
        decimal string form and other non-string values yield an empty string.

        Args:
            value: A raw DOAJ field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
