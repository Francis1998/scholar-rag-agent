"""Europe PMC REST API ingestion connector.

Europe PMC (https://europepmc.org) is a large life-sciences literature database
that federates PubMed/MEDLINE, PubMed Central, preprint servers, patents, and
agricultural and biomedical sources, complementing arXiv, Semantic Scholar,
OpenAlex, PubMed, and Crossref. Its ``search`` endpoint is queried by keyword
and, with ``resultType=core``, returns each hit's title, abstract
(``abstractText``), DOI, and publication year in a single response. This
connector runs one ``search`` request and normalizes every hit into a
:class:`Document`, so one call can ingest several papers for a topic.
"""

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

EUROPEPMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePmcConnector:
    """Search Europe PMC and normalize matching articles into documents."""

    def __init__(self, email: str | None = None) -> None:
        """Create a connector.

        Args:
            email: Optional contact email forwarded to Europe PMC so it can
                identify polite API traffic.
        """
        self._email = email

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Europe PMC documents matching a query.

        Args:
            query: Free-text Europe PMC query.
            max_results: Maximum number of articles to fetch.

        Returns:
            Normalized documents for the matching articles. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "query": query,
            "resultType": "core",
            "format": "json",
            "pageSize": max_results,
        }
        if self._email:
            params["email"] = self._email

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(EUROPEPMC_BASE_URL, params=params)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Europe PMC ``search`` JSON payload into documents.

        Args:
            payload: Decoded Europe PMC response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each result in the payload.
        """
        if not isinstance(payload, dict):
            return []
        result_list = payload.get("resultList")
        if not isinstance(result_list, dict):
            return []
        results = result_list.get("result")
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
        """Build a document from one Europe PMC result item.

        Args:
            item: A single result object from ``resultList.result``.

        Returns:
            Normalized document, or None when the item carries no usable title.
        """
        title = cls._as_str(item.get("title")).strip() or "Untitled Europe PMC article"
        doi = cls._as_str(item.get("doi")).strip()
        source = cls._resolve_source(item, doi, title)
        abstract = cls._as_str(item.get("abstractText")).strip()
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=" ".join(abstract.split()),
            source=source,
            metadata={
                "source_type": "europepmc",
                "doi": doi,
                "year": cls._as_str(item.get("pubYear")).strip(),
                "pmid": cls._as_str(item.get("pmid")).strip(),
            },
        )

    @staticmethod
    def _resolve_source(item: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for a Europe PMC article.

        Europe PMC exposes a stable article page at
        ``europepmc.org/article/{source}/{id}`` (for example ``MED/40012345``).
        That page is preferred; a DOI link is used when the id pair is absent,
        and the title is used as a last resort.

        Args:
            item: A single result object.
            doi: The normalized DOI, if any.
            title: The article title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        article_source = EuropePmcConnector._as_str(item.get("source")).strip()
        article_id = EuropePmcConnector._as_str(item.get("id")).strip()
        if article_source and article_id:
            return f"https://europepmc.org/article/{article_source}/{article_id}"
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _as_str(value: object) -> str:
        """Return a string field verbatim, coercing integers to their text form.

        Europe PMC encodes some fields (for example ``pubYear``) as either a JSON
        string or a number depending on the record, so integers are coerced to
        their decimal string form and other non-string values yield an empty
        string.

        Args:
            value: A raw Europe PMC field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
