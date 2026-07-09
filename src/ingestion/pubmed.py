"""PubMed E-utilities ingestion connector.

PubMed (https://pubmed.ncbi.nlm.nih.gov) is the primary index for biomedical
literature and a natural complement to arXiv, Semantic Scholar, and OpenAlex.
Unlike the single-record connectors, PubMed is queried by keyword: this
connector runs a two-step E-utilities flow — ``esearch`` to resolve a query to a
list of PMIDs, then ``efetch`` to retrieve each article's title and abstract —
and normalizes every hit into a :class:`Document`, so a single call can ingest
several papers for a topic.
"""

import httpx
from defusedxml import ElementTree

from ingestion.chunking import stable_id
from retrieval.models import Document

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE_URL}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE_URL}/efetch.fcgi"
PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov"


class PubMedConnector:
    """Search PubMed and normalize matching articles into documents."""

    def __init__(self, api_key: str | None = None) -> None:
        """Create a connector with an optional NCBI API key.

        Args:
            api_key: Optional NCBI API key. When present it raises the E-utilities
                rate limit and is attached to every request.
        """
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized PubMed documents matching a query.

        Args:
            query: Free-text PubMed query.
            max_results: Maximum number of articles to fetch.

        Returns:
            Normalized documents for the matching articles. An empty list is
            returned when the query matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        async with httpx.AsyncClient(timeout=30.0) as client:
            esearch_response = await client.get(
                ESEARCH_URL,
                params=self._with_api_key(
                    {
                        "db": "pubmed",
                        "term": query,
                        "retmax": max_results,
                        "retmode": "json",
                    }
                ),
            )
            esearch_response.raise_for_status()
            pmids = self._extract_pmids(esearch_response.json(), max_results)
            if not pmids:
                return []
            efetch_response = await client.get(
                EFETCH_URL,
                params=self._with_api_key(
                    {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
                ),
            )
            efetch_response.raise_for_status()

        return self._parse_articles(efetch_response.text)

    def _with_api_key(self, params: dict[str, str | int]) -> dict[str, str | int]:
        """Attach the NCBI API key to request params when configured.

        Args:
            params: Base request parameters.

        Returns:
            Parameters including the API key when one is set.
        """
        if self._api_key:
            return {**params, "api_key": self._api_key}
        return params

    @staticmethod
    def _extract_pmids(payload: object, max_results: int) -> list[str]:
        """Extract the PMID list from an ``esearch`` JSON payload.

        Args:
            payload: Decoded ``esearch`` JSON response.
            max_results: Upper bound on the number of PMIDs returned.

        Returns:
            Ordered list of PMID strings, capped at ``max_results``.
        """
        if not isinstance(payload, dict):
            return []
        result = payload.get("esearchresult")
        if not isinstance(result, dict):
            return []
        idlist = result.get("idlist")
        if not isinstance(idlist, list):
            return []
        pmids = [str(pmid) for pmid in idlist if isinstance(pmid, (str, int))]
        return pmids[:max_results]

    @classmethod
    def _parse_articles(cls, xml_text: str) -> list[Document]:
        """Parse an ``efetch`` PubMed XML payload into documents.

        Args:
            xml_text: ``efetch`` response body in PubMed XML format.

        Returns:
            Normalized documents for each ``PubmedArticle`` in the payload.
        """
        root = ElementTree.fromstring(xml_text)
        documents: list[Document] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
            title = (
                article.findtext(".//Article/ArticleTitle") or "Untitled PubMed article"
            ).strip()
            abstract = cls._extract_abstract(article)
            year = (article.findtext(".//Article/Journal/JournalIssue/PubDate/Year") or "").strip()
            source = f"{PUBMED_ARTICLE_URL}/{pmid}/" if pmid else title
            documents.append(
                Document(
                    document_id=stable_id(source, "doc"),
                    title=" ".join(title.split()),
                    text=abstract,
                    source=source,
                    metadata={
                        "source_type": "pubmed",
                        "pmid": pmid,
                        "year": year,
                    },
                )
            )
        return documents

    @staticmethod
    def _extract_abstract(article: object) -> str:
        """Join every ``AbstractText`` segment of a PubMed article in order.

        A structured abstract splits its body across several ``AbstractText``
        elements (each carrying a section ``Label`` such as ``METHODS``). All
        segments are concatenated with single spaces so the full abstract is
        preserved rather than truncated to its first section.

        PubMed also embeds inline formatting elements inside an ``AbstractText``
        (for example ``<i>`` for gene names or ``<sup>`` for exponents). Reading
        only ``node.text`` captured just the run of text before the first inline
        child and silently dropped the remainder, so every segment's full text is
        gathered with ``itertext`` instead.

        Args:
            article: A ``PubmedArticle`` XML element.

        Returns:
            The reconstructed abstract, or an empty string when none is present.
        """
        findall = getattr(article, "findall", None)
        if findall is None:
            return ""
        segments: list[str] = []
        for node in findall(".//Abstract/AbstractText"):
            itertext = getattr(node, "itertext", None)
            if itertext is None:
                continue
            segment = " ".join("".join(itertext()).split())
            if segment:
                segments.append(segment)
        return " ".join(segments)
