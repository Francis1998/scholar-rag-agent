"""PubMed Central full-text ingestion connector.

PubMed Central (PMC, https://pmc.ncbi.nlm.nih.gov) is NCBI's full-text archive
for biomedical and life-sciences literature. Unlike PubMed, which primarily
indexes citations and abstracts, PMC exposes open-access article XML that can
include abstracts, article bodies, contributors, DOI, PMCID, PMID, and
publication dates. This connector uses a mockable two-step E-utilities flow:
``esearch`` resolves a keyword query to PMC record identifiers, then ``efetch``
retrieves the matching JATS/NLM article XML for normalization into
:class:`Document` objects.
"""

import re

import httpx
from defusedxml import ElementTree

from ingestion.chunking import stable_id
from retrieval.models import Document

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE_URL}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE_URL}/efetch.fcgi"
PMC_ARTICLE_BASE_URL = "https://pmc.ncbi.nlm.nih.gov/articles"
_RESULT_CAP = 100
_BODY_EXCERPT_CHARS = 2_000
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")


class PmcConnector:
    """Search PMC and normalize matching full-text article records."""

    def __init__(self, api_key: str | None = None, email: str | None = None) -> None:
        """Create a connector with optional NCBI request metadata.

        Args:
            api_key: Optional NCBI API key. When present it raises the
                E-utilities rate limit and is attached to every request.
            email: Optional contact email forwarded to NCBI for polite API use.
        """
        self._api_key = api_key
        self._email = email

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized PMC documents matching a query.

        Args:
            query: Free-text PMC query.
            max_results: Maximum number of articles to fetch. Requests are
                bounded to PMC's connector cap of 100 records per call.

        Returns:
            Normalized documents for matching PMC articles. An empty list is
            returned when the query is blank, non-positive, or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        retmax = min(max_results, _RESULT_CAP)
        async with httpx.AsyncClient(timeout=30.0) as client:
            esearch_response = await client.get(
                ESEARCH_URL,
                params=self._with_ncbi_params(
                    {
                        "db": "pmc",
                        "term": query.strip(),
                        "retmax": retmax,
                        "retmode": "json",
                    }
                ),
            )
            esearch_response.raise_for_status()
            pmc_ids = self._extract_pmc_ids(esearch_response.json(), retmax)
            if not pmc_ids:
                return []

            efetch_response = await client.get(
                EFETCH_URL,
                params=self._with_ncbi_params(
                    {"db": "pmc", "id": ",".join(pmc_ids), "retmode": "xml"}
                ),
            )
            efetch_response.raise_for_status()

        return self._parse_articles(efetch_response.text, max_results)

    def _with_ncbi_params(self, params: dict[str, str | int]) -> dict[str, str | int]:
        """Attach optional NCBI API metadata to request params."""
        enriched = dict(params)
        if self._api_key:
            enriched["api_key"] = self._api_key
        if self._email:
            enriched["email"] = self._email
        return enriched

    @staticmethod
    def _extract_pmc_ids(payload: object, max_results: int) -> list[str]:
        """Extract ordered PMC E-utilities identifiers from an ``esearch`` payload."""
        if not isinstance(payload, dict):
            return []
        result = payload.get("esearchresult")
        if not isinstance(result, dict):
            return []
        idlist = result.get("idlist")
        if not isinstance(idlist, list):
            return []
        pmc_ids = [str(pmc_id) for pmc_id in idlist if isinstance(pmc_id, (str, int))]
        return pmc_ids[:max_results]

    @classmethod
    def _parse_articles(cls, xml_text: str, max_results: int) -> list[Document]:
        """Parse a PMC ``efetch`` JATS XML payload into documents."""
        root = ElementTree.fromstring(xml_text)
        documents: list[Document] = []
        for article in root.findall(".//article"):
            document = cls._build_document(article)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, article: object) -> Document | None:
        """Build a document from one PMC article XML element."""
        title = cls._extract_first_text(article, ".//article-meta/title-group/article-title")
        if not title:
            return None

        pmcid = cls._normalize_pmcid(cls._extract_article_id(article, "pmc"))
        doi = cls._extract_article_id(article, "doi")
        pmid = cls._extract_article_id(article, "pmid")
        year = cls._resolve_year(article)
        authors = cls._extract_authors(article)
        source = cls._resolve_source(pmcid, doi, title)
        text = cls._extract_document_text(article)
        return Document(
            document_id=stable_id(source, "doc"),
            title=title,
            text=text,
            source=source,
            metadata={
                "source_type": "pmc",
                "pmcid": pmcid,
                "pmid": pmid,
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
            },
        )

    @classmethod
    def _extract_document_text(cls, article: object) -> str:
        """Prefer abstracts and append a bounded full-text excerpt when present."""
        abstract = cls._extract_sections(article, ".//article-meta/abstract")
        body = cls._extract_sections(article, ".//body")
        if body:
            excerpt = body[:_BODY_EXCERPT_CHARS].rstrip()
            if abstract:
                return f"{abstract}\n\nFull-text excerpt: {excerpt}"
            return excerpt
        return abstract

    @classmethod
    def _extract_authors(cls, article: object) -> list[str]:
        """Extract ordered author names from PMC contributor metadata."""
        findall = getattr(article, "findall", None)
        if findall is None:
            return []
        authors: list[str] = []
        for contrib in findall(".//article-meta/contrib-group/contrib[@contrib-type='author']"):
            given = cls._extract_first_text(contrib, "./name/given-names")
            surname = cls._extract_first_text(contrib, "./name/surname")
            collab = cls._extract_first_text(contrib, "./collab")
            name = " ".join(part for part in (given, surname) if part).strip() or collab
            if name:
                authors.append(name)
        return authors

    @classmethod
    def _resolve_year(cls, article: object) -> str:
        """Resolve the publication year from PMC article metadata."""
        for path in (
            ".//article-meta/pub-date/year",
            ".//front-stub/pub-date/year",
            ".//article-meta/history/date/year",
        ):
            year = cls._extract_first_text(article, path)
            if year:
                return year
        date_text = cls._extract_first_text(article, ".//article-meta/pub-date")
        match = _YEAR_PREFIX_PATTERN.match(date_text)
        return match.group(1) if match else ""

    @classmethod
    def _extract_article_id(cls, article: object, pub_id_type: str) -> str:
        """Extract an ``article-id`` with a specific ``pub-id-type``."""
        findall = getattr(article, "findall", None)
        if findall is None:
            return ""
        for node in findall(".//article-meta/article-id"):
            get = getattr(node, "get", None)
            if get is None or get("pub-id-type") != pub_id_type:
                continue
            text = cls._collapse_node_text(node)
            if text:
                return text
        return ""

    @staticmethod
    def _normalize_pmcid(pmcid: str) -> str:
        """Return a PMCID with the canonical ``PMC`` prefix."""
        value = pmcid.strip()
        if not value:
            return ""
        if value.upper().startswith("PMC"):
            suffix = value[3:]
            return f"PMC{suffix}"
        return f"PMC{value}"

    @staticmethod
    def _resolve_source(pmcid: str, doi: str, title: str) -> str:
        """Resolve the canonical article URL, preferring PMCID landing pages."""
        if pmcid:
            return f"{PMC_ARTICLE_BASE_URL}/{pmcid}/"
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @classmethod
    def _extract_first_text(cls, article: object, path: str) -> str:
        """Extract and collapse the first XML element matching ``path``."""
        find = getattr(article, "find", None)
        if find is None:
            return ""
        node = find(path)
        if node is None:
            return ""
        return cls._collapse_node_text(node)

    @classmethod
    def _extract_sections(cls, article: object, path: str) -> str:
        """Extract and collapse text from every XML section matching ``path``."""
        findall = getattr(article, "findall", None)
        if findall is None:
            return ""
        sections: list[str] = []
        for node in findall(path):
            text = cls._collapse_node_text(node)
            if text:
                sections.append(text)
        return " ".join(sections)

    @staticmethod
    def _collapse_node_text(node: object) -> str:
        """Collapse all text contained in an XML node, preserving inline markup."""
        itertext = getattr(node, "itertext", None)
        if itertext is None:
            return ""
        return " ".join("".join(itertext()).split())
