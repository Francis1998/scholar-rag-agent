"""HAL (Hyper Articles en Ligne) open-archive ingestion connector.

HAL (https://hal.science) is France's multidisciplinary open archive, run by the
CCSD/CNRS, indexing preprints, articles, theses, and conference papers across the
sciences and humanities with especially strong coverage of European
mathematics, physics, and computer science that the biomedical- and
US-leaning sources (arXiv, Semantic Scholar, OpenAlex, PubMed, Crossref,
Europe PMC, DOAJ, DBLP) cover unevenly.

Its public Solr-backed ``search`` endpoint takes a free-text ``q`` query and,
with an explicit ``fl`` field list, returns each hit inside ``response.docs``
carrying the title, full author names, abstract, landing-page URI, DOI, and
publication year. Solr multi-valued fields (``title_s``, ``abstract_s``,
``authFullName_s``) arrive as arrays even when single-valued, so this connector
normalises both the array and scalar shapes. When a record has no abstract
(common for bibliographic-only deposits) a concise author/year descriptor is
synthesised as the document text. One request can ingest several publications
for a topic, and the endpoint is unauthenticated.
"""

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

HAL_BASE_URL = "https://api.archives-ouvertes.fr/search/"
_FIELD_LIST = "title_s,authFullName_s,abstract_s,uri_s,doiId_s,publicationDateY_i,producedDateY_i"


class HalConnector:
    """Search HAL and normalize matching publications into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized HAL documents matching a query.

        Args:
            query: Free-text HAL query.
            max_results: Maximum number of publications to fetch (HAL caps the
                ``rows`` count at 10000).

        Returns:
            Normalized documents for the matching publications. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "q": query.strip(),
            "wt": "json",
            "fl": _FIELD_LIST,
            "rows": min(max_results, 10000),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(HAL_BASE_URL, params=params)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a HAL ``search`` JSON payload into documents.

        Args:
            payload: Decoded HAL (Solr) response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each doc in ``response.docs``.
        """
        if not isinstance(payload, dict):
            return []
        response_block = payload.get("response")
        if not isinstance(response_block, dict):
            return []
        docs = response_block.get("docs")
        if not isinstance(docs, list):
            return []

        documents: list[Document] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            document = cls._build_document(doc)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, doc: dict[str, object]) -> Document | None:
        """Build a document from one HAL Solr ``doc`` object.

        Args:
            doc: A single publication object from ``response.docs``.

        Returns:
            Normalized document, or None when the entry carries no usable title.
        """
        title = cls._first_str(doc.get("title_s")).strip()
        if not title:
            return None
        authors = cls._string_list(doc.get("authFullName_s"))
        abstract = cls._first_str(doc.get("abstract_s")).strip()
        doi = cls._first_str(doc.get("doiId_s")).strip()
        year = cls._first_str(doc.get("publicationDateY_i", doc.get("producedDateY_i"))).strip()
        source = cls._resolve_source(doc, doi, title)
        text = " ".join(abstract.split()) if abstract else cls._build_descriptor(authors, year)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "hal",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
            },
        )

    @staticmethod
    def _build_descriptor(authors: list[str], year: str) -> str:
        """Compose a citation-style descriptor used when no abstract is present.

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
    def _resolve_source(doc: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for a HAL publication.

        The HAL landing-page URI (``uri_s``) is preferred; a DOI link and finally
        the title are used as ordered fallbacks.

        Args:
            doc: The publication ``doc`` object.
            doi: The normalized DOI, if any.
            title: The publication title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        uri = HalConnector._first_str(doc.get("uri_s")).strip()
        if uri:
            return uri
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _string_list(value: object) -> list[str]:
        """Return an ordered list of non-empty strings from a Solr field.

        Solr multi-valued fields arrive as a list, but a single-valued field may
        arrive as a bare scalar; both shapes (and integer scalars) are supported.

        Args:
            value: A raw HAL field value.

        Returns:
            Ordered non-empty string values.
        """
        if isinstance(value, list):
            entries: list[object] = value
        else:
            entries = [value]
        names: list[str] = []
        for entry in entries:
            text = HalConnector._scalar_str(entry).strip()
            if text:
                names.append(text)
        return names

    @classmethod
    def _first_str(cls, value: object) -> str:
        """Return the first non-empty string from a Solr field.

        Args:
            value: A raw HAL field value (list or scalar).

        Returns:
            The first non-empty string, or an empty string when none is present.
        """
        values = cls._string_list(value)
        return values[0] if values else ""

    @staticmethod
    def _scalar_str(value: object) -> str:
        """Coerce a scalar Solr value to a string.

        HAL encodes year fields (``publicationDateY_i``) as integers, so integers
        are coerced to their decimal string form and other non-string values
        yield an empty string.

        Args:
            value: A raw scalar HAL field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
