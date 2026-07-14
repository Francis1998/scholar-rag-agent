"""OpenAIRE Graph open-science ingestion connector.

OpenAIRE (https://www.openaire.eu) aggregates a very large, cross-disciplinary
open-science graph that harvests repositories, aggregators, and CRIS systems
across Europe and beyond, linking publications to their funding, projects, and
open-access copies. It complements the existing sources (arXiv, Semantic
Scholar, OpenAlex, PubMed, Crossref, Europe PMC, DOAJ, DBLP, HAL) with strong
coverage of EU-funded and institutionally deposited research.

Its public Graph ``researchProducts`` endpoint takes a free-text ``search``
query and returns each hit in a ``results`` array as a research-product object
carrying ``mainTitle``, an ordered ``authors`` list (each with a ``fullName``),
a ``descriptions`` list of abstract paragraphs, a ``publicationDate``, typed
``pids`` (from which the DOI is read), and ``instances`` whose ``urls`` point at
the open-access landing pages. When a record has no abstract, a concise
author/year descriptor is synthesised as the document text. One request can
ingest several publications for a topic, and the endpoint is unauthenticated.
"""

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENAIRE_BASE_URL = "https://api.openaire.eu/graph/researchProducts"


class OpenAireConnector:
    """Search OpenAIRE and normalize matching research products into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized OpenAIRE documents matching a query.

        Args:
            query: Free-text OpenAIRE query.
            max_results: Maximum number of research products to fetch (OpenAIRE
                caps ``pageSize`` at 100).

        Returns:
            Normalized documents for the matching research products. An empty
            list is returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "search": query.strip(),
            "type": "publication",
            "page": 1,
            "pageSize": min(max_results, 100),
            "sortBy": "relevance DESC",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(OPENAIRE_BASE_URL, params=params)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse an OpenAIRE ``researchProducts`` JSON payload into documents.

        Args:
            payload: Decoded OpenAIRE Graph response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each research product in ``results``.
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
    def _build_document(cls, product: dict[str, object]) -> Document | None:
        """Build a document from one OpenAIRE research-product object.

        Args:
            product: A single research product from ``results``.

        Returns:
            Normalized document, or None when the record carries no usable title.
        """
        title = cls._as_str(product.get("mainTitle")).strip()
        if not title:
            return None
        authors = cls._extract_authors(product.get("authors"))
        abstract = cls._first_str(product.get("descriptions")).strip()
        year = cls._as_str(product.get("publicationDate")).strip()[:4]
        doi = cls._extract_doi(product.get("pids"))
        source = cls._resolve_source(product, doi, title)
        text = " ".join(abstract.split()) if abstract else cls._build_descriptor(authors, year)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "openaire",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
            },
        )

    @staticmethod
    def _extract_authors(authors: object) -> list[str]:
        """Extract ordered author names from an OpenAIRE ``authors`` list.

        Each author is an object carrying a ``fullName``; bare string entries are
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
                name = OpenAireConnector._as_str(entry.get("fullName")).strip()
            else:
                name = ""
            if name:
                names.append(name)
        return names

    @staticmethod
    def _extract_doi(pids: object) -> str:
        """Extract the DOI from an OpenAIRE ``pids`` list.

        Args:
            pids: The ``pids`` value (a list of ``{scheme, value}`` objects).

        Returns:
            The DOI string when a ``doi``-scheme pid is present, else an empty
            string.
        """
        if not isinstance(pids, list):
            return ""
        for pid in pids:
            if not isinstance(pid, dict):
                continue
            if OpenAireConnector._as_str(pid.get("scheme")).strip().lower() == "doi":
                return OpenAireConnector._as_str(pid.get("value")).strip()
        return ""

    @classmethod
    def _resolve_source(cls, product: dict[str, object], doi: str, title: str) -> str:
        """Resolve the canonical source URL for an OpenAIRE research product.

        The first open-access landing URL advertised under ``instances[].urls``
        is preferred; a DOI link and finally the title are used as ordered
        fallbacks.

        Args:
            product: The research-product object.
            doi: The normalized DOI, if any.
            title: The title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        instances = product.get("instances")
        if isinstance(instances, list):
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                url = cls._first_str(instance.get("urls")).strip()
                if url:
                    return url
        if doi:
            return f"https://doi.org/{doi}"
        return title

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

    @classmethod
    def _first_str(cls, value: object) -> str:
        """Return the first non-empty string from a scalar or list field.

        Args:
            value: A raw OpenAIRE field value (list or scalar).

        Returns:
            The first non-empty string, or an empty string when none is present.
        """
        if isinstance(value, list):
            for entry in value:
                coerced = cls._as_str(entry).strip()
                if coerced:
                    return coerced
            return ""
        return cls._as_str(value)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar OpenAIRE field value to a string.

        Args:
            value: A raw scalar OpenAIRE field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
