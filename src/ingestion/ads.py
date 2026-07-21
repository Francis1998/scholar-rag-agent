"""NASA ADS (Astrophysics Data System) ingestion connector.

NASA ADS (https://ui.adsabs.harvard.edu) is the Smithsonian/NASA Astrophysics
Data System — the primary bibliographic index for astronomy, astrophysics, and
related physics literature. It complements the existing sources (arXiv, Semantic
Scholar, OpenAlex, PubMed, Crossref, Europe PMC, DOAJ, DBLP, HAL, OpenAIRE,
Zenodo, Figshare, CORE) with deep coverage of observatory papers, arXiv
astro-ph records, and citation-linked astronomy literature that other
general-purpose indexes surface less completely.

Its public search endpoint
``GET https://api.adsabs.harvard.edu/v1/search/query`` takes a free-text ``q``
query and returns hits under ``response.docs``, each carrying ``bibcode``,
``title`` (list), ``abstract``, ``author`` (list), ``year``, ``doi`` (list),
and ``pub``. An API token (``ADS_API_TOKEN``) is required by ADS for search;
when no token is configured the connector returns an empty list rather than
raising, so optional deployments stay quiet.
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
_ADS_FIELDS = "bibcode,title,abstract,author,year,doi,pub"
_PAGE_SIZE_CAP = 100
_YEAR_PATTERN = re.compile(r"^(\d{4})")


class AdsConnector:
    """Search NASA ADS and normalize matching records into documents."""

    def __init__(self, api_key: str | None = None) -> None:
        """Create a connector.

        Args:
            api_key: Optional NASA ADS API token sent as a Bearer token. When
                omitted, ``ADS_API_TOKEN`` is read from the environment. When
                neither is set, ``search`` returns an empty list without calling
                the API.
        """
        env_token = os.environ.get("ADS_API_TOKEN", "").strip()
        resolved = (api_key or env_token or "").strip()
        self._api_key = resolved or None

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized NASA ADS documents matching a query.

        Args:
            query: Free-text ADS query (Solr syntax is accepted by the API).
            max_results: Maximum number of records to fetch (ADS ``rows`` is
                capped at 100).

        Returns:
            Normalized documents for the matching records. An empty list is
            returned when the query is blank, ``max_results`` is non-positive,
            no API key is configured, or nothing matches.
        """
        if max_results <= 0 or not query.strip() or self._api_key is None:
            return []

        params: dict[str, str | int] = {
            "q": query.strip(),
            "fl": _ADS_FIELDS,
            "rows": min(max_results, _PAGE_SIZE_CAP),
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(ADS_SEARCH_URL, params=params, headers=headers)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a NASA ADS search JSON payload into documents.

        Args:
            payload: Decoded ADS response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each record in ``response.docs``.
        """
        if not isinstance(payload, dict):
            return []
        response = payload.get("response")
        if not isinstance(response, dict):
            return []
        docs = response.get("docs")
        if not isinstance(docs, list):
            return []

        documents: list[Document] = []
        for item in docs:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, record: dict[str, object]) -> Document | None:
        """Build a document from one ADS search hit.

        Args:
            record: A single object from ``response.docs``.

        Returns:
            Normalized document, or None when the record carries no usable title.
        """
        title = cls._first_str(record.get("title")).strip()
        if not title:
            return None
        authors = cls._extract_authors(record.get("author"))
        abstract = cls._as_str(record.get("abstract")).strip()
        year = cls._extract_year(record.get("year"))
        doi = cls._first_str(record.get("doi")).strip()
        bibcode = cls._as_str(record.get("bibcode")).strip()
        pub = cls._as_str(record.get("pub")).strip()
        source = cls._resolve_source(bibcode, doi, title)
        text = " ".join(abstract.split()) if abstract else cls._build_descriptor(authors, year, pub)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "ads",
                "doi": doi,
                "year": year,
                "authors": ", ".join(authors),
                "bibcode": bibcode,
                "pub": pub,
            },
        )

    @staticmethod
    def _extract_authors(authors: object) -> list[str]:
        """Extract ordered author names from an ADS ``author`` list.

        Args:
            authors: The ``author`` value (typically a list of strings).

        Returns:
            Ordered author names, empty when none are present.
        """
        if isinstance(authors, str):
            return [authors.strip()] if authors.strip() else []
        if not isinstance(authors, list):
            return []
        names: list[str] = []
        for entry in authors:
            name = AdsConnector._as_str(entry).strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def _extract_year(year: object) -> str:
        """Extract a four-digit publication year from ADS ``year``.

        Args:
            year: The raw ``year`` field (string or int).

        Returns:
            The year as a string, or an empty string when absent/invalid.
        """
        if isinstance(year, int) and not isinstance(year, bool):
            return str(year) if year >= 1000 else ""
        if isinstance(year, str):
            match = _YEAR_PATTERN.match(year.strip())
            return match.group(1) if match else ""
        return ""

    @classmethod
    def _resolve_source(cls, bibcode: str, doi: str, title: str) -> str:
        """Resolve the canonical source URL for an ADS record.

        The ADS abstract page for the bibcode is preferred, then a DOI link,
        and finally the title as an anchor of last resort.

        Args:
            bibcode: ADS bibcode, if any.
            doi: The normalized DOI, if any.
            title: The title, used as a final fallback anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        if bibcode:
            return f"https://ui.adsabs.harvard.edu/abs/{bibcode}"
        if doi:
            return f"https://doi.org/{doi}"
        return title

    @staticmethod
    def _build_descriptor(authors: list[str], year: str, pub: str) -> str:
        """Compose a citation-style descriptor used when no abstract exists.

        Args:
            authors: Ordered author names.
            year: Publication year, if any.
            pub: Journal / publication name, if any.

        Returns:
            A single-line descriptor, or an empty string when nothing is known.
        """
        parts: list[str] = []
        if authors:
            parts.append("By " + ", ".join(authors))
        if pub:
            parts.append(f"in {pub}")
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @classmethod
    def _first_str(cls, value: object) -> str:
        """Return the first string-like value from a scalar or list field.

        ADS returns multi-valued fields (``title``, ``doi``) as lists even when
        a record carries a single value.

        Args:
            value: A raw ADS field value.

        Returns:
            The first coercible string, or an empty string.
        """
        if isinstance(value, list):
            for entry in value:
                text = cls._as_str(entry).strip()
                if text:
                    return text
            return ""
        return cls._as_str(value)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar ADS field value to a string.

        Args:
            value: A raw scalar ADS field value.

        Returns:
            The string value, or an empty string when not string- or int-like.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
