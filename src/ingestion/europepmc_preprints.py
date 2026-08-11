"""Europe PMC preprints ingestion connector.

Europe PMC indexes life-science preprints under the ``PPR`` source. This
connector searches the public REST endpoint with an enforced ``SRC:PPR`` query
filter and normalizes only preprint records into :class:`Document` objects. It
is distinct from ``europepmc.py`` (all publication sources) and
``europepmc_grants.py`` (GRIST funding awards).

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

EUROPEPMC_PREPRINTS_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_PAGE_SIZE_CAP = 1000


class EuropePmcPreprintsConnector:
    """Search Europe PMC's PPR source and normalize preprints."""

    def __init__(self, email: str | None = None) -> None:
        """Create a connector with an optional polite-traffic contact email."""
        self._email = email

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return Europe PMC preprints matching a query.

        Blank queries, non-positive limits, unavailable API responses, and
        malformed payloads yield an empty list.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        params: dict[str, str | int] = {
            "query": f"({stripped}) AND SRC:PPR",
            "resultType": "core",
            "format": "json",
            "pageSize": min(max_results, _PAGE_SIZE_CAP),
        }
        if self._email:
            params["email"] = self._email

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_results(payload, max_results)

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch Europe PMC preprints, returning an empty payload on failure."""
        try:
            response = await client.get(EUROPEPMC_PREPRINTS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse only ``PPR`` records from a Europe PMC search payload."""
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
            if cls._as_str(item.get("source")).strip().upper() != "PPR":
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one Europe PMC preprint record."""
        title = cls._as_str(item.get("title")).strip()
        preprint_id = cls._as_str(item.get("id")).strip()
        if not title and not preprint_id:
            return None
        if not title:
            title = f"Europe PMC preprint {preprint_id}"

        doi = cls._as_str(item.get("doi")).strip()
        authors = cls._as_str(item.get("authorString")).strip()
        abstract = " ".join(cls._as_str(item.get("abstractText")).split())
        year = cls._resolve_year(item)
        preprint_server = cls._as_str(item.get("journalTitle")).strip()
        cited_by_count = cls._as_str(item.get("citedByCount")).strip()
        is_open_access = cls._as_bool_str(item.get("isOpenAccess"))
        source = (
            f"https://europepmc.org/article/PPR/{preprint_id}"
            if preprint_id
            else f"https://doi.org/{doi}"
            if doi
            else title
        )
        text = abstract or cls._build_descriptor(
            authors=authors,
            year=year,
            preprint_server=preprint_server,
            doi=doi,
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "europepmc_preprints",
                "preprint_id": preprint_id,
                "doi": doi,
                "year": year,
                "authors": authors,
                "preprint_server": preprint_server,
                "cited_by_count": cited_by_count,
                "is_open_access": is_open_access,
            },
        )

    @classmethod
    def _resolve_year(cls, item: dict[str, object]) -> str:
        """Resolve publication year with first-publication-date fallback."""
        year = cls._as_str(item.get("pubYear")).strip()
        if year:
            return year
        first_publication_date = cls._as_str(item.get("firstPublicationDate")).strip()
        return first_publication_date[:4] if first_publication_date[:4].isdigit() else ""

    @staticmethod
    def _build_descriptor(
        *,
        authors: str,
        year: str,
        preprint_server: str,
        doi: str,
    ) -> str:
        """Compose searchable text when Europe PMC omits an abstract."""
        parts = ["Europe PMC preprint."]
        if authors:
            parts.append(f"Authors: {authors}.")
        if preprint_server:
            parts.append(f"Preprint server: {preprint_server}.")
        if year:
            parts.append(f"Year: {year}.")
        if doi:
            parts.append(f"DOI: {doi}.")
        return " ".join(parts)

    @staticmethod
    def _as_bool_str(value: object) -> str:
        """Normalize bool-like API values to lowercase strings."""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower()
        return ""

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar Europe PMC fields to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
