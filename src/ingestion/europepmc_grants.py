"""Europe PMC GRIST grants ingestion connector.

Europe PMC exposes a Grants RESTful (Grist) API over awards from Europe PMC
Funders. This connector searches that registry and normalizes each grant into a
:class:`Document` for funding-aware scholarly RAG.

Search requests use the documented URL form:

``GET https://www.ebi.ac.uk/europepmc/GristAPI/rest/get/query={query}&resultType=core&format=json``

Free-text queries default to keyword (``kw``) matching. Fielded Grist queries such
as ``ga:"Wellcome Trust"``, ``gid:083611``, ``pi:smith``, or ``title:cancer`` are
passed through unchanged. Prefer frontier models for downstream synthesis:
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

EUROPEPMC_GRANTS_BASE = "https://www.ebi.ac.uk/europepmc/GristAPI/rest/get/"
_PAGE_SIZE = 25


class EuropePmcGrantsConnector:
    """Search Europe PMC GRIST grants and normalize matching awards."""

    def __init__(self, email: str | None = None) -> None:
        """Create a connector.

        Args:
            email: Optional contact email forwarded so Europe PMC can identify
                polite API traffic.
        """
        self._email = email

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Europe PMC grant documents matching a query.

        Args:
            query: Free-text or fielded Grist query (for example ``malaria`` or
                ``ga:BBSRC``).
            max_results: Maximum number of grant documents to return.

        Returns:
            Normalized grant documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        documents: list[Document] = []
        page = 1
        pages_needed = max(1, (max_results + _PAGE_SIZE - 1) // _PAGE_SIZE)

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            while page <= pages_needed and len(documents) < max_results:
                payload = await self._fetch_page(client, stripped, page)
                page_docs = self._parse_results(payload, max_results - len(documents))
                if not page_docs:
                    break
                documents.extend(page_docs)
                page += 1

        return documents[:max_results]

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        query: str,
        page: int,
    ) -> object:
        """Fetch one Grist results page, returning {} on failure."""
        # Grist embeds ``query=`` in the path (not as a normal ``?query=`` param).
        safe_chars = ":' "
        url = (
            f"{EUROPEPMC_GRANTS_BASE}query={quote(query, safe=safe_chars)}"
            f"&resultType=core&format=json&page={page}"
        )
        if self._email:
            url = f"{url}&email={quote(self._email)}"
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Grist JSON payload into documents."""
        if not isinstance(payload, dict):
            return []
        record_list = payload.get("RecordList")
        if not isinstance(record_list, dict):
            return []
        records = record_list.get("Record")
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            return []

        documents: list[Document] = []
        for item in records:
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
        """Build a document from one Grist ``Record`` object."""
        grant = cls._as_dict(item.get("Grant"))
        title = cls._as_str(grant.get("Title")).strip()
        grant_id = cls._as_str(grant.get("Id")).strip() or cls._as_str(grant.get("Alias")).strip()
        if not title and not grant_id:
            return None
        if not title:
            title = f"Europe PMC grant {grant_id}"

        funder = cls._as_dict(grant.get("Funder"))
        funder_name = cls._as_str(funder.get("Name")).strip()
        fundref_id = cls._as_str(funder.get("FundRefID")).strip()
        abstract = cls._extract_abstract(grant.get("Abstract"))
        grant_type = cls._as_str(grant.get("Type")).strip()
        stream = cls._as_str(grant.get("Stream")).strip()
        category = cls._as_str(grant.get("Category")).strip()
        start_date = cls._as_str(grant.get("StartDate")).strip()
        end_date = cls._as_str(grant.get("EndDate")).strip()
        doi = cls._as_str(grant.get("Doi")).strip()
        amount = cls._format_amount(grant.get("Amount"))

        person = cls._as_dict(item.get("Person"))
        pi_name = cls._format_person(person)
        institution = cls._as_dict(item.get("Institution"))
        institution_name = cls._as_str(institution.get("Name")).strip()
        ror = cls._as_str(institution.get("RORID")).strip()

        source = (
            f"https://doi.org/{doi}"
            if doi
            else (
                f"https://europepmc.org/grantFinder/result?query=gid:{grant_id}"
                if grant_id
                else title
            )
        )
        text = cls._build_text(
            title=title,
            abstract=abstract,
            funder_name=funder_name,
            grant_id=grant_id,
            pi_name=pi_name,
            institution_name=institution_name,
            grant_type=grant_type,
            stream=stream,
            category=category,
            start_date=start_date,
            end_date=end_date,
            amount=amount,
            doi=doi,
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "europepmc_grants",
                "grant_id": grant_id,
                "doi": doi,
                "funder": funder_name,
                "fundref_id": fundref_id,
                "pi": pi_name,
                "institution": institution_name,
                "ror": ror,
                "grant_type": grant_type,
                "stream": stream,
                "category": category,
                "start_date": start_date,
                "end_date": end_date,
                "amount": amount,
                "abstract": abstract,
            },
        )

    @staticmethod
    def _extract_abstract(value: object) -> str:
        """Return grant abstract text from a string or ``{value: ...}`` object."""
        if isinstance(value, dict):
            return EuropePmcGrantsConnector._as_str(value.get("value")).strip()
        return EuropePmcGrantsConnector._as_str(value).strip()

    @staticmethod
    def _format_amount(value: object) -> str:
        """Format a Grist Amount object into a compact currency string."""
        if not isinstance(value, dict):
            return EuropePmcGrantsConnector._as_str(value).strip()
        amount = EuropePmcGrantsConnector._as_str(value.get("value")).strip()
        currency = EuropePmcGrantsConnector._as_str(value.get("Currency")).strip()
        if amount and currency:
            return f"{amount} {currency}"
        return amount or currency

    @staticmethod
    def _format_person(person: dict[str, object]) -> str:
        """Compose a PI display name from Grist Person fields."""
        given = EuropePmcGrantsConnector._as_str(person.get("GivenName")).strip()
        family = EuropePmcGrantsConnector._as_str(person.get("FamilyName")).strip()
        initials = EuropePmcGrantsConnector._as_str(person.get("Initials")).strip()
        if given and family:
            return f"{given} {family}"
        if initials and family:
            return f"{initials} {family}"
        return family or given

    @staticmethod
    def _build_text(
        *,
        title: str,
        abstract: str,
        funder_name: str,
        grant_id: str,
        pi_name: str,
        institution_name: str,
        grant_type: str,
        stream: str,
        category: str,
        start_date: str,
        end_date: str,
        amount: str,
        doi: str,
    ) -> str:
        """Compose searchable text for a Europe PMC grant record."""
        parts: list[str] = [f"Europe PMC grant {title}."]
        if abstract:
            parts.append(abstract)
        if funder_name:
            parts.append(f"Funder: {funder_name}.")
        if grant_id:
            parts.append(f"Grant id: {grant_id}.")
        if pi_name:
            parts.append(f"PI: {pi_name}.")
        if institution_name:
            parts.append(f"Institution: {institution_name}.")
        if grant_type:
            parts.append(f"Type: {grant_type}.")
        if stream:
            parts.append(f"Stream: {stream}.")
        if category:
            parts.append(f"Category: {category}.")
        if start_date or end_date:
            parts.append(f"Dates: {start_date or '?'} to {end_date or '?'}.")
        if amount:
            parts.append(f"Amount: {amount}.")
        if doi:
            parts.append(f"DOI: {doi}.")
        return " ".join(parts)

    @staticmethod
    def _as_dict(value: object) -> dict[str, object]:
        """Return a dict value or an empty dict."""
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar Grist values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return str(value)
        return ""
