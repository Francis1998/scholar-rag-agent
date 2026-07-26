"""ORCID Public API ingestion connector.

ORCID (https://orcid.org) provides persistent researcher identifiers and public
record metadata. It complements paper-first sources (arXiv, Semantic Scholar,
OpenAlex, PubMed, Crossref, Europe PMC, DOAJ, DBLP, HAL, OpenAIRE, Zenodo,
Figshare, CORE, bioRxiv/medRxiv, NASA ADS) with author-curated work summaries
that can be looked up either through public profile search or a known ORCID iD.

The public API exposes unauthenticated JSON endpoints under
``https://pub.orcid.org/v3.0``. Keyword search uses ``expanded-search`` to find
public ORCID records, then fetches each record's ``works`` endpoint and filters
work summaries by query tokens. ORCID iD queries (for example
``0000-0002-1825-0097`` or ``https://orcid.org/0000-0002-1825-0097``) fetch the
works endpoint directly.
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

ORCID_API_BASE = "https://pub.orcid.org/v3.0"
ORCID_EXPANDED_SEARCH_URL = f"{ORCID_API_BASE}/expanded-search/"
_PAGE_SIZE_CAP = 100
_ORCID_ID_PATTERN = re.compile(
    r"(?:https?://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])\b",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class OrcidConnector:
    """Search ORCID public records and normalize their work summaries."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized ORCID work documents for a keyword or ORCID iD.

        Args:
            query: Free-text ORCID search query, or a bare/URL ORCID iD.
            max_results: Maximum number of work summaries to return.

        Returns:
            Normalized documents for public ORCID works. An empty list is
            returned when the query is blank, ``max_results`` is non-positive,
            no public profiles match, or matching profiles expose no works.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        ) as client:
            orcid_id = self._extract_orcid_id(stripped)
            if orcid_id:
                payload = await self._fetch_works(client, orcid_id)
                return self._parse_works(payload, max_results, orcid_id, orcid_id)

            search_payload = await self._search_records(client, stripped, max_results)
            candidates = self._extract_candidates(search_payload)
            tokens = self._tokens(stripped)
            documents: list[Document] = []
            seen_document_ids: set[str] = set()
            for candidate_orcid, candidate_name in candidates:
                works_payload = await self._fetch_works(client, candidate_orcid)
                for document in self._parse_works(
                    works_payload,
                    max_results - len(documents),
                    candidate_orcid,
                    candidate_name or candidate_orcid,
                    tokens,
                ):
                    if document.document_id in seen_document_ids:
                        continue
                    seen_document_ids.add(document.document_id)
                    documents.append(document)
                    if len(documents) >= max_results:
                        return documents
            return documents

    async def _search_records(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> object:
        """Fetch matching ORCID records via expanded search."""
        response = await client.get(
            ORCID_EXPANDED_SEARCH_URL,
            params={"q": query, "rows": min(max_results, _PAGE_SIZE_CAP)},
        )
        response.raise_for_status()
        return response.json()

    async def _fetch_works(self, client: httpx.AsyncClient, orcid_id: str) -> object:
        """Fetch public work summaries for one ORCID iD."""
        response = await client.get(f"{ORCID_API_BASE}/{orcid_id}/works")
        response.raise_for_status()
        return response.json()

    @classmethod
    def _parse_works(
        cls,
        payload: object,
        max_results: int,
        orcid_id: str,
        profile_name: str,
        filter_tokens: set[str] | None = None,
    ) -> list[Document]:
        """Parse an ORCID ``works`` payload into normalized documents."""
        if max_results <= 0 or not isinstance(payload, dict):
            return []
        groups = payload.get("group")
        if not isinstance(groups, list):
            return []

        documents: list[Document] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            summaries = cls._as_list(group.get("work-summary"))
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                if filter_tokens and not cls._matches(summary, filter_tokens):
                    continue
                document = cls._build_document(summary, orcid_id, profile_name)
                if document is not None:
                    documents.append(document)
                if len(documents) >= max_results:
                    return documents
        return documents

    @classmethod
    def _build_document(
        cls,
        summary: dict[str, object],
        orcid_id: str,
        profile_name: str,
    ) -> Document | None:
        """Build a document from one ORCID work summary."""
        title = cls._extract_title(summary.get("title")).strip()
        if not title:
            return None
        external_ids = cls._extract_external_ids(summary.get("external-ids"))
        doi = cls._preferred_external_id(external_ids, "doi")
        year = cls._extract_year(summary.get("publication-date"))
        journal = cls._extract_value(summary.get("journal-title"))
        work_type = cls._as_str(summary.get("type")).strip()
        put_code = cls._as_str(summary.get("put-code")).strip()
        url = cls._extract_value(summary.get("url"))
        source = cls._resolve_source(url, doi, external_ids, orcid_id, put_code, title)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=cls._build_descriptor(title, profile_name, journal, work_type, doi, year),
            source=source,
            metadata={
                "source_type": "orcid",
                "orcid": orcid_id,
                "doi": doi,
                "year": year,
                "authors": profile_name,
                "journal": journal,
                "work_type": work_type,
                "put_code": put_code,
            },
        )

    @classmethod
    def _extract_candidates(cls, payload: object) -> list[tuple[str, str]]:
        """Extract ordered ``(orcid_id, display_name)`` pairs from search JSON."""
        if not isinstance(payload, dict):
            return []
        raw_results = payload.get("expanded-result")
        if not isinstance(raw_results, list):
            raw_results = payload.get("result")
        if not isinstance(raw_results, list):
            return []

        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            orcid_id = cls._extract_orcid_from_result(item)
            if not orcid_id or orcid_id in seen:
                continue
            seen.add(orcid_id)
            candidates.append((orcid_id, cls._extract_profile_name(item)))
        return candidates

    @classmethod
    def _extract_orcid_from_result(cls, item: dict[str, object]) -> str:
        """Extract an ORCID iD from expanded-search or search result shapes."""
        direct = cls._extract_orcid_id(cls._as_str(item.get("orcid-id")))
        if direct:
            return direct
        identifier = item.get("orcid-identifier")
        if isinstance(identifier, dict):
            for key in ("path", "uri"):
                value = cls._extract_orcid_id(cls._as_str(identifier.get(key)))
                if value:
                    return value
        return ""

    @classmethod
    def _extract_profile_name(cls, item: dict[str, object]) -> str:
        """Extract a readable profile name from an expanded-search result."""
        credit_name = cls._as_str(item.get("credit-name")).strip()
        if credit_name:
            return credit_name
        given = cls._as_str(item.get("given-names")).strip()
        family = cls._as_str(item.get("family-names")).strip()
        return " ".join(part for part in (given, family) if part)

    @classmethod
    def _extract_title(cls, title: object) -> str:
        """Extract the main work title from an ORCID title object."""
        if isinstance(title, str):
            return title
        if not isinstance(title, dict):
            return ""
        main_title = title.get("title")
        return cls._extract_value(main_title)

    @staticmethod
    def _extract_year(publication_date: object) -> str:
        """Extract the publication year from an ORCID date object."""
        if not isinstance(publication_date, dict):
            return ""
        year = publication_date.get("year")
        if isinstance(year, dict):
            value = year.get("value")
            if isinstance(value, str) and value.strip().isdigit():
                return value.strip()
            if isinstance(value, int) and not isinstance(value, bool):
                return str(value)
        return ""

    @classmethod
    def _extract_external_ids(cls, external_ids: object) -> list[dict[str, str]]:
        """Extract ORCID external identifiers as normalized dictionaries."""
        if not isinstance(external_ids, dict):
            return []
        raw_ids = cls._as_list(external_ids.get("external-id"))
        normalized: list[dict[str, str]] = []
        for raw_id in raw_ids:
            if not isinstance(raw_id, dict):
                continue
            identifier_type = cls._as_str(raw_id.get("external-id-type")).strip().lower()
            value = cls._as_str(raw_id.get("external-id-value")).strip()
            if not identifier_type or not value:
                continue
            url = cls._extract_value(raw_id.get("external-id-url"))
            normalized.append(
                {
                    "type": identifier_type,
                    "value": cls._normalize_external_value(identifier_type, value),
                    "url": url,
                }
            )
        return normalized

    @staticmethod
    def _preferred_external_id(external_ids: list[dict[str, str]], identifier_type: str) -> str:
        """Return the first external identifier value matching ``identifier_type``."""
        for external_id in external_ids:
            if external_id["type"] == identifier_type:
                return external_id["value"]
        return ""

    @classmethod
    def _resolve_source(
        cls,
        url: str,
        doi: str,
        external_ids: list[dict[str, str]],
        orcid_id: str,
        put_code: str,
        title: str,
    ) -> str:
        """Resolve a stable source URL for an ORCID work summary."""
        if url:
            return url
        if doi:
            return f"https://doi.org/{doi}"
        for external_id in external_ids:
            if external_id["url"]:
                return external_id["url"]
        if put_code:
            return f"https://orcid.org/{orcid_id}/work/{put_code}"
        return title

    @staticmethod
    def _build_descriptor(
        title: str,
        profile_name: str,
        journal: str,
        work_type: str,
        doi: str,
        year: str,
    ) -> str:
        """Compose searchable text for ORCID summaries, which lack abstracts."""
        parts = [title]
        if profile_name:
            parts.append(f"By {profile_name}")
        if journal:
            parts.append(f"in {journal}")
        if work_type:
            parts.append(f"type: {work_type}")
        if doi:
            parts.append(f"DOI {doi}")
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @classmethod
    def _matches(cls, summary: dict[str, object], query_tokens: set[str]) -> bool:
        """Return True when every query token appears in ORCID work metadata."""
        if not query_tokens:
            return False
        external_ids = cls._extract_external_ids(summary.get("external-ids"))
        haystack = " ".join(
            [
                cls._extract_title(summary.get("title")),
                cls._extract_value(summary.get("journal-title")),
                cls._as_str(summary.get("type")),
                " ".join(external_id["value"] for external_id in external_ids),
            ]
        ).lower()
        return all(token in haystack for token in query_tokens)

    @staticmethod
    def _tokens(query: str) -> set[str]:
        """Split a free-text query into lowercase alphanumeric tokens."""
        return set(_TOKEN_PATTERN.findall(query.lower()))

    @staticmethod
    def _extract_orcid_id(value: str) -> str:
        """Extract a normalized ORCID iD from a bare identifier or URL."""
        match = _ORCID_ID_PATTERN.search(value.strip())
        return match.group(1).upper() if match else ""

    @classmethod
    def _extract_value(cls, value_object: object) -> str:
        """Extract ORCID ``{'value': ...}`` objects or scalar string values."""
        if isinstance(value_object, dict):
            return cls._as_str(value_object.get("value")).strip()
        return cls._as_str(value_object).strip()

    @staticmethod
    def _normalize_external_value(identifier_type: str, value: str) -> str:
        """Normalize selected external identifier values for metadata/source use."""
        stripped = value.strip()
        if identifier_type == "doi":
            lower = stripped.lower()
            for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
                if lower.startswith(prefix):
                    return stripped[len(prefix) :]
        return stripped

    @staticmethod
    def _as_list(value: object) -> list[object]:
        """Return a list for ORCID fields that may collapse to a single object."""
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [value]

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar ORCID field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
