"""DataCite client-id and prefix ingestion connector.

DataCite DOI records can be scoped to a repository client:

``GET https://api.datacite.org/dois?client-id={client_id}``

The connector also accepts DOI prefixes through the endpoint's ``prefix``
parameter, while prioritizing client-id filtering so it remains distinct from
``datacite_dois_prefix.py``. Free text can be scoped to a configured client.

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re

import httpx

from ingestion.datacite import DataCiteConnector
from ingestion.datacite_dois_prefix import DataCiteDoisPrefixConnector
from retrieval.models import Document

DATACITE_DOIS_URL = "https://api.datacite.org/dois"
_PAGE_SIZE_CAP = 100
_CLIENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*\.[a-z0-9][a-z0-9_.-]*$", re.IGNORECASE)


class DataCiteClientPrefixConnector(DataCiteConnector):
    """List and normalize DataCite DOI records by client id or DOI prefix."""

    def __init__(self, default_client_id: str | None = None) -> None:
        """Create a connector with an optional client scope for free text."""
        self._default_client_id = self._normalize_client_id(default_client_id or "")

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return DataCite works selected by client id, prefix, or scoped text."""
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        prefix = DataCiteDoisPrefixConnector._normalize_prefix(stripped) or ""
        client_id = "" if prefix else self._extract_client_query(stripped)
        params: dict[str, str | int] = {
            "page[size]": min(max_results, _PAGE_SIZE_CAP),
        }
        if client_id:
            params["client-id"] = client_id
        elif prefix:
            params["prefix"] = prefix
        else:
            params["query"] = stripped
            if self._default_client_id:
                params["client-id"] = self._default_client_id
                client_id = self._default_client_id

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, params)
        return self._parse_client_results(payload, max_results, client_id, prefix)

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch DataCite DOI resources, returning an empty payload on failure."""
        try:
            response = await client.get(DATACITE_DOIS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _extract_client_query(cls, query: str) -> str:
        """Extract a DataCite client id from raw or labeled input."""
        candidate = query.strip()
        lowered = candidate.lower()
        for prefix in ("client-id:", "client-id="):
            if lowered.startswith(prefix):
                candidate = candidate[len(prefix) :].strip()
                break
        return cls._normalize_client_id(candidate) or ""

    @staticmethod
    def _normalize_client_id(value: str) -> str | None:
        """Return a lowercase DataCite client id when syntactically valid."""
        candidate = value.strip().lower()
        if _CLIENT_ID_PATTERN.fullmatch(candidate):
            return candidate
        return None

    @classmethod
    def _parse_client_results(
        cls,
        payload: object,
        max_results: int,
        selected_client_id: str,
        selected_prefix: str,
    ) -> list[Document]:
        """Normalize DataCite resources with client and prefix provenance."""
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, list):
            return []

        documents: list[Document] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            document = DataCiteConnector._build_document(item)
            if document is None:
                continue
            doi = document.metadata.get("doi", "")
            document.metadata.update(
                {
                    "source_type": "datacite_client_prefix",
                    "client_id": cls._extract_item_client_id(item) or selected_client_id,
                    "doi_prefix": DataCiteDoisPrefixConnector._extract_prefix(doi)
                    or selected_prefix,
                }
            )
            documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _extract_item_client_id(cls, item: dict[str, object]) -> str:
        """Extract a client id from DataCite attributes or relationships."""
        attributes = item.get("attributes")
        if isinstance(attributes, dict):
            client_id = cls._normalize_client_id(
                DataCiteConnector._as_str(attributes.get("clientId"))
            )
            if client_id:
                return client_id

        relationships = item.get("relationships")
        if not isinstance(relationships, dict):
            return ""
        client = relationships.get("client")
        if not isinstance(client, dict):
            return ""
        data = client.get("data")
        if not isinstance(data, dict):
            return ""
        return cls._normalize_client_id(DataCiteConnector._as_str(data.get("id"))) or ""
