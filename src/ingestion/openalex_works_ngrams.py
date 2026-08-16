"""OpenAlex work n-grams ingestion connector.

OpenAlex exposes statistically salient phrases for individual works at:

``GET https://api.openalex.org/works/{id}/ngrams``

The connector accepts OpenAlex work ids/URLs and DOI ids/URLs, then normalizes
each returned n-gram into a separate :class:`Document`. It is distinct from
``openalex.py``, which normalizes complete work records and abstracts.

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from urllib.parse import quote

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_WORK_ID_PATTERN = re.compile(r"^W\d+$", re.IGNORECASE)
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


class OpenAlexWorksNgramsConnector:
    """Fetch and normalize the n-grams associated with one OpenAlex work."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector with an optional OpenAlex polite-pool email."""
        self._mailto = mailto.strip() if mailto and mailto.strip() else None

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return n-gram documents for an OpenAlex work id, URL, or DOI."""
        if max_results <= 0:
            return []
        identifier, work_id, doi = self._normalize_identifier(query)
        if not identifier:
            return []

        request_url = f"{OPENALEX_WORKS_URL}/{quote(identifier, safe='')}/ngrams"
        params = {"mailto": self._mailto} if self._mailto else None
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            payload = await self._fetch_payload(client, request_url, params)
        return self._parse_ngrams(payload, max_results, request_url, identifier, work_id, doi)

    @staticmethod
    async def _fetch_payload(
        client: httpx.AsyncClient,
        request_url: str,
        params: dict[str, str] | None,
    ) -> object:
        """Fetch an OpenAlex n-grams payload, returning empty data on failure."""
        try:
            response = await client.get(request_url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _normalize_identifier(cls, query: str) -> tuple[str, str, str]:
        """Return API identifier, bare OpenAlex work id, and DOI."""
        candidate = query.strip().rstrip("/")
        if not candidate:
            return "", "", ""

        lowered = candidate.lower()
        if lowered.startswith(("https://openalex.org/", "http://openalex.org/")):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        elif "api.openalex.org/works/" in lowered:
            candidate = candidate.split("/works/", maxsplit=1)[-1].split("/", maxsplit=1)[0]

        if _WORK_ID_PATTERN.fullmatch(candidate):
            work_id = candidate.upper()
            return work_id, work_id, ""

        doi = candidate
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if doi.lower().startswith(prefix):
                doi = doi[len(prefix) :]
                break
        doi = doi.strip()
        if _DOI_PATTERN.fullmatch(doi):
            normalized_doi = doi.lower()
            return f"https://doi.org/{normalized_doi}", "", normalized_doi
        return "", "", ""

    @classmethod
    def _parse_ngrams(
        cls,
        payload: object,
        max_results: int,
        request_url: str,
        queried_identifier: str,
        work_id: str,
        doi: str,
    ) -> list[Document]:
        """Parse an OpenAlex n-grams response into phrase documents."""
        if not isinstance(payload, dict):
            return []
        ngrams = payload.get("ngrams")
        if not isinstance(ngrams, list):
            return []

        documents: list[Document] = []
        for index, item in enumerate(ngrams):
            if not isinstance(item, dict):
                continue
            ngram = cls._as_str(item.get("ngram")).strip()
            if not ngram:
                continue
            document_anchor = f"{request_url}#{index}:{ngram}"
            documents.append(
                Document(
                    document_id=stable_id(document_anchor, "doc"),
                    title=f"OpenAlex n-gram: {ngram}",
                    text=ngram,
                    source=request_url,
                    metadata={
                        "source_type": "openalex_works_ngrams",
                        "queried_identifier": queried_identifier,
                        "openalex_work_id": work_id,
                        "doi": doi,
                        "ngram": ngram,
                        "ngram_tokens": cls._as_str(item.get("ngram_tokens")),
                        "ngram_count": cls._as_str(item.get("ngram_count")),
                        "term_frequency": cls._as_str(item.get("term_frequency")),
                    },
                )
            )
            if len(documents) >= max_results:
                break
        return documents

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar OpenAlex n-gram values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return ""
