"""Crossref Funder Registry ingestion connector.

Crossref's Open Funder Registry (https://www.crossref.org/services/funder-registry/)
is the authoritative index of funding organizations used in grant acknowledgements
and Crossref funding metadata. Searching ``GET https://api.crossref.org/funders``
returns matching funders with stable Open Funder Registry IDs (``10.13039/...``),
preferred names, alternate names, and locations — useful context for grant-aware
RAG alongside Crossref works, DataCite, OpenAlex, and Semantic Scholar.

Free-text queries use:

``GET https://api.crossref.org/funders?query=...&rows={n}``

Funder-id-shaped queries (bare registry ids such as ``100000001``,
``10.13039/100000001``, or ``https://doi.org/10.13039/100000001``) resolve a
single funder via:

``GET https://api.crossref.org/funders/{id}``
"""

from __future__ import annotations

import os
import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

CROSSREF_FUNDERS_URL = "https://api.crossref.org/funders"
_FUNDER_DOI_PREFIX = "10.13039/"
_FUNDER_DOI_PATTERN = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.13039/\d+)",
    re.IGNORECASE,
)
_BARE_FUNDER_ID_PATTERN = re.compile(r"^\d{4,}$")


class CrossrefFunderConnector:
    """Search the Crossref Funder Registry and normalize funders into documents."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional contact email added to requests so Crossref routes
                traffic to its faster, polite API pool. When omitted,
                ``CROSSREF_MAILTO`` (then ``OPENALEX_MAILTO``) is read from the
                environment when present.
        """
        self._mailto = (
            mailto
            or os.environ.get("CROSSREF_MAILTO", "").strip()
            or os.environ.get("OPENALEX_MAILTO", "").strip()
            or None
        )

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Funder Registry documents matching a query.

        Args:
            query: Free-text funder name search or an Open Funder Registry id /
                DOI-shaped identifier.
            max_results: Maximum number of funders to return for free-text
                search (ignored for single-id lookups beyond returning one).

        Returns:
            Normalized funder documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        funder_id = self._extract_funder_id(stripped)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            if funder_id is not None:
                payload = await self._fetch_funder(client, funder_id)
                return self._parse_single(payload)

            params: dict[str, str | int] = {"query": stripped, "rows": max_results}
            if self._mailto:
                params["mailto"] = self._mailto
            payload = await self._fetch_search(client, params)
        return self._parse_results(payload, max_results)

    async def _fetch_search(
        self,
        client: httpx.AsyncClient,
        params: dict[str, str | int],
    ) -> object:
        """Fetch a Funder Registry search payload, returning {} on API failure."""
        try:
            response = await client.get(CROSSREF_FUNDERS_URL, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    async def _fetch_funder(self, client: httpx.AsyncClient, funder_id: str) -> object:
        """Fetch one funder by Open Funder Registry id, returning {} on failure."""
        params: dict[str, str] = {}
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            response = await client.get(f"{CROSSREF_FUNDERS_URL}/{funder_id}", params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a Crossref ``funder-list`` JSON payload into documents.

        Args:
            payload: Decoded Crossref response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each funder item in the payload.
        """
        if not isinstance(payload, dict):
            return []
        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        items = message.get("items")
        if not isinstance(items, list):
            return []

        documents: list[Document] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _parse_single(cls, payload: object) -> list[Document]:
        """Parse a single Crossref ``funder`` JSON payload into documents.

        Args:
            payload: Decoded Crossref single-funder response.

        Returns:
            A one-element list when the funder is usable, otherwise empty.
        """
        if not isinstance(payload, dict):
            return []
        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        document = cls._build_document(message)
        return [document] if document is not None else []

    @classmethod
    def _build_document(cls, item: dict[str, object]) -> Document | None:
        """Build a document from one Crossref funder object.

        Args:
            item: A funder object from ``message.items`` or a single-funder
                ``message``.

        Returns:
            Normalized document, or None when the funder carries no usable name.
        """
        name = cls._as_str(item.get("name")).strip()
        if not name:
            return None

        funder_id = cls._as_str(item.get("id")).strip()
        uri = cls._as_str(item.get("uri")).strip()
        location = cls._as_str(item.get("location")).strip()
        alt_names = cls._extract_alt_names(item.get("alt-names"))
        work_count = cls._as_count(item.get("work-count"))
        descendant_work_count = cls._as_count(item.get("descendant-work-count"))
        source = cls._resolve_source(uri, funder_id, name)
        text = cls._build_descriptor(
            name=name,
            location=location,
            alt_names=alt_names,
            work_count=work_count,
            descendant_work_count=descendant_work_count,
        )
        metadata = {
            "source_type": "crossref_funder",
            "funder_id": funder_id,
            "uri": uri,
            "location": location,
            "alt_names": ", ".join(alt_names),
            "work_count": work_count,
            "descendant_work_count": descendant_work_count,
        }
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(name.split()),
            text=text,
            source=source,
            metadata=metadata,
        )

    @classmethod
    def _extract_funder_id(cls, query: str) -> str | None:
        """Extract an Open Funder Registry id from a query when present.

        Args:
            query: Stripped user query.

        Returns:
            Bare funder id (digits), or None when the query is free text.
        """
        doi_match = _FUNDER_DOI_PATTERN.fullmatch(query.strip())
        if doi_match:
            return doi_match.group(1).split("/", 1)[1]
        if _BARE_FUNDER_ID_PATTERN.fullmatch(query):
            return query
        return None

    @staticmethod
    def _extract_alt_names(value: object) -> list[str]:
        """Extract ordered alternate names from a Crossref funder field.

        Args:
            value: The ``alt-names`` field.

        Returns:
            Ordered non-empty alternate names.
        """
        if not isinstance(value, list):
            return []
        names: list[str] = []
        for entry in value:
            if isinstance(entry, str) and entry.strip():
                names.append(" ".join(entry.split()))
        return names

    @classmethod
    def _resolve_source(cls, uri: str, funder_id: str, name: str) -> str:
        """Resolve the canonical source URL for a funder record.

        Args:
            uri: Crossref-provided URI when present.
            funder_id: Bare Open Funder Registry id.
            name: Preferred funder name used as a last-resort anchor.

        Returns:
            A source string suitable for provenance and stable-id derivation.
        """
        if uri:
            return uri
        if funder_id:
            return f"https://doi.org/{_FUNDER_DOI_PREFIX}{funder_id}"
        return name

    @staticmethod
    def _build_descriptor(
        *,
        name: str,
        location: str,
        alt_names: list[str],
        work_count: str,
        descendant_work_count: str,
    ) -> str:
        """Compose searchable descriptor text for a funding organization.

        Args:
            name: Preferred funder name.
            location: Country/region when known.
            alt_names: Alternate names / acronyms.
            work_count: Works directly attributed to the funder, if known.
            descendant_work_count: Works including descendants, if known.

        Returns:
            A single-line descriptor for sparse/entity retrieval.
        """
        parts = [f"Funding organization: {name}"]
        if location:
            parts.append(f"Location: {location}")
        if alt_names:
            parts.append("Also known as: " + ", ".join(alt_names[:8]))
        if work_count:
            parts.append(f"Registered works: {work_count}")
        if descendant_work_count and descendant_work_count != work_count:
            parts.append(f"Descendant works: {descendant_work_count}")
        return ". ".join(parts) + "."

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar Crossref field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""

    @staticmethod
    def _as_count(value: object) -> str:
        """Coerce a Crossref count field to a digit string when present."""
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, str) and value.strip().isdigit():
            return value.strip()
        return ""
