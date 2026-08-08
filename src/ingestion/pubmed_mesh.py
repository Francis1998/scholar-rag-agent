"""PubMed MeSH vocabulary ingestion connector.

NCBI MeSH (Medical Subject Headings) is the controlled vocabulary used to index
PubMed/MEDLINE. This connector searches the MeSH database via E-utilities
(``esearch`` + ``esummary``, ``db=mesh``) and normalizes each descriptor into a
:class:`Document` with UI, preferred name, and tree numbers in metadata.

Flow:

1. ``GET .../esearch.fcgi?db=mesh&term=...&retmode=json``
2. ``GET .../esummary.fcgi?db=mesh&id=...&retmode=json``
"""

from __future__ import annotations

import os

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE_URL}/esearch.fcgi"
ESUMMARY_URL = f"{EUTILS_BASE_URL}/esummary.fcgi"
MESH_BROWSER_URL = "https://meshb.nlm.nih.gov/record/ui"


class PubmedMeshConnector:
    """Search the NCBI MeSH vocabulary and normalize descriptor documents."""

    def __init__(self, api_key: str | None = None) -> None:
        """Create a connector with an optional NCBI API key.

        Args:
            api_key: Optional NCBI API key. When omitted, ``NCBI_API_KEY`` is
                read from the environment when present.
        """
        self._api_key = api_key or os.environ.get("NCBI_API_KEY", "").strip() or None

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized MeSH descriptor documents matching a query.

        Args:
            query: Free-text MeSH vocabulary search.
            max_results: Maximum number of descriptors to return.

        Returns:
            Normalized MeSH documents. Blank queries, non-positive
            ``max_results``, unavailable API responses, and malformed payloads
            yield an empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                uids = await self._esearch(client, stripped, max_results)
                if not uids:
                    return []
                payload = await self._esummary(client, uids)
        except (httpx.HTTPError, ValueError):
            return []

        return self._parse_summaries(payload, uids, max_results)

    async def _esearch(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> list[str]:
        """Resolve a MeSH query to Entrez UIDs."""
        response = await client.get(
            ESEARCH_URL,
            params=self._with_api_key(
                {
                    "db": "mesh",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                }
            ),
        )
        response.raise_for_status()
        return self._extract_uids(response.json(), max_results)

    async def _esummary(
        self,
        client: httpx.AsyncClient,
        uids: list[str],
    ) -> object:
        """Fetch MeSH descriptor summaries for the given UIDs."""
        response = await client.get(
            ESUMMARY_URL,
            params=self._with_api_key(
                {
                    "db": "mesh",
                    "id": ",".join(uids),
                    "retmode": "json",
                }
            ),
        )
        response.raise_for_status()
        return response.json()

    def _with_api_key(self, params: dict[str, str | int]) -> dict[str, str | int]:
        """Attach the NCBI API key to request params when configured."""
        if self._api_key:
            return {**params, "api_key": self._api_key}
        return params

    @staticmethod
    def _extract_uids(payload: object, max_results: int) -> list[str]:
        """Extract the UID list from an ``esearch`` JSON payload."""
        if not isinstance(payload, dict):
            return []
        result = payload.get("esearchresult")
        if not isinstance(result, dict):
            return []
        idlist = result.get("idlist")
        if not isinstance(idlist, list):
            return []
        uids = [str(uid) for uid in idlist if isinstance(uid, (str, int))]
        return uids[:max_results]

    @classmethod
    def _parse_summaries(
        cls,
        payload: object,
        uids: list[str],
        max_results: int,
    ) -> list[Document]:
        """Parse an ``esummary`` MeSH JSON payload into documents."""
        if not isinstance(payload, dict):
            return []
        result = payload.get("result")
        if not isinstance(result, dict):
            return []

        documents: list[Document] = []
        for uid in uids:
            item = result.get(uid)
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item, uid)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_document(cls, item: dict[str, object], uid: str) -> Document | None:
        """Build a document from one MeSH descriptor summary."""
        mesh_ui = cls._as_str(item.get("ds_meshui")).strip()
        terms = item.get("ds_meshterms")
        preferred_name = ""
        entry_terms: list[str] = []
        if isinstance(terms, list):
            for term in terms:
                cleaned = cls._as_str(term).strip()
                if not cleaned:
                    continue
                if not preferred_name:
                    preferred_name = cleaned
                else:
                    entry_terms.append(cleaned)
        elif isinstance(terms, str):
            preferred_name = terms.strip()

        if not preferred_name:
            return None

        tree_numbers = cls._extract_tree_numbers(item.get("ds_idxlinks"))
        tree_joined = ", ".join(tree_numbers)
        scope_note = cls._as_str(item.get("ds_scopenote")).strip()
        year_introduced = cls._as_str(item.get("ds_yearintroduced")).strip()
        record_type = cls._as_str(item.get("ds_recordtype")).strip()

        source = (
            f"{MESH_BROWSER_URL}/{mesh_ui}"
            if mesh_ui
            else (f"https://www.ncbi.nlm.nih.gov/mesh/{uid}" if uid else preferred_name)
        )
        text = cls._build_text(
            preferred_name,
            mesh_ui,
            tree_joined,
            scope_note,
            entry_terms[:12],
        )

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(preferred_name.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "pubmed_mesh",
                "mesh_ui": mesh_ui,
                "mesh_uid": uid,
                "name": preferred_name,
                "tree_numbers": tree_joined,
                "year_introduced": year_introduced,
                "record_type": record_type,
                "entry_terms": ", ".join(entry_terms[:12]),
            },
        )

    @staticmethod
    def _extract_tree_numbers(idxlinks: object) -> list[str]:
        """Extract ordered unique MeSH tree numbers from ``ds_idxlinks``."""
        if not isinstance(idxlinks, list):
            return []
        numbers: list[str] = []
        for link in idxlinks:
            if not isinstance(link, dict):
                continue
            treenum = PubmedMeshConnector._as_str(link.get("treenum")).strip()
            if treenum and treenum not in numbers:
                numbers.append(treenum)
        return numbers

    @staticmethod
    def _build_text(
        preferred_name: str,
        mesh_ui: str,
        tree_joined: str,
        scope_note: str,
        entry_terms: list[str],
    ) -> str:
        """Compose searchable text for a MeSH descriptor."""
        parts: list[str] = [f"MeSH descriptor {preferred_name}."]
        if mesh_ui:
            parts.append(f"UI: {mesh_ui}.")
        if tree_joined:
            parts.append(f"Tree numbers: {tree_joined}.")
        if scope_note:
            parts.append(scope_note)
        if entry_terms:
            parts.append(f"Entry terms: {', '.join(entry_terms)}.")
        return " ".join(parts)

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar MeSH field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
