"""ORCID works summaries ingestion connector.

ORCID (https://orcid.org) exposes author-curated public work summaries under
``https://pub.orcid.org/v3.0/{orcid}/works``. This connector focuses on
summary-oriented normalization for a known ORCID iD (put-code, work type,
journal, external ids) without year/type deep filters. Distinct from:

* ``orcid.py`` — keyword profile search + token-filtered works
* ``orcid_works_filter.py`` — year / work-type deep filters
* ``orcid_employments.py`` — employment affiliations

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import httpx

from ingestion.orcid import OrcidConnector
from retrieval.models import Document

ORCID_API_BASE = "https://pub.orcid.org/v3.0"


class OrcidWorksSummariesConnector:
    """Fetch ORCID public work summaries for an ORCID iD."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return work-summary documents for an ORCID iD (bare or URL).

        Blank queries, non-ORCID queries, non-positive limits, failed requests,
        and malformed payloads yield an empty list.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        orcid_id = OrcidConnector._extract_orcid_id(stripped)
        if not orcid_id:
            return []

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        ) as client:
            payload = await self._fetch_works(client, orcid_id)
        return self._parse_works(payload, max_results, orcid_id)

    async def _fetch_works(self, client: httpx.AsyncClient, orcid_id: str) -> object:
        """Fetch public work summaries for one ORCID iD."""
        try:
            response = await client.get(f"{ORCID_API_BASE}/{orcid_id}/works")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_works(
        cls,
        payload: object,
        max_results: int,
        orcid_id: str,
    ) -> list[Document]:
        """Parse ORCID works into summary-oriented documents."""
        if max_results <= 0 or not isinstance(payload, dict):
            return []
        groups = payload.get("group")
        if not isinstance(groups, list):
            return []

        documents: list[Document] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            summaries = OrcidConnector._as_list(group.get("work-summary"))
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                document = cls._build_document(summary, orcid_id)
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
    ) -> Document | None:
        """Build a summary-oriented ORCID work document."""
        base = OrcidConnector._build_document(summary, orcid_id, orcid_id)
        if base is None:
            return None

        external_ids = OrcidConnector._extract_external_ids(summary.get("external-ids"))
        external_id_labels = [
            f"{entry['type']}:{entry['value']}" for entry in external_ids if entry.get("value")
        ]
        work_type = OrcidConnector._as_str(summary.get("type")).strip()
        journal = OrcidConnector._extract_value(summary.get("journal-title"))
        year = OrcidConnector._extract_year(summary.get("publication-date"))
        put_code = OrcidConnector._as_str(summary.get("put-code")).strip()
        doi = OrcidConnector._preferred_external_id(external_ids, "doi")

        text_parts = [f"ORCID work summary: {base.title}."]
        if work_type:
            text_parts.append(f"Type: {work_type}.")
        if journal:
            text_parts.append(f"Journal: {journal}.")
        if year:
            text_parts.append(f"Year: {year}.")
        if doi:
            text_parts.append(f"DOI: {doi}.")
        if external_id_labels:
            text_parts.append(f"External ids: {', '.join(external_id_labels)}.")
        text_parts.append(f"ORCID: {orcid_id}.")

        metadata = dict(base.metadata)
        metadata["source_type"] = "orcid_works_summaries"
        metadata["external_ids"] = ", ".join(external_id_labels)
        metadata["orcid"] = orcid_id
        metadata["put_code"] = put_code

        return Document(
            document_id=base.document_id,
            title=base.title,
            text=" ".join(text_parts),
            source=base.source,
            metadata=metadata,
        )
