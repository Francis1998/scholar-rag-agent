"""ORCID public employments ingestion connector.

ORCID exposes researcher-maintained public employment affiliations separately
from works. This connector resolves a researcher by ORCID iD or expanded public
profile search, calls ``/{orcid}/employments``, and normalizes employment
summaries into :class:`Document` objects. It is distinct from ``orcid.py`` and
``orcid_works_filter.py``, which ingest work summaries.

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import httpx

from ingestion.chunking import stable_id
from ingestion.orcid import OrcidConnector
from retrieval.models import Document

ORCID_API_BASE = "https://pub.orcid.org/v3.0"
ORCID_EXPANDED_SEARCH_URL = f"{ORCID_API_BASE}/expanded-search/"
_PAGE_SIZE_CAP = 100


class OrcidEmploymentsConnector:
    """Resolve researchers and normalize their public ORCID employments."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return public employment affiliations for a researcher.

        ``query`` may be a bare/URL ORCID iD or a public profile search. Blank
        queries, non-positive limits, unavailable endpoints, and malformed
        payloads yield an empty list.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        ) as client:
            orcid_id = OrcidConnector._extract_orcid_id(stripped)
            if orcid_id:
                payload = await self._fetch_employments(client, orcid_id)
                return self._parse_employments(payload, max_results, orcid_id, orcid_id)

            search_payload = await self._search_records(client, stripped, max_results)
            candidates = OrcidConnector._extract_candidates(search_payload)
            documents: list[Document] = []
            seen_document_ids: set[str] = set()
            for candidate_orcid, candidate_name in candidates:
                payload = await self._fetch_employments(client, candidate_orcid)
                for document in self._parse_employments(
                    payload,
                    max_results - len(documents),
                    candidate_orcid,
                    candidate_name or candidate_orcid,
                ):
                    if document.document_id in seen_document_ids:
                        continue
                    seen_document_ids.add(document.document_id)
                    documents.append(document)
                    if len(documents) >= max_results:
                        return documents
            return documents

    @staticmethod
    async def _search_records(
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> object:
        """Search public ORCID profiles, returning an empty payload on failure."""
        try:
            response = await client.get(
                ORCID_EXPANDED_SEARCH_URL,
                params={"q": query, "rows": min(max_results, _PAGE_SIZE_CAP)},
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @staticmethod
    async def _fetch_employments(client: httpx.AsyncClient, orcid_id: str) -> object:
        """Fetch one public ORCID employments section."""
        try:
            response = await client.get(f"{ORCID_API_BASE}/{orcid_id}/employments")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    @classmethod
    def _parse_employments(
        cls,
        payload: object,
        max_results: int,
        orcid_id: str,
        profile_name: str,
    ) -> list[Document]:
        """Parse grouped ORCID employment summaries into documents."""
        if max_results <= 0 or not isinstance(payload, dict):
            return []

        groups = payload.get("affiliation-group")
        if not isinstance(groups, list):
            direct = payload.get("employment-summary")
            groups = [{"employment-summary": direct}] if direct is not None else []

        documents: list[Document] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            for summary in cls._employment_summaries(group):
                document = cls._build_document(summary, orcid_id, profile_name)
                if document is not None:
                    documents.append(document)
                if len(documents) >= max_results:
                    return documents
        return documents

    @classmethod
    def _employment_summaries(cls, group: dict[str, object]) -> list[dict[str, object]]:
        """Return summaries from grouped and direct ORCID response shapes."""
        summaries: list[dict[str, object]] = []
        raw_summaries = group.get("summaries")
        for entry in OrcidConnector._as_list(raw_summaries):
            if not isinstance(entry, dict):
                continue
            summary = entry.get("employment-summary")
            if isinstance(summary, dict):
                summaries.append(summary)

        for entry in OrcidConnector._as_list(group.get("employment-summary")):
            if isinstance(entry, dict):
                summaries.append(entry)
        return summaries

    @classmethod
    def _build_document(
        cls,
        summary: dict[str, object],
        orcid_id: str,
        profile_name: str,
    ) -> Document | None:
        """Build a document from one ORCID employment summary."""
        organization = cls._as_dict(summary.get("organization"))
        organization_name = cls._as_str(organization.get("name")).strip()
        role_title = cls._as_str(summary.get("role-title")).strip()
        department_name = cls._as_str(summary.get("department-name")).strip()
        if not organization_name and not role_title:
            return None

        put_code = cls._as_str(summary.get("put-code")).strip()
        start_date = cls._format_date(summary.get("start-date"))
        end_date = cls._format_date(summary.get("end-date"))
        address = cls._as_dict(organization.get("address"))
        city = cls._as_str(address.get("city")).strip()
        region = cls._as_str(address.get("region")).strip()
        country = cls._as_str(address.get("country")).strip()
        location = ", ".join(value for value in (city, region, country) if value)

        disambiguated = cls._as_dict(organization.get("disambiguated-organization"))
        organization_id = cls._as_str(
            disambiguated.get("disambiguated-organization-identifier")
        ).strip()
        organization_id_source = cls._as_str(disambiguated.get("disambiguation-source")).strip()
        employment_url = OrcidConnector._extract_value(summary.get("url"))
        source = employment_url or (
            f"https://orcid.org/{orcid_id}/employment/{put_code}"
            if put_code
            else f"https://orcid.org/{orcid_id}"
        )
        title = cls._build_title(role_title, organization_name)

        return Document(
            document_id=stable_id(source, "doc"),
            title=title,
            text=cls._build_text(
                profile_name=profile_name,
                role_title=role_title,
                department_name=department_name,
                organization_name=organization_name,
                location=location,
                start_date=start_date,
                end_date=end_date,
                organization_id=organization_id,
            ),
            source=source,
            metadata={
                "source_type": "orcid_employments",
                "orcid": orcid_id,
                "profile_name": profile_name,
                "put_code": put_code,
                "role_title": role_title,
                "department": department_name,
                "organization": organization_name,
                "organization_id": organization_id,
                "organization_id_source": organization_id_source,
                "location": location,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    @staticmethod
    def _build_title(role_title: str, organization_name: str) -> str:
        """Compose a concise employment title."""
        if role_title and organization_name:
            return f"{role_title} at {organization_name}"
        return role_title or f"Employment at {organization_name}"

    @staticmethod
    def _build_text(
        *,
        profile_name: str,
        role_title: str,
        department_name: str,
        organization_name: str,
        location: str,
        start_date: str,
        end_date: str,
        organization_id: str,
    ) -> str:
        """Compose searchable text for an ORCID employment."""
        parts = [f"ORCID employment for {profile_name}."]
        if role_title:
            parts.append(f"Role: {role_title}.")
        if department_name:
            parts.append(f"Department: {department_name}.")
        if organization_name:
            parts.append(f"Organization: {organization_name}.")
        if location:
            parts.append(f"Location: {location}.")
        if start_date or end_date:
            parts.append(f"Dates: {start_date or '?'} to {end_date or 'present'}.")
        if organization_id:
            parts.append(f"Organization identifier: {organization_id}.")
        return " ".join(parts)

    @classmethod
    def _format_date(cls, value: object) -> str:
        """Format an ORCID partial date as YYYY[-MM[-DD]]."""
        date = cls._as_dict(value)
        parts: list[str] = []
        for key in ("year", "month", "day"):
            component = OrcidConnector._extract_value(date.get(key))
            if not component:
                break
            parts.append(component.zfill(2) if key != "year" else component)
        return "-".join(parts)

    @staticmethod
    def _as_dict(value: object) -> dict[str, object]:
        """Return a dict value or an empty dict."""
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar ORCID employment values to strings."""
        return OrcidConnector._as_str(value)
