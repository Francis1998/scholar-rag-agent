"""ClinicalTrials.gov registry ingestion connector.

ClinicalTrials.gov (https://clinicaltrials.gov) is the U.S. National Library of
Medicine registry of clinical studies. Its public API v2 search endpoint accepts
a free-text ``query.term`` and returns study records under ``studies``, each
carrying an NCT ID, brief/official title, brief summary, overall status, start
date, conditions, study type/phases, and lead sponsor. One request can ingest
several trial registrations for a topic, and the endpoint is unauthenticated.
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

CLINICALTRIALS_SEARCH_URL = "https://clinicaltrials.gov/api/v2/studies"
_PAGE_SIZE_CAP = 100
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})")
_STUDY_URL_TEMPLATE = "https://clinicaltrials.gov/study/{nct_id}"


class ClinicalTrialsConnector:
    """Search ClinicalTrials.gov and normalize matching studies into documents."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized ClinicalTrials.gov documents matching a query.

        Args:
            query: Free-text ClinicalTrials.gov query.
            max_results: Maximum number of studies to fetch (``pageSize`` is
                capped at 100).

        Returns:
            Normalized documents for the matching studies. An empty list is
            returned when the query is blank or matches nothing.
        """
        if max_results <= 0 or not query.strip():
            return []

        params: dict[str, str | int] = {
            "query.term": query.strip(),
            "pageSize": min(max_results, _PAGE_SIZE_CAP),
            "format": "json",
        }

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(CLINICALTRIALS_SEARCH_URL, params=params)
            response.raise_for_status()

        return self._parse_results(response.json(), max_results)

    @classmethod
    def _parse_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse a ClinicalTrials.gov search JSON payload into documents.

        Args:
            payload: Decoded ClinicalTrials.gov response.
            max_results: Upper bound on the number of documents returned.

        Returns:
            Normalized documents for each study in the ``studies`` list.
        """
        studies = cls._extract_studies(payload)
        documents: list[Document] = []
        for item in studies:
            if not isinstance(item, dict):
                continue
            document = cls._build_document(item)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _extract_studies(cls, payload: object) -> list[object]:
        """Return study objects from a ClinicalTrials.gov search payload."""
        if not isinstance(payload, dict):
            return []
        studies = payload.get("studies")
        if isinstance(studies, list):
            return studies
        return []

    @classmethod
    def _build_document(cls, study: dict[str, object]) -> Document | None:
        """Build a document from one ClinicalTrials.gov study record.

        Args:
            study: A single study object from the search response.

        Returns:
            Normalized document, or None when the study carries no usable title
            or NCT ID.
        """
        protocol = study.get("protocolSection")
        if not isinstance(protocol, dict):
            return None

        identification = cls._as_dict(protocol.get("identificationModule"))
        nct_id = cls._as_str(identification.get("nctId")).strip()
        brief_title = cls._as_str(identification.get("briefTitle")).strip()
        official_title = cls._as_str(identification.get("officialTitle")).strip()
        title = brief_title or official_title
        if not title or not nct_id:
            return None

        description = cls._as_dict(protocol.get("descriptionModule"))
        summary = " ".join(cls._as_str(description.get("briefSummary")).split())

        status_module = cls._as_dict(protocol.get("statusModule"))
        overall_status = cls._as_str(status_module.get("overallStatus")).strip()
        year = cls._extract_year(status_module.get("startDateStruct"))

        conditions_module = cls._as_dict(protocol.get("conditionsModule"))
        conditions = cls._extract_string_list(conditions_module.get("conditions"))

        design = cls._as_dict(protocol.get("designModule"))
        study_type = cls._as_str(design.get("studyType")).strip()
        phases = cls._extract_string_list(design.get("phases"))

        sponsors = cls._as_dict(protocol.get("sponsorCollaboratorsModule"))
        lead_sponsor = cls._as_dict(sponsors.get("leadSponsor"))
        sponsor_name = cls._as_str(lead_sponsor.get("name")).strip()

        source = _STUDY_URL_TEMPLATE.format(nct_id=nct_id)
        text = (
            summary
            if summary
            else cls._build_descriptor(
                overall_status,
                conditions,
                study_type,
                phases,
                sponsor_name,
                year,
            )
        )
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "clinicaltrials",
                "nct_id": nct_id,
                "year": year,
                "overall_status": overall_status,
                "conditions": ", ".join(conditions),
                "study_type": study_type,
                "phases": ", ".join(phases),
                "lead_sponsor": sponsor_name,
            },
        )

    @staticmethod
    def _extract_year(start_date_struct: object) -> str:
        """Extract a four-digit year from a ClinicalTrials.gov start date struct.

        Args:
            start_date_struct: The raw ``startDateStruct`` object.

        Returns:
            The four-digit year string, or an empty string when absent/invalid.
        """
        if not isinstance(start_date_struct, dict):
            return ""
        date_value = start_date_struct.get("date")
        if not isinstance(date_value, str):
            return ""
        match = _YEAR_PREFIX_PATTERN.match(date_value.strip())
        return match.group(1) if match else ""

    @classmethod
    def _extract_string_list(cls, value: object) -> list[str]:
        """Extract ordered non-empty strings from a list field."""
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for entry in value:
            text = cls._as_str(entry).strip()
            if text:
                items.append(text)
        return items

    @staticmethod
    def _build_descriptor(
        overall_status: str,
        conditions: list[str],
        study_type: str,
        phases: list[str],
        sponsor_name: str,
        year: str,
    ) -> str:
        """Compose a descriptor used when no brief summary exists."""
        parts: list[str] = []
        if overall_status:
            parts.append(f"Status: {overall_status}")
        if conditions:
            parts.append("Conditions: " + ", ".join(conditions))
        if study_type:
            parts.append(f"Type: {study_type}")
        if phases:
            parts.append("Phases: " + ", ".join(phases))
        if sponsor_name:
            parts.append(f"Sponsor: {sponsor_name}")
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    @staticmethod
    def _as_dict(value: object) -> dict[str, object]:
        """Return a dict value or an empty dict."""
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce a scalar ClinicalTrials.gov field value to a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
