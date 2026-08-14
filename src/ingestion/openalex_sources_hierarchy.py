"""OpenAlex sources hierarchy ingestion connector.

OpenAlex sources expose venue identity together with host organization, source
type, and ISSN fields. This connector searches or resolves sources and
normalizes those values into an ancestry path for hierarchy-aware retrieval.
It is distinct from ``openalex_sources.py`` (general venue profiles) and
``openalex_topics_hierarchy.py`` (research-topic taxonomy).

Free-text queries use ``GET /sources?search=...`` and ``S####`` identifiers use
``GET /sources/{id}``.

Prefer frontier models for downstream synthesis: GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

from ingestion.openalex_sources import OpenAlexSourcesConnector
from retrieval.models import Document


class OpenAlexSourcesHierarchyConnector(OpenAlexSourcesConnector):
    """Search OpenAlex sources and normalize venue hierarchy paths."""

    @classmethod
    def _build_source_document(cls, item: dict[str, object]) -> Document | None:
        """Build a source document enriched with normalized hierarchy fields."""
        document = super()._build_source_document(item)
        if document is None:
            return None

        display_name = cls._as_str(item.get("display_name")).strip()
        source_type = cls._as_str(item.get("type")).strip()
        host_organization = cls._extract_openalex_id(item.get("host_organization"))
        host_organization_name = cls._as_str(item.get("host_organization_name")).strip()
        issns = cls._extract_issns(item.get("issn"))
        issn_l = cls._as_str(item.get("issn_l")).strip()
        primary_issn = issn_l or (issns[0] if issns else "")
        ancestry_path = cls._build_ancestry_path(
            host_organization_name or host_organization,
            source_type,
            primary_issn,
            display_name,
        )

        document.text = f"OpenAlex source hierarchy: {ancestry_path}. {document.text}".strip()
        document.metadata.update(
            {
                "source_type": "openalex_sources_hierarchy",
                "host_organization": host_organization,
                "host_organization_name": host_organization_name,
                "type": source_type,
                "issn_l": issn_l,
                "issn": ", ".join(issns),
                "ancestry_path": ancestry_path,
                "hierarchy_path": ancestry_path,
            }
        )
        return document

    @staticmethod
    def _extract_openalex_id(value: object) -> str:
        """Normalize a bare or URL-shaped OpenAlex entity identifier."""
        ref = OpenAlexSourcesHierarchyConnector._as_str(value).strip()
        if not ref:
            return ""
        return ref.rsplit("/", maxsplit=1)[-1].upper()

    @classmethod
    def _extract_issns(cls, value: object) -> list[str]:
        """Return non-empty, de-duplicated ISSNs in API order."""
        if not isinstance(value, list):
            return []
        issns: list[str] = []
        for entry in value:
            issn = cls._as_str(entry).strip()
            if issn and issn not in issns:
                issns.append(issn)
        return issns

    @staticmethod
    def _build_ancestry_path(
        host_organization: str,
        source_type: str,
        primary_issn: str,
        display_name: str,
    ) -> str:
        """Compose ``host > type > ISSN > source`` without duplicate nodes."""
        nodes: list[str] = []
        for value in (host_organization, source_type, primary_issn, display_name):
            normalized = " ".join(value.split())
            if normalized and normalized not in nodes:
                nodes.append(normalized)
        return " > ".join(nodes)
