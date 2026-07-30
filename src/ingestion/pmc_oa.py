"""PMC Open Access package link discovery connector.

PubMed Central (PMC) exposes an Open Access (OA) web service at
``https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi`` that resolves a PMCID to
downloadable full-text package URLs (typically ``tgz`` article packages and,
when available, ``pdf`` files on NCBI FTP). This complements
:class:`ingestion.pmc.PmcConnector`, which searches and fetches article XML via
E-utilities, by focusing on OA package link discovery for known PMCIDs.
"""

from __future__ import annotations

import re

import httpx
from defusedxml import ElementTree

from ingestion.chunking import stable_id
from retrieval.models import Document

OA_SERVICE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PMC_ARTICLE_BASE_URL = "https://pmc.ncbi.nlm.nih.gov/articles"
_RESULT_CAP = 20
_PMCID_PATTERN = re.compile(
    r"(?:https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/"
    r"|https?://pmc\.ncbi\.nlm\.nih\.gov/articles/)(?:PMC)?(\d+)"
    r"|pmc[:\s#_-]*(\d+)"
    r"|(PMC\d+)",
    re.IGNORECASE,
)
_YEAR_IN_CITATION_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


class PmcOaPackageConnector:
    """Resolve PMCID queries to NCBI OA package and PDF download links."""

    def __init__(self, email: str | None = None) -> None:
        """Create a connector with optional NCBI contact metadata.

        Args:
            email: Optional contact email forwarded to NCBI for polite API use.
        """
        self._email = (email or "").strip()

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return documents carrying OA package links for PMCIDs in a query.

        Args:
            query: PMCID-like string or free text containing one or more PMC
                identifiers (``PMC123``, ``pmc:123``, or article URLs).
            max_results: Maximum number of PMCID lookups to issue. Requests are
                bounded by the connector cap of 20 records per call.

        Returns:
            Normalized documents with package/PDF URLs in metadata. A blank
            query, non-positive ``max_results``, or query with no PMCID
            identifiers returns an empty list without HTTP requests.
        """
        if max_results <= 0 or not query.strip():
            return []

        pmcids = self._extract_pmcids(query)[: min(max_results, _RESULT_CAP)]
        if not pmcids:
            return []

        documents: list[Document] = []
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for pmcid in pmcids:
                payload = await self._fetch_oa_xml(client, pmcid)
                document = self._build_document(payload, pmcid)
                if document is not None:
                    documents.append(document)
                if len(documents) >= max_results:
                    break
        return documents

    async def _fetch_oa_xml(self, client: httpx.AsyncClient, pmcid: str) -> str | None:
        """Fetch one OA service XML response, returning ``None`` on failure."""
        params: dict[str, str] = {"id": pmcid}
        if self._email:
            params["email"] = self._email
        try:
            response = await client.get(OA_SERVICE_URL, params=params)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError:
            return None

    @classmethod
    def _build_document(cls, xml_text: str | None, requested_pmcid: str) -> Document | None:
        """Build a document from one OA service XML payload."""
        if not xml_text:
            return None

        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return None

        error = root.find("error")
        if error is not None:
            return None

        record = root.find(".//record")
        if record is None:
            return None

        pmcid = cls._normalize_pmcid(record.get("id") or requested_pmcid)
        if not pmcid:
            return None

        citation = (record.get("citation") or "").strip()
        license_name = (record.get("license") or "").strip()
        retracted = (record.get("retracted") or "").strip().lower()
        links = cls._extract_links(record)
        package_url = links.get("tgz", "")
        pdf_url = links.get("pdf", "")
        formats = ",".join(sorted(links))
        year = cls._year_from_citation(citation)
        title = citation or f"PMC OA package for {pmcid}"
        source = pdf_url or package_url or f"{PMC_ARTICLE_BASE_URL}/{pmcid}/"
        text = cls._build_text(pmcid, citation, license_name, retracted, package_url, pdf_url)

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "pmc_oa",
                "pmcid": pmcid,
                "citation": citation,
                "license": license_name,
                "retracted": retracted,
                "year": year,
                "package_url": package_url,
                "pdf_url": pdf_url,
                "formats": formats,
            },
        )

    @classmethod
    def _extract_links(cls, record: object) -> dict[str, str]:
        """Extract format -> HTTPS URL mappings from OA ``link`` elements."""
        findall = getattr(record, "findall", None)
        if findall is None:
            return {}
        links: dict[str, str] = {}
        for link in findall("./link"):
            get = getattr(link, "get", None)
            if get is None:
                continue
            fmt = (get("format") or "").strip().lower()
            href = cls._normalize_ftp_url((get("href") or "").strip())
            if fmt and href:
                links[fmt] = href
        return links

    @staticmethod
    def _normalize_ftp_url(url: str) -> str:
        """Prefer HTTPS NCBI FTP endpoints over bare ``ftp://`` URLs."""
        if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
            return "https://" + url[len("ftp://") :]
        if url.startswith("ftp://ftp.ncbi.nih.gov/"):
            return "https://" + url[len("ftp://") :]
        return url

    @staticmethod
    def _extract_pmcids(query: str) -> list[str]:
        """Extract unique normalized PMCIDs from free text."""
        pmcids: list[str] = []
        seen: set[str] = set()
        for match in _PMCID_PATTERN.finditer(query):
            raw = next((group for group in match.groups() if group), "")
            pmcid = PmcOaPackageConnector._normalize_pmcid(raw)
            if pmcid and pmcid not in seen:
                seen.add(pmcid)
                pmcids.append(pmcid)
        return pmcids

    @staticmethod
    def _normalize_pmcid(pmcid: str) -> str:
        """Return a PMCID with the canonical ``PMC`` prefix."""
        value = pmcid.strip()
        if not value:
            return ""
        if value.upper().startswith("PMC"):
            suffix = value[3:]
            if not suffix.isdigit():
                return ""
            return f"PMC{suffix}"
        if value.isdigit():
            return f"PMC{value}"
        return ""

    @staticmethod
    def _year_from_citation(citation: str) -> str:
        """Best-effort year extraction from an OA citation string."""
        match = _YEAR_IN_CITATION_PATTERN.search(citation)
        return match.group(0) if match else ""

    @staticmethod
    def _build_text(
        pmcid: str,
        citation: str,
        license_name: str,
        retracted: str,
        package_url: str,
        pdf_url: str,
    ) -> str:
        """Compose searchable text summarizing OA package availability."""
        parts = [f"PMC OA package for {pmcid}"]
        if citation:
            parts.append(citation)
        if license_name and license_name.lower() != "none":
            parts.append(f"License: {license_name}")
        if retracted:
            parts.append(f"Retracted: {retracted}")
        if package_url:
            parts.append(f"Package: {package_url}")
        if pdf_url:
            parts.append(f"PDF: {pdf_url}")
        return " ".join(parts)
