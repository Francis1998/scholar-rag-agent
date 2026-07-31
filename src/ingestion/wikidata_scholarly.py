"""Wikidata scholarly entity ingestion connector.

Wikidata (https://www.wikidata.org) indexes scholarly articles, journals, and
related research entities. This connector searches the public MediaWiki Action
API with ``wbsearchentities``, enriches hits via ``wbgetentities``, and falls
back to a lightweight SPARQL query on the scholarly graph endpoint when entity
search returns no scholarly matches.

Entity search:

``GET https://www.wikidata.org/w/api.php?action=wbsearchentities&search=...``

Scholarly SPARQL-lite fallback:

``GET https://query-scholarly.wikidata.org/sparql?query=...``
"""

from __future__ import annotations

import re

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SCHOLARLY_SPARQL_URL = "https://query-scholarly.wikidata.org/sparql"
_PAGE_SIZE_CAP = 50
_QID_PATTERN = re.compile(r"^Q\d+$", re.IGNORECASE)
_SCHOLARLY_INSTANCE_IDS = frozenset(
    {
        "Q13442814",  # scholarly article
        "Q5633421",  # scientific journal article
        "Q18918145",  # academic journal article
        "Q571",  # book
        "Q7725634",  # literary work
    }
)


class WikidataScholarlyConnector:
    """Search Wikidata scholarly entities and normalize matching records."""

    async def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Return normalized Wikidata scholarly entity documents.

        Args:
            query: Free-text entity search or a Wikidata item id such as
                ``Q210272``.
            max_results: Maximum number of entity documents to return.

        Returns:
            Normalized documents for scholarly entities. Blank queries,
            non-positive ``max_results``, and unavailable API responses yield an
            empty list rather than raising.
        """
        stripped = query.strip()
        if max_results <= 0 or not stripped:
            return []

        qid = self._normalize_qid(stripped)
        if qid is not None:
            entity_payload = await self._fetch_entities([qid])
            return self._parse_entities(entity_payload, max_results)

        search_payload = await self._search_entities(stripped, max_results)
        entity_ids = self._extract_entity_ids(search_payload)
        if not entity_ids:
            sparql_payload = await self._sparql_search(stripped, max_results)
            return self._parse_sparql_results(sparql_payload, max_results)

        entity_payload = await self._fetch_entities(entity_ids[:max_results])
        documents = self._parse_entities(entity_payload, max_results)
        if documents:
            return documents

        sparql_payload = await self._sparql_search(stripped, max_results)
        return self._parse_sparql_results(sparql_payload, max_results)

    @staticmethod
    def _normalize_qid(query: str) -> str | None:
        """Return a bare Wikidata item id when ``query`` is QID-shaped."""
        candidate = query.strip()
        if candidate.lower().startswith("https://www.wikidata.org/wiki/"):
            candidate = candidate.rsplit("/", maxsplit=1)[-1]
        if _QID_PATTERN.fullmatch(candidate):
            return candidate.upper()
        return None

    async def _search_entities(self, query: str, max_results: int) -> object:
        """Search Wikidata entities via wbsearchentities."""
        params: dict[str, str | int] = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "type": "item",
            "limit": min(max_results, _PAGE_SIZE_CAP),
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            try:
                response = await client.get(WIKIDATA_API_URL, params=params)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError):
                return {}

    async def _fetch_entities(self, entity_ids: list[str]) -> object:
        """Fetch entity records via wbgetentities."""
        if not entity_ids:
            return {}
        params: dict[str, str | int] = {
            "action": "wbgetentities",
            "ids": "|".join(entity_ids),
            "props": "labels|descriptions|claims|sitelinks",
            "languages": "en",
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            try:
                response = await client.get(WIKIDATA_API_URL, params=params)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError):
                return {}

    async def _sparql_search(self, query: str, max_results: int) -> object:
        """Run a lightweight scholarly-graph SPARQL search fallback."""
        escaped = query.replace("\\", "\\\\").replace('"', '\\"')
        sparql = f"""
SELECT ?item ?itemLabel ?itemDescription ?doi WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:api "EntitySearch" .
    bd:serviceParam wikibase:endpoint "www.wikidata.org" .
    bd:serviceParam mwapi:search "{escaped}" .
    bd:serviceParam mwapi:language "en" .
    ?item wikibase:apiOutputItem mwapi:item .
    ?num wikibase:apiOrdinal true .
  }}
  OPTIONAL {{ ?item wdt:P356 ?doi . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}
ORDER BY ASC(?num)
LIMIT {min(max_results, _PAGE_SIZE_CAP)}
""".strip()
        headers = {"Accept": "application/sparql-results+json"}
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            try:
                response = await client.get(
                    WIKIDATA_SCHOLARLY_SPARQL_URL,
                    params={"query": sparql},
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError):
                return {}

    @classmethod
    def _extract_entity_ids(cls, payload: object) -> list[str]:
        """Return entity ids from a wbsearchentities payload."""
        if not isinstance(payload, dict):
            return []
        search = payload.get("search")
        if not isinstance(search, list):
            return []
        ids: list[str] = []
        for item in search:
            if not isinstance(item, dict):
                continue
            entity_id = cls._as_str(item.get("id")).strip().upper()
            if entity_id:
                ids.append(entity_id)
        return ids

    @classmethod
    def _parse_entities(cls, payload: object, max_results: int) -> list[Document]:
        """Parse wbgetentities results into documents."""
        if not isinstance(payload, dict):
            return []
        entities = payload.get("entities")
        if not isinstance(entities, dict):
            return []

        documents: list[Document] = []
        for entity_id, entity in entities.items():
            if not isinstance(entity, dict):
                continue
            document = cls._build_entity_document(entity_id, entity)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _parse_sparql_results(cls, payload: object, max_results: int) -> list[Document]:
        """Parse scholarly SPARQL JSON results into documents."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, dict):
            return []
        bindings = results.get("bindings")
        if not isinstance(bindings, list):
            return []

        documents: list[Document] = []
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            document = cls._build_sparql_document(binding)
            if document is not None:
                documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    @classmethod
    def _build_entity_document(cls, entity_id: str, entity: dict[str, object]) -> Document | None:
        """Build a document from one Wikidata entity record."""
        label = cls._localized_text(entity.get("labels")).strip()
        if not label:
            return None

        description = cls._localized_text(entity.get("descriptions")).strip()
        claims = cls._as_dict(entity.get("claims"))
        doi = cls._claim_string(claims, "P356")
        publication_year = cls._claim_year(claims, "P577")
        instance_of = cls._claim_entity_ids(claims, "P31")
        scholarly = bool(instance_of & _SCHOLARLY_INSTANCE_IDS) or bool(doi)

        source = f"https://www.wikidata.org/wiki/{entity_id}"
        text = description or cls._build_descriptor(label, doi, publication_year, scholarly)
        sitelinks = cls._as_dict(entity.get("sitelinks"))
        wikipedia_title = ""
        enwiki = sitelinks.get("enwiki")
        if isinstance(enwiki, dict):
            wikipedia_title = cls._as_str(enwiki.get("title")).strip()

        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(label.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "wikidata_scholarly",
                "wikidata_id": entity_id,
                "description": description,
                "doi": doi,
                "year": publication_year,
                "scholarly": "true" if scholarly else "false",
                "wikipedia_title": wikipedia_title,
            },
        )

    @classmethod
    def _build_sparql_document(cls, binding: dict[str, object]) -> Document | None:
        """Build a document from one SPARQL result binding."""
        item = cls._binding_value(binding.get("item"))
        label = cls._binding_value(binding.get("itemLabel"))
        if not label:
            return None
        description = cls._binding_value(binding.get("itemDescription"))
        doi = cls._binding_value(binding.get("doi"))
        entity_id = item.rsplit("/", maxsplit=1)[-1] if item else ""
        source = item or f"https://www.wikidata.org/wiki/{entity_id}"
        text = description or cls._build_descriptor(label, doi, "", bool(doi))
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(label.split()),
            text=text,
            source=source,
            metadata={
                "source_type": "wikidata_scholarly",
                "wikidata_id": entity_id,
                "description": description,
                "doi": doi,
                "year": "",
                "scholarly": "true" if doi else "false",
                "wikipedia_title": "",
            },
        )

    @staticmethod
    def _localized_text(value: object) -> str:
        """Return the English localized text from a Wikidata labels/descriptions map."""
        if not isinstance(value, dict):
            return ""
        english = value.get("en")
        if isinstance(english, dict):
            return WikidataScholarlyConnector._as_str(english.get("value"))
        for entry in value.values():
            if isinstance(entry, dict):
                text = WikidataScholarlyConnector._as_str(entry.get("value")).strip()
                if text:
                    return text
        return ""

    @classmethod
    def _claim_string(cls, claims: dict[str, object], property_id: str) -> str:
        """Return the first string claim value for a Wikidata property."""
        claim_list = claims.get(property_id)
        if not isinstance(claim_list, list):
            return ""
        for claim in claim_list:
            if not isinstance(claim, dict):
                continue
            mainsnak = cls._as_dict(claim.get("mainsnak"))
            datavalue = cls._as_dict(mainsnak.get("datavalue"))
            value = datavalue.get("value")
            if isinstance(value, str):
                return value.strip()
        return ""

    @classmethod
    def _claim_year(cls, claims: dict[str, object], property_id: str) -> str:
        """Return a four-digit year from a Wikidata time claim."""
        claim_list = claims.get(property_id)
        if not isinstance(claim_list, list):
            return ""
        for claim in claim_list:
            if not isinstance(claim, dict):
                continue
            mainsnak = cls._as_dict(claim.get("mainsnak"))
            datavalue = cls._as_dict(mainsnak.get("datavalue"))
            value = datavalue.get("value")
            if isinstance(value, dict):
                time_value = cls._as_str(value.get("time")).strip()
                if len(time_value) >= 5 and time_value.startswith("+"):
                    return time_value[1:5]
        return ""

    @classmethod
    def _claim_entity_ids(cls, claims: dict[str, object], property_id: str) -> set[str]:
        """Return entity ids referenced by a Wikidata entity-type claim."""
        claim_list = claims.get(property_id)
        if not isinstance(claim_list, list):
            return set()
        ids: set[str] = set()
        for claim in claim_list:
            if not isinstance(claim, dict):
                continue
            mainsnak = cls._as_dict(claim.get("mainsnak"))
            datavalue = cls._as_dict(mainsnak.get("datavalue"))
            value = datavalue.get("value")
            if isinstance(value, dict):
                entity_id = cls._as_str(value.get("id")).strip().upper()
                if entity_id:
                    ids.add(entity_id)
        return ids

    @staticmethod
    def _binding_value(value: object) -> str:
        """Return the string value from a SPARQL JSON binding."""
        if isinstance(value, dict):
            return WikidataScholarlyConnector._as_str(value.get("value")).strip()
        return ""

    @staticmethod
    def _build_descriptor(
        label: str,
        doi: str,
        year: str,
        scholarly: bool,
    ) -> str:
        """Compose searchable text when no description is available."""
        kind = "scholarly entity" if scholarly else "Wikidata entity"
        parts = [f"Wikidata {kind} {label}."]
        if doi:
            parts.append(f"DOI: {doi}.")
        if year:
            parts.append(f"Published: {year}.")
        return " ".join(parts)

    @staticmethod
    def _as_dict(value: object) -> dict[str, object]:
        """Return a dict value or an empty dict."""
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _as_str(value: object) -> str:
        """Coerce scalar Wikidata values to strings."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return ""
