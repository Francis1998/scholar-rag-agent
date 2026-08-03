"""Tests for ingestion connectors."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.ads import AdsConnector
from ingestion.arxiv import ArxivConnector
from ingestion.biorxiv import BioRxivConnector
from ingestion.biorxiv_collections import BioRxivCollectionsConnector
from ingestion.clinicaltrials import ClinicalTrialsConnector
from ingestion.core import CoreConnector
from ingestion.crossref import CrossrefConnector
from ingestion.crossref_events import CrossrefEventsConnector
from ingestion.crossref_funder import CrossrefFunderConnector
from ingestion.crossref_members import CrossrefMembersConnector
from ingestion.datacite import DataCiteConnector
from ingestion.datacite_related import DataciteRelatedConnector
from ingestion.dblp import DblpConnector
from ingestion.doaj import DoajConnector
from ingestion.dryad import DryadConnector
from ingestion.europepmc import EuropePmcConnector
from ingestion.figshare import FigshareConnector
from ingestion.hal import HalConnector
from ingestion.openaire import OpenAireConnector
from ingestion.openaire_projects import OpenaireProjectsConnector
from ingestion.openalex import OpenAlexConnector
from ingestion.openalex_authors import OpenAlexAuthorsConnector
from ingestion.openalex_concepts import OpenAlexConceptsConnector
from ingestion.openalex_institutions import OpenAlexInstitutionsConnector
from ingestion.openalex_sources import OpenAlexSourcesConnector
from ingestion.openalex_topics import OpenAlexTopicsConnector
from ingestion.opencitations import OpenCitationsConnector
from ingestion.orcid import OrcidConnector
from ingestion.osf import OsfConnector
from ingestion.pdf import PDFConnector
from ingestion.pmc import PmcConnector
from ingestion.pmc_oa import PmcOaPackageConnector
from ingestion.pubmed import PubMedConnector
from ingestion.retraction_watch import RetractionWatchConnector
from ingestion.semantic_scholar import SemanticScholarConnector
from ingestion.ssrn import SsrnConnector
from ingestion.unpaywall import UnpaywallConnector
from ingestion.wikidata_scholarly import WIKIDATA_SCHOLARLY_SPARQL_URL, WikidataScholarlyConnector
from ingestion.zenodo import ZenodoConnector

ARXIV_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678</id>
    <title>GraphRAG Paper</title>
    <summary>GraphRAG connects retrieval and agents.</summary>
  </entry>
</feed>
"""


@pytest.mark.asyncio
async def test_arxiv_connector_parses_atom_feed() -> None:
    """ArxivConnector normalizes Atom API responses into documents."""
    response = httpx.Response(200, text=ARXIV_FIXTURE, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.arxiv.httpx.AsyncClient", return_value=mock_client):
        documents = await ArxivConnector().fetch("1234.5678")

    assert len(documents) == 1
    assert documents[0].title == "GraphRAG Paper"
    assert "GraphRAG connects retrieval and agents." in documents[0].text
    assert documents[0].metadata["source_type"] == "arxiv"


@pytest.mark.asyncio
async def test_arxiv_connector_uses_id_list_for_versioned_id() -> None:
    """A versioned arXiv id (e.g. ``2301.00001v2``) must resolve via id_list.

    The previous ``replace('.', '').isdigit()`` id detection failed on the
    trailing ``vN`` version suffix, so versioned ids were misrouted to a
    keyword ``search_query`` instead of an exact ``id_list`` lookup.
    """
    response = httpx.Response(200, text=ARXIV_FIXTURE, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.arxiv.httpx.AsyncClient", return_value=mock_client):
        await ArxivConnector().fetch("2301.00001v2")

    params = mock_client.get.call_args.kwargs["params"]
    assert params.get("id_list") == "2301.00001v2"
    assert "search_query" not in params


@pytest.mark.asyncio
async def test_crossref_connector_searches_and_normalizes_works() -> None:
    """CrossrefConnector normalizes work items, stripping JATS abstract markup."""
    response = httpx.Response(
        200,
        json={
            "message": {
                "items": [
                    {
                        "title": ["Retrieval Augmented Generation Survey"],
                        "DOI": "10.1000/rag.survey",
                        "abstract": "<jats:p>RAG grounds answers in evidence.</jats:p>",
                        "published": {"date-parts": [[2024, 3]]},
                    }
                ]
            }
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.crossref.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefConnector(mailto="dev@example.org").search("rag", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Retrieval Augmented Generation Survey"
    assert document.text == "RAG grounds answers in evidence."
    assert document.source == "https://doi.org/10.1000/rag.survey"
    assert document.metadata["source_type"] == "crossref"
    assert document.metadata["doi"] == "10.1000/rag.survey"
    assert document.metadata["year"] == "2024"


@pytest.mark.asyncio
async def test_crossref_connector_resolves_year_from_issued_when_published_absent() -> None:
    """The year must be read from ``issued`` when ``published`` is absent.

    Crossref does not always populate the unified ``published`` field; ``issued``
    is its canonical, most widely populated publication date. Reading only
    ``published`` dropped the year for the many records that carry it solely
    under ``issued``, leaving ``metadata['year']`` empty.
    """
    response = httpx.Response(
        200,
        json={
            "message": {
                "items": [
                    {
                        "title": ["Sparse-Dense Hybrid Retrieval"],
                        "DOI": "10.1000/hybrid",
                        "abstract": "<jats:p>Hybrid retrieval.</jats:p>",
                        "issued": {"date-parts": [[2023, 11]]},
                    }
                ]
            }
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.crossref.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefConnector().search("hybrid", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["year"] == "2023"


def _hal_client(payload: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a fixed HAL JSON payload.

    Args:
        payload: JSON body the mocked GET should return.

    Returns:
        Configured async mock usable as an async context manager.
    """
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_hal_connector_searches_and_normalizes_docs() -> None:
    """HalConnector normalizes Solr multi-valued fields into documents."""
    payload: dict[str, object] = {
        "response": {
            "docs": [
                {
                    "title_s": ["Federated Retrieval over Open Archives"],
                    "authFullName_s": ["Ada Lovelace", "Alan Turing"],
                    "abstract_s": ["HAL indexes multidisciplinary open science."],
                    "uri_s": "https://hal.science/hal-04123456",
                    "doiId_s": "10.1000/hal.rag",
                    "publicationDateY_i": 2024,
                }
            ]
        }
    }

    with patch("ingestion.hal.httpx.AsyncClient", return_value=_hal_client(payload)):
        documents = await HalConnector().search("retrieval", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Federated Retrieval over Open Archives"
    assert document.text == "HAL indexes multidisciplinary open science."
    assert document.source == "https://hal.science/hal-04123456"
    assert document.metadata["source_type"] == "hal"
    assert document.metadata["doi"] == "10.1000/hal.rag"
    assert document.metadata["year"] == "2024"
    assert document.metadata["authors"] == "Ada Lovelace, Alan Turing"


@pytest.mark.asyncio
async def test_hal_connector_builds_descriptor_and_doi_source_without_abstract() -> None:
    """A record with no abstract or URI uses a descriptor and DOI-anchored source."""
    payload: dict[str, object] = {
        "response": {
            "docs": [
                {
                    "title_s": ["A Bibliographic-Only Deposit"],
                    "authFullName_s": "Grace Hopper",
                    "doiId_s": "10.1000/hal.solo",
                    "producedDateY_i": 2019,
                }
            ]
        }
    }

    with patch("ingestion.hal.httpx.AsyncClient", return_value=_hal_client(payload)):
        documents = await HalConnector().search("compilers", max_results=1)

    assert len(documents) == 1
    document = documents[0]
    assert document.text == "By Grace Hopper (2019)"
    assert document.source == "https://doi.org/10.1000/hal.solo"
    assert document.metadata["year"] == "2019"
    assert document.metadata["authors"] == "Grace Hopper"


@pytest.mark.asyncio
async def test_hal_connector_skips_docs_without_title() -> None:
    """A doc carrying no usable title is skipped rather than surfaced empty."""
    payload: dict[str, object] = {
        "response": {"docs": [{"abstract_s": ["No title here."], "uri_s": "https://hal.science/x"}]}
    }

    with patch("ingestion.hal.httpx.AsyncClient", return_value=_hal_client(payload)):
        documents = await HalConnector().search("anything", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_hal_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.hal.httpx.AsyncClient", return_value=mock_client):
        documents = await HalConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_crossref_connector_decodes_entities_in_jats_abstract() -> None:
    """XML/HTML entities in a JATS abstract are decoded to their characters.

    Crossref abstracts are JATS XML in which literal ``<``, ``>``, ``&`` and
    non-ASCII characters are entity-encoded (for example ``&lt;``, ``&amp;``,
    ``&#233;``). Stripping only the tags left those entities as raw text in the
    stored abstract; they must be decoded to their characters so the prose is
    readable and searchable.
    """
    response = httpx.Response(
        200,
        json={
            "message": {
                "items": [
                    {
                        "title": ["Entity Handling"],
                        "DOI": "10.1000/entities",
                        "abstract": (
                            "<jats:p>Results show x &lt; y &amp; z &gt; 0 in a caf&#233;.</jats:p>"
                        ),
                        "published": {"date-parts": [[2025]]},
                    }
                ]
            }
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.crossref.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefConnector().search("entities", max_results=1)

    assert len(documents) == 1
    assert documents[0].text == "Results show x < y & z > 0 in a café."


@pytest.mark.asyncio
async def test_crossref_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.crossref.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_crossref_connector_handles_missing_abstract_and_doi() -> None:
    """A work without an abstract or DOI still yields a titled document."""
    response = httpx.Response(
        200,
        json={"message": {"items": [{"title": ["Preprint Without Metadata"]}]}},
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.crossref.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefConnector().search("preprint", max_results=5)

    assert len(documents) == 1
    assert documents[0].title == "Preprint Without Metadata"
    assert documents[0].text == ""
    assert documents[0].source == "Preprint Without Metadata"
    assert documents[0].metadata["doi"] == ""


@pytest.mark.asyncio
async def test_europepmc_connector_searches_and_normalizes_results() -> None:
    """EuropePmcConnector normalizes result items and builds the article URL."""
    response = httpx.Response(
        200,
        json={
            "resultList": {
                "result": [
                    {
                        "id": "40012345",
                        "source": "MED",
                        "title": "Federated Retrieval for Biomedicine",
                        "abstractText": "Europe PMC federates many sources.",
                        "doi": "10.1000/epmc.rag",
                        "pubYear": "2025",
                        "pmid": "40012345",
                    }
                ]
            }
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.europepmc.httpx.AsyncClient", return_value=mock_client):
        documents = await EuropePmcConnector(email="dev@example.org").search("rag", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Federated Retrieval for Biomedicine"
    assert document.text == "Europe PMC federates many sources."
    assert document.source == "https://europepmc.org/article/MED/40012345"
    assert document.metadata["source_type"] == "europepmc"
    assert document.metadata["doi"] == "10.1000/epmc.rag"
    assert document.metadata["year"] == "2025"
    assert document.metadata["pmid"] == "40012345"


@pytest.mark.asyncio
async def test_europepmc_connector_coerces_numeric_year_and_uses_doi_fallback() -> None:
    """A numeric ``pubYear`` is coerced and the DOI anchors a source-less result."""
    response = httpx.Response(
        200,
        json={
            "resultList": {
                "result": [
                    {
                        "title": "Preprint Without Source Id",
                        "abstractText": "Body.",
                        "doi": "10.1000/epmc.preprint",
                        "pubYear": 2024,
                    }
                ]
            }
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.europepmc.httpx.AsyncClient", return_value=mock_client):
        documents = await EuropePmcConnector().search("preprint", max_results=5)

    assert len(documents) == 1
    assert documents[0].source == "https://doi.org/10.1000/epmc.preprint"
    assert documents[0].metadata["year"] == "2024"


@pytest.mark.asyncio
async def test_europepmc_connector_falls_back_to_first_publication_date_year() -> None:
    """A record without ``pubYear`` must derive its year from ``firstPublicationDate``.

    Some Europe PMC records (notably preprints and ahead-of-print articles) omit
    ``pubYear`` while still carrying a full ``firstPublicationDate``. Reading only
    ``pubYear`` previously dropped the year entirely; the 4-digit prefix of
    ``firstPublicationDate`` must be used as a fallback.
    """
    response = httpx.Response(
        200,
        json={
            "resultList": {
                "result": [
                    {
                        "title": "Ahead of Print Without pubYear",
                        "abstractText": "Body.",
                        "doi": "10.1000/epmc.aheadofprint",
                        "firstPublicationDate": "2021-07-01",
                    }
                ]
            }
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.europepmc.httpx.AsyncClient", return_value=mock_client):
        documents = await EuropePmcConnector().search("ahead", max_results=5)

    assert len(documents) == 1
    assert documents[0].metadata["year"] == "2021"


@pytest.mark.asyncio
async def test_europepmc_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.europepmc.httpx.AsyncClient", return_value=mock_client):
        documents = await EuropePmcConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_doaj_connector_searches_and_normalizes_articles() -> None:
    """DoajConnector normalizes bibjson articles and prefers the full-text link."""
    response = httpx.Response(
        200,
        json={
            "total": 1,
            "results": [
                {
                    "id": "abc123",
                    "bibjson": {
                        "title": "Open Access Retrieval",
                        "abstract": "DOAJ indexes open access articles.",
                        "year": "2025",
                        "identifier": [
                            {"type": "doi", "id": "10.1000/doaj.rag"},
                            {"type": "eissn", "id": "1234-5678"},
                        ],
                        "link": [
                            {"type": "fulltext", "url": "https://journal.example.org/article/1"},
                        ],
                    },
                }
            ],
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.doaj.httpx.AsyncClient", return_value=mock_client):
        documents = await DoajConnector().search("rag", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Open Access Retrieval"
    assert document.text == "DOAJ indexes open access articles."
    assert document.source == "https://journal.example.org/article/1"
    assert document.metadata["source_type"] == "doaj"
    assert document.metadata["doi"] == "10.1000/doaj.rag"
    assert document.metadata["year"] == "2025"


@pytest.mark.asyncio
async def test_doaj_connector_coerces_numeric_year_and_falls_back_to_doi() -> None:
    """A numeric ``year`` is coerced and the DOI anchors a link-less article."""
    response = httpx.Response(
        200,
        json={
            "results": [
                {
                    "id": "no-link",
                    "bibjson": {
                        "title": "Article Without Full-Text Link",
                        "abstract": "Body.",
                        "year": 2024,
                        "identifier": [{"type": "DOI", "id": "10.1000/doaj.nolink"}],
                    },
                }
            ]
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.doaj.httpx.AsyncClient", return_value=mock_client):
        documents = await DoajConnector().search("preprint", max_results=5)

    assert len(documents) == 1
    # Identifier ``type`` is matched case-insensitively (``DOI`` -> ``doi``).
    assert documents[0].source == "https://doi.org/10.1000/doaj.nolink"
    assert documents[0].metadata["year"] == "2024"


@pytest.mark.asyncio
async def test_doaj_connector_skips_items_without_bibjson() -> None:
    """A result item lacking a ``bibjson`` object is skipped, not crashed on."""
    response = httpx.Response(
        200,
        json={"results": [{"id": "malformed"}]},
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.doaj.httpx.AsyncClient", return_value=mock_client):
        documents = await DoajConnector().search("anything", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_doaj_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.doaj.httpx.AsyncClient", return_value=mock_client):
        documents = await DoajConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_dblp_connector_searches_and_normalizes_publications() -> None:
    """DblpConnector normalizes info hits and prefers the electronic edition."""
    response = httpx.Response(
        200,
        json={
            "result": {
                "hits": {
                    "@total": "1",
                    "hit": [
                        {
                            "info": {
                                "title": "Retrieval-Augmented Generation",
                                "authors": {
                                    "author": [
                                        {"@pid": "1", "text": "Ada Lovelace"},
                                        {"@pid": "2", "text": "Alan Turing"},
                                    ]
                                },
                                "venue": "NeurIPS",
                                "year": "2020",
                                "doi": "10.5555/rag",
                                "ee": "https://example.org/rag.pdf",
                                "url": "https://dblp.org/rec/conf/nips/rag",
                            }
                        }
                    ],
                }
            }
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.dblp.httpx.AsyncClient", return_value=mock_client):
        documents = await DblpConnector().search("rag", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Retrieval-Augmented Generation"
    assert document.text == "By Ada Lovelace, Alan Turing In NeurIPS (2020)"
    assert document.source == "https://example.org/rag.pdf"
    assert document.metadata["source_type"] == "dblp"
    assert document.metadata["doi"] == "10.5555/rag"
    assert document.metadata["venue"] == "NeurIPS"
    assert document.metadata["authors"] == "Ada Lovelace, Alan Turing"


@pytest.mark.asyncio
async def test_dblp_connector_handles_single_hit_and_author_objects() -> None:
    """A single match collapses ``hit``/``author`` to objects and falls back to DOI.

    DBLP returns ``hits.hit`` (and ``authors.author``) as a lone object rather
    than a list when exactly one result/author is present, and omits ``ee`` for
    some records; the connector must normalize the object shapes and anchor the
    source on the DOI when no electronic edition is advertised.
    """
    response = httpx.Response(
        200,
        json={
            "result": {
                "hits": {
                    "@total": "1",
                    "hit": {
                        "info": {
                            "title": "A Solo Systems Paper",
                            "authors": {"author": {"@pid": "3", "text": "Grace Hopper"}},
                            "year": 2019,
                            "doi": "10.1000/solo",
                        }
                    },
                }
            }
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.dblp.httpx.AsyncClient", return_value=mock_client):
        documents = await DblpConnector().search("systems", max_results=5)

    assert len(documents) == 1
    document = documents[0]
    assert document.source == "https://doi.org/10.1000/solo"
    assert document.text == "By Grace Hopper (2019)"
    assert document.metadata["year"] == "2019"
    assert document.metadata["authors"] == "Grace Hopper"


@pytest.mark.asyncio
async def test_dblp_connector_handles_list_valued_venue_and_ee() -> None:
    """List-valued ``venue``/``ee`` fields must be read, not dropped.

    DBLP collapses a single value to a scalar but returns a *list* when a record
    carries several values (multiple electronic editions or venues). The list
    form previously coerced to an empty string, so the venue was lost from the
    metadata and descriptor and the source URL fell back off the electronic
    edition onto a weaker anchor. The first element must be used.
    """
    response = httpx.Response(
        200,
        json={
            "result": {
                "hits": {
                    "hit": [
                        {
                            "info": {
                                "title": "A Multi-Edition Paper",
                                "authors": {"author": {"text": "Ada Lovelace"}},
                                "venue": ["PVLDB", "VLDB J."],
                                "ee": ["https://example.org/pdf", "https://example.org/alt"],
                                "year": "2024",
                                "doi": "10.1000/multi",
                            }
                        }
                    ]
                }
            }
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.dblp.httpx.AsyncClient", return_value=mock_client):
        documents = await DblpConnector().search("systems", max_results=5)

    assert len(documents) == 1
    document = documents[0]
    assert document.metadata["venue"] == "PVLDB"
    assert document.source == "https://example.org/pdf"
    assert "In PVLDB" in document.text


@pytest.mark.asyncio
async def test_dblp_connector_skips_hits_without_title() -> None:
    """A hit whose ``info`` carries no title is skipped rather than crashed on."""
    response = httpx.Response(
        200,
        json={"result": {"hits": {"hit": [{"info": {"year": "2021"}}]}}},
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.dblp.httpx.AsyncClient", return_value=mock_client):
        documents = await DblpConnector().search("anything", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_dblp_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.dblp.httpx.AsyncClient", return_value=mock_client):
        documents = await DblpConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_scholar_connector_parses_paper_payload() -> None:
    """SemanticScholarConnector normalizes paper JSON into a document."""
    response = httpx.Response(
        200,
        json={
            "title": "Hybrid Retrieval",
            "abstract": "Dense and sparse retrieval improve recall.",
            "year": 2024,
            "url": "https://example.org/paper/1",
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.semantic_scholar.httpx.AsyncClient", return_value=mock_client):
        document = await SemanticScholarConnector(api_key="test-key").fetch_paper("abc123")

    assert document.title == "Hybrid Retrieval"
    assert document.text == "Dense and sparse retrieval improve recall."
    assert document.metadata["source_type"] == "semantic_scholar"
    assert document.metadata["year"] == "2024"


@pytest.mark.asyncio
async def test_semantic_scholar_missing_year_uses_publication_date_not_none_string() -> None:
    """A null ``year`` must not become the literal ``\"None\"``; prefer ``publicationDate``.

    Semantic Scholar often returns ``year: null`` while still carrying
    ``publicationDate`` (e.g. ``2023-05-01``). A naive ``str(year)`` coercion
    leaked ``\"None\"`` into ``metadata['year']``, and ignoring
    ``publicationDate`` dropped a usable year. The year must be the four-digit
    prefix of ``publicationDate`` (never the string ``\"None\"``).
    """
    response = httpx.Response(
        200,
        json={
            "title": "Undated Draft",
            "abstract": "Abstract text.",
            "year": None,
            "publicationDate": "2023-05-01",
            "url": "https://example.org/paper/undated",
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.semantic_scholar.httpx.AsyncClient", return_value=mock_client):
        document = await SemanticScholarConnector().fetch_paper("undated")

    assert document.metadata["year"] == "2023"
    assert document.metadata["year"] != "None"


@pytest.mark.asyncio
async def test_semantic_scholar_connector_searches_and_normalizes_papers() -> None:
    """SemanticScholarConnector.search normalizes paper/search hits into documents."""
    response = httpx.Response(
        200,
        json={
            "data": [
                {
                    "paperId": "abc",
                    "title": "Retrieval Paper",
                    "abstract": "About retrieval.",
                    "year": 2022,
                    "url": "https://example.org/paper/abc",
                }
            ]
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.semantic_scholar.httpx.AsyncClient", return_value=mock_client):
        documents = await SemanticScholarConnector().search("retrieval", max_results=3)

    assert len(documents) == 1
    assert documents[0].title == "Retrieval Paper"
    assert documents[0].metadata["year"] == "2022"
    assert documents[0].metadata["source_type"] == "semantic_scholar"


@pytest.mark.asyncio
async def test_semantic_scholar_search_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.semantic_scholar.httpx.AsyncClient", return_value=mock_client):
        assert await SemanticScholarConnector().search("   ", max_results=5) == []
        assert await SemanticScholarConnector().search("q", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_scholar_recommendations_normalize_related_papers() -> None:
    """SemanticScholarConnector.recommendations normalizes recommendedPapers."""
    response = httpx.Response(
        200,
        json={
            "recommendedPapers": [
                {
                    "paperId": "recommended-1",
                    "title": "Related Graph Retrieval",
                    "abstract": "Recommendations surface adjacent retrieval work.",
                    "year": 2025,
                    "url": "https://example.org/paper/recommended-1",
                    "externalIds": {"DOI": "10.1000/recommended"},
                    "authors": [{"name": "Ada Lovelace"}, {"name": "Alan Turing"}],
                }
            ]
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.semantic_scholar.httpx.AsyncClient", return_value=mock_client):
        documents = await SemanticScholarConnector(api_key="test-key").recommendations(
            "DOI:10.1000/root.paper",
            max_results=50,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Related Graph Retrieval"
    assert document.text == "Recommendations surface adjacent retrieval work."
    assert document.source == "https://example.org/paper/recommended-1"
    assert document.metadata["source_type"] == "semantic_scholar_recommendations"
    assert document.metadata["seed_paper"] == "DOI:10.1000/root.paper"
    assert document.metadata["semantic_scholar_id"] == "recommended-1"
    assert document.metadata["doi"] == "10.1000/recommended"
    assert document.metadata["authors"] == "Ada Lovelace, Alan Turing"
    assert document.metadata["year"] == "2025"

    requested_url = mock_client.get.call_args.args[0]
    assert requested_url.endswith("/papers/forpaper/DOI%3A10.1000%2Froot.paper")
    assert mock_client.get.call_args.kwargs["params"]["limit"] == 20
    assert "externalIds" in mock_client.get.call_args.kwargs["params"]["fields"]
    assert mock_client.get.call_args.kwargs["headers"] == {"x-api-key": "test-key"}


@pytest.mark.asyncio
async def test_semantic_scholar_recommendations_reject_blank_and_non_positive() -> None:
    """Blank seeds and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.semantic_scholar.httpx.AsyncClient", return_value=mock_client):
        assert await SemanticScholarConnector().recommendations("   ", max_results=5) == []
        assert await SemanticScholarConnector().recommendations("abc", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_openalex_connector_reconstructs_inverted_abstract() -> None:
    """OpenAlexConnector rebuilds the abstract from its inverted index."""
    response = httpx.Response(
        200,
        json={
            "id": "https://openalex.org/W2741809807",
            "title": "Retrieval Augmented Generation",
            "publication_year": 2023,
            "abstract_inverted_index": {
                "Dense": [0],
                "and": [1],
                "sparse": [2],
                "retrieval": [3],
            },
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.openalex.httpx.AsyncClient", return_value=mock_client):
        document = await OpenAlexConnector(mailto="dev@example.org").fetch_work("W2741809807")

    assert document.title == "Retrieval Augmented Generation"
    assert document.text == "Dense and sparse retrieval"
    assert document.source == "https://openalex.org/W2741809807"
    assert document.metadata["source_type"] == "openalex"
    assert document.metadata["year"] == "2023"


def test_openalex_reconstruct_abstract_orders_repeated_words() -> None:
    """The inverted-index reconstruction restores original word order.

    A word may appear at several positions; each occurrence must be placed at its
    own index so the reconstructed text preserves the source ordering rather than
    collapsing duplicates.
    """
    inverted_index = {
        "graph": [0, 3],
        "based": [1],
        "retrieval": [2, 4],
    }

    reconstructed = OpenAlexConnector._reconstruct_abstract(inverted_index)

    assert reconstructed == "graph based retrieval graph retrieval"


def test_openalex_reconstruct_abstract_handles_missing_index() -> None:
    """A missing or non-dict inverted index yields an empty abstract."""
    assert OpenAlexConnector._reconstruct_abstract(None) == ""
    assert OpenAlexConnector._reconstruct_abstract({}) == ""


PUBMED_EFETCH_FIXTURE = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>40012345</PMID>
      <Article>
        <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Retrieval Augmented Generation for Clinical QA</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">RAG grounds answers in evidence.</AbstractText>
          <AbstractText Label="RESULTS">It improves factual accuracy.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


@pytest.mark.asyncio
async def test_pubmed_connector_searches_and_normalizes_articles() -> None:
    """PubMedConnector resolves a query to PMIDs then fetches normalized docs."""
    esearch_response = httpx.Response(
        200,
        json={"esearchresult": {"idlist": ["40012345"]}},
        request=httpx.Request("GET", "http://test"),
    )
    efetch_response = httpx.Response(
        200, text=PUBMED_EFETCH_FIXTURE, request=httpx.Request("GET", "http://test")
    )
    mock_client = AsyncMock()
    mock_client.get.side_effect = [esearch_response, efetch_response]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pubmed.httpx.AsyncClient", return_value=mock_client):
        documents = await PubMedConnector(api_key="test-key").search("clinical RAG", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Retrieval Augmented Generation for Clinical QA"
    # Structured abstract sections are joined, not truncated to the first.
    assert document.text == "RAG grounds answers in evidence. It improves factual accuracy."
    assert document.source == "https://pubmed.ncbi.nlm.nih.gov/40012345/"
    assert document.metadata["source_type"] == "pubmed"
    assert document.metadata["pmid"] == "40012345"
    assert document.metadata["year"] == "2024"


PUBMED_INLINE_MARKUP_FIXTURE = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>40067890</PMID>
      <Article>
        <Journal><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Inline Markup Abstract</ArticleTitle>
        <Abstract>
          <AbstractText>The <i>BRCA1</i> gene is <b>essential</b> for repair.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


@pytest.mark.asyncio
async def test_pubmed_connector_falls_back_to_medline_date_year() -> None:
    """A record without ``Year`` must derive its year from ``MedlineDate``.

    Many PubMed records (seasonal issues, date ranges) omit ``PubDate/Year`` and
    carry only a ``MedlineDate`` such as ``2024 Spring``. Reading only ``Year``
    previously dropped the year entirely; the leading four characters of
    ``MedlineDate`` must be used as a fallback.
    """
    medline_only_fixture = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>40099999</PMID>
      <Article>
        <Journal>
          <JournalIssue>
            <PubDate><MedlineDate>2024 Spring</MedlineDate></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Seasonal PubDate Only</ArticleTitle>
        <Abstract><AbstractText>Body.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""
    esearch_response = httpx.Response(
        200,
        json={"esearchresult": {"idlist": ["40099999"]}},
        request=httpx.Request("GET", "http://test"),
    )
    efetch_response = httpx.Response(
        200, text=medline_only_fixture, request=httpx.Request("GET", "http://test")
    )
    mock_client = AsyncMock()
    mock_client.get.side_effect = [esearch_response, efetch_response]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pubmed.httpx.AsyncClient", return_value=mock_client):
        documents = await PubMedConnector().search("seasonal", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["year"] == "2024"


@pytest.mark.asyncio
async def test_pubmed_connector_preserves_abstract_with_inline_markup() -> None:
    """Inline formatting tags in an AbstractText must not truncate the abstract.

    PubMed embeds inline elements (``<i>`` for gene names, ``<sup>`` for
    exponents, ``<b>`` for emphasis) inside an ``AbstractText``. Reading only
    ``node.text`` captured just the run before the first inline child, silently
    dropping the rest of the abstract; the full text of every segment must be
    reconstructed instead.
    """
    esearch_response = httpx.Response(
        200,
        json={"esearchresult": {"idlist": ["40067890"]}},
        request=httpx.Request("GET", "http://test"),
    )
    efetch_response = httpx.Response(
        200, text=PUBMED_INLINE_MARKUP_FIXTURE, request=httpx.Request("GET", "http://test")
    )
    mock_client = AsyncMock()
    mock_client.get.side_effect = [esearch_response, efetch_response]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pubmed.httpx.AsyncClient", return_value=mock_client):
        documents = await PubMedConnector().search("brca1", max_results=1)

    assert len(documents) == 1
    assert documents[0].text == "The BRCA1 gene is essential for repair."


@pytest.mark.asyncio
async def test_pubmed_connector_returns_empty_on_no_hits() -> None:
    """An empty PMID list short-circuits before any efetch call."""
    esearch_response = httpx.Response(
        200,
        json={"esearchresult": {"idlist": []}},
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = esearch_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pubmed.httpx.AsyncClient", return_value=mock_client):
        documents = await PubMedConnector().search("no such topic", max_results=5)

    assert documents == []
    assert mock_client.get.await_count == 1


@pytest.mark.asyncio
async def test_pubmed_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pubmed.httpx.AsyncClient", return_value=mock_client):
        documents = await PubMedConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.get.assert_not_called()


PMC_EFETCH_FIXTURE = """<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <front>
      <article-meta>
        <article-id pub-id-type="pmc">PMC7654321</article-id>
        <article-id pub-id-type="pmid">41000001</article-id>
        <article-id pub-id-type="doi">10.1000/pmc.fulltext</article-id>
        <title-group>
          <article-title>Full Text Retrieval for Biomedical RAG</article-title>
        </title-group>
        <contrib-group>
          <contrib contrib-type="author">
            <name><surname>Lovelace</surname><given-names>Ada</given-names></name>
          </contrib>
          <contrib contrib-type="author">
            <name><surname>Turing</surname><given-names>Alan</given-names></name>
          </contrib>
        </contrib-group>
        <pub-date><year>2025</year></pub-date>
        <abstract>
          <p>PMC provides open full-text article records.</p>
        </abstract>
      </article-meta>
    </front>
    <body>
      <sec>
        <title>Results</title>
        <p>Full text contains methods, findings, and citation context.</p>
      </sec>
    </body>
  </article>
</pmc-articleset>
"""


@pytest.mark.asyncio
async def test_pmc_connector_searches_and_normalizes_fulltext_articles() -> None:
    """PmcConnector resolves a query to PMC IDs then fetches full-text XML."""
    esearch_response = httpx.Response(
        200,
        json={"esearchresult": {"idlist": ["7654321"]}},
        request=httpx.Request("GET", "http://test"),
    )
    efetch_response = httpx.Response(
        200, text=PMC_EFETCH_FIXTURE, request=httpx.Request("GET", "http://test")
    )
    mock_client = AsyncMock()
    mock_client.get.side_effect = [esearch_response, efetch_response]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pmc.httpx.AsyncClient", return_value=mock_client):
        documents = await PmcConnector(api_key="test-key", email="dev@example.org").search(
            "biomedical RAG",
            max_results=3,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Full Text Retrieval for Biomedical RAG"
    assert document.source == "https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/"
    assert document.metadata["source_type"] == "pmc"
    assert document.metadata["pmcid"] == "PMC7654321"
    assert document.metadata["pmid"] == "41000001"
    assert document.metadata["doi"] == "10.1000/pmc.fulltext"
    assert document.metadata["year"] == "2025"
    assert document.metadata["authors"] == "Ada Lovelace, Alan Turing"
    assert "PMC provides open full-text article records." in document.text
    assert "Full-text excerpt: Results Full text contains methods" in document.text

    esearch_params = mock_client.get.await_args_list[0].kwargs["params"]
    assert esearch_params["db"] == "pmc"
    assert esearch_params["retmax"] == 3
    assert esearch_params["api_key"] == "test-key"
    assert esearch_params["email"] == "dev@example.org"
    efetch_params = mock_client.get.await_args_list[1].kwargs["params"]
    assert efetch_params["db"] == "pmc"
    assert efetch_params["id"] == "7654321"


@pytest.mark.asyncio
async def test_pmc_connector_uses_body_excerpt_without_abstract() -> None:
    """Full-text body content is used when a PMC article lacks an abstract."""
    xml_text = """<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <front>
      <article-meta>
        <article-id pub-id-type="pmc">7654322</article-id>
        <title-group><article-title>Body Only PMC Record</article-title></title-group>
        <pub-date><year>2024</year></pub-date>
      </article-meta>
    </front>
    <body><p>The article body is still useful for retrieval.</p></body>
  </article>
</pmc-articleset>
"""
    esearch_response = httpx.Response(
        200,
        json={"esearchresult": {"idlist": ["7654322"]}},
        request=httpx.Request("GET", "http://test"),
    )
    efetch_response = httpx.Response(
        200, text=xml_text, request=httpx.Request("GET", "http://test")
    )
    mock_client = AsyncMock()
    mock_client.get.side_effect = [esearch_response, efetch_response]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pmc.httpx.AsyncClient", return_value=mock_client):
        documents = await PmcConnector().search("body only", max_results=1)

    assert len(documents) == 1
    assert documents[0].source == "https://pmc.ncbi.nlm.nih.gov/articles/PMC7654322/"
    assert documents[0].text == "The article body is still useful for retrieval."
    assert documents[0].metadata["pmcid"] == "PMC7654322"


@pytest.mark.asyncio
async def test_pmc_connector_returns_empty_on_no_hits() -> None:
    """An empty PMC ID list short-circuits before any efetch call."""
    esearch_response = httpx.Response(
        200,
        json={"esearchresult": {"idlist": []}},
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = esearch_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pmc.httpx.AsyncClient", return_value=mock_client):
        documents = await PmcConnector().search("no such topic", max_results=5)

    assert documents == []
    assert mock_client.get.await_count == 1


@pytest.mark.asyncio
async def test_pmc_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pmc.httpx.AsyncClient", return_value=mock_client):
        assert await PmcConnector().search("   ", max_results=5) == []
        assert await PmcConnector().search("pmc", max_results=0) == []

    mock_client.get.assert_not_called()


def test_pdf_connector_extracts_text(tmp_path: Path) -> None:
    """PDFConnector extracts text from a local PDF file."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "GraphRAG supports scientific retrieval."
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("pypdf.PdfReader", return_value=mock_reader):
        document = PDFConnector().load(pdf_path)

    assert document.title == "sample"
    assert "GraphRAG supports scientific retrieval." in document.text
    assert document.metadata["source_type"] == "pdf"


def _openaire_client(payload: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a fixed OpenAIRE JSON payload.

    Args:
        payload: The decoded JSON body the mocked client should return.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_openaire_connector_searches_and_normalizes_products() -> None:
    """OpenAireConnector normalizes research products and prefers instance URLs."""
    payload: dict[str, object] = {
        "header": {"numFound": 1},
        "results": [
            {
                "mainTitle": "Open Science Retrieval",
                "authors": [
                    {"fullName": "Ada Lovelace", "rank": 1},
                    {"fullName": "Alan Turing", "rank": 2},
                ],
                "descriptions": ["An abstract about open-science retrieval graphs."],
                "publicationDate": "2025-03-18",
                "pids": [
                    {"scheme": "pmc", "value": "PMC1"},
                    {"scheme": "doi", "value": "10.1000/openaire.rag"},
                ],
                "instances": [{"urls": ["https://example.org/oa/1"]}],
            }
        ],
    }
    with patch("ingestion.openaire.httpx.AsyncClient", return_value=_openaire_client(payload)):
        documents = await OpenAireConnector().search("retrieval", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Open Science Retrieval"
    assert document.text == "An abstract about open-science retrieval graphs."
    assert document.source == "https://example.org/oa/1"
    assert document.metadata["source_type"] == "openaire"
    assert document.metadata["doi"] == "10.1000/openaire.rag"
    assert document.metadata["year"] == "2025"
    assert document.metadata["authors"] == "Ada Lovelace, Alan Turing"


@pytest.mark.asyncio
async def test_openaire_connector_builds_descriptor_and_doi_source_without_abstract() -> None:
    """A record without a description falls back to a descriptor and DOI source."""
    payload: dict[str, object] = {
        "results": [
            {
                "mainTitle": "Bibliographic Only",
                "authors": [{"fullName": "Grace Hopper"}],
                "publicationDate": "2020",
                "pids": [{"scheme": "doi", "value": "10.1000/openaire.solo"}],
            }
        ]
    }
    with patch("ingestion.openaire.httpx.AsyncClient", return_value=_openaire_client(payload)):
        documents = await OpenAireConnector().search("history", max_results=1)

    assert len(documents) == 1
    document = documents[0]
    assert document.text == "By Grace Hopper (2020)"
    assert document.source == "https://doi.org/10.1000/openaire.solo"
    assert document.metadata["year"] == "2020"


@pytest.mark.asyncio
async def test_openaire_connector_skips_products_without_title() -> None:
    """A research product carrying no ``mainTitle`` is skipped, not crashed on."""
    payload: dict[str, object] = {
        "results": [{"descriptions": ["No title here."], "publicationDate": "2023"}]
    }
    with patch("ingestion.openaire.httpx.AsyncClient", return_value=_openaire_client(payload)):
        documents = await OpenAireConnector().search("anything", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_openaire_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.openaire.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAireConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.get.assert_not_called()


def _zenodo_client(payload: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a fixed Zenodo JSON payload.

    Args:
        payload: The decoded JSON body the mocked client should return.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_zenodo_connector_searches_and_normalizes_records() -> None:
    """ZenodoConnector normalizes records, strips HTML, and prefers the html link."""
    payload: dict[str, object] = {
        "hits": {
            "total": 1,
            "hits": [
                {
                    "doi": "10.5281/zenodo.123",
                    "links": {
                        "self": "https://zenodo.org/api/records/123",
                        "html": "https://zenodo.org/records/123",
                    },
                    "metadata": {
                        "title": "Open Retrieval Toolkit",
                        "creators": [
                            {"name": "Ada Lovelace"},
                            {"name": "Alan Turing"},
                        ],
                        "description": "<p>A <b>toolkit</b> for retrieval &amp; agents.</p>",
                        "publication_date": "2025-02-10",
                        "doi": "10.5281/zenodo.123",
                    },
                }
            ],
        }
    }
    with patch("ingestion.zenodo.httpx.AsyncClient", return_value=_zenodo_client(payload)):
        documents = await ZenodoConnector().search("retrieval", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Open Retrieval Toolkit"
    assert document.text == "A toolkit for retrieval & agents."
    assert document.source == "https://zenodo.org/records/123"
    assert document.metadata["source_type"] == "zenodo"
    assert document.metadata["doi"] == "10.5281/zenodo.123"
    assert document.metadata["year"] == "2025"
    assert document.metadata["authors"] == "Ada Lovelace, Alan Turing"


@pytest.mark.asyncio
async def test_zenodo_connector_builds_descriptor_and_doi_source_without_description() -> None:
    """A record without a description falls back to a descriptor and DOI source."""
    payload: dict[str, object] = {
        "hits": {
            "hits": [
                {
                    "doi": "10.5281/zenodo.999",
                    "metadata": {
                        "title": "Dataset Only",
                        "creators": [{"name": "Grace Hopper"}],
                        "publication_date": "2020-01-01",
                    },
                }
            ]
        }
    }
    with patch("ingestion.zenodo.httpx.AsyncClient", return_value=_zenodo_client(payload)):
        documents = await ZenodoConnector().search("dataset", max_results=1)

    assert len(documents) == 1
    document = documents[0]
    assert document.text == "By Grace Hopper (2020)"
    assert document.source == "https://doi.org/10.5281/zenodo.999"
    assert document.metadata["year"] == "2020"


@pytest.mark.asyncio
async def test_zenodo_connector_skips_records_without_title() -> None:
    """A record carrying no title is skipped, not crashed on."""
    payload: dict[str, object] = {
        "hits": {"hits": [{"metadata": {"description": "No title here."}}]}
    }
    with patch("ingestion.zenodo.httpx.AsyncClient", return_value=_zenodo_client(payload)):
        documents = await ZenodoConnector().search("anything", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_zenodo_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.zenodo.httpx.AsyncClient", return_value=mock_client):
        documents = await ZenodoConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_zenodo_connector_rejects_non_digit_publication_date_year() -> None:
    """A ``publication_date`` that does not start with four digits must not yield a year.

    Zenodo previously took ``publication_date[:4]`` unconditionally, so values
    such as ``unpublished`` or ``TBA`` leaked garbage into ``metadata['year']``.
    Only dates matching ``^\\d{4}`` are accepted.
    """
    payload: dict[str, object] = {
        "hits": {
            "hits": [
                {
                    "metadata": {
                        "title": "Undated Deposit",
                        "creators": [{"name": "Ada Lovelace"}],
                        "publication_date": "unpublished",
                    },
                }
            ]
        }
    }
    with patch("ingestion.zenodo.httpx.AsyncClient", return_value=_zenodo_client(payload)):
        documents = await ZenodoConnector().search("undated", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["year"] == ""
    assert documents[0].text == "By Ada Lovelace"


def _figshare_client(payload: object) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a fixed Figshare JSON payload.

    Args:
        payload: The decoded JSON body the mocked client should return.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    response = httpx.Response(200, json=payload, request=httpx.Request("POST", "http://test"))
    mock_client = AsyncMock()
    mock_client.post.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_figshare_connector_searches_and_normalizes_articles() -> None:
    """FigshareConnector normalizes articles, strips HTML, and prefers public HTML."""
    payload: list[dict[str, object]] = [
        {
            "id": 1434614,
            "title": "Open Retrieval Dataset",
            "doi": "10.6084/m9.figshare.1434614",
            "published_date": "2025-03-15T12:00:00Z",
            "url_public_html": "https://figshare.com/articles/Open_Retrieval_Dataset/1434614",
            "description": "<p>A <b>dataset</b> for retrieval &amp; agents.</p>",
        }
    ]
    with patch("ingestion.figshare.httpx.AsyncClient", return_value=_figshare_client(payload)):
        documents = await FigshareConnector().search("retrieval", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Open Retrieval Dataset"
    assert document.text == "A dataset for retrieval & agents."
    assert document.source == "https://figshare.com/articles/Open_Retrieval_Dataset/1434614"
    assert document.metadata["source_type"] == "figshare"
    assert document.metadata["doi"] == "10.6084/m9.figshare.1434614"
    assert document.metadata["year"] == "2025"


@pytest.mark.asyncio
async def test_figshare_connector_builds_descriptor_and_doi_source_without_description() -> None:
    """An article without a description falls back to a year descriptor and DOI."""
    payload: list[dict[str, object]] = [
        {
            "title": "Figure Only",
            "doi": "10.6084/m9.figshare.999",
            "published_date": "2020-01-01T00:00:00Z",
        }
    ]
    with patch("ingestion.figshare.httpx.AsyncClient", return_value=_figshare_client(payload)):
        documents = await FigshareConnector().search("figure", max_results=1)

    assert len(documents) == 1
    document = documents[0]
    assert document.text == "(2020)"
    assert document.source == "https://doi.org/10.6084/m9.figshare.999"
    assert document.metadata["year"] == "2020"


@pytest.mark.asyncio
async def test_figshare_connector_skips_articles_without_title() -> None:
    """An article carrying no title is skipped, not crashed on."""
    payload: list[dict[str, object]] = [{"description": "No title here.", "doi": "10.0/x"}]
    with patch("ingestion.figshare.httpx.AsyncClient", return_value=_figshare_client(payload)):
        documents = await FigshareConnector().search("anything", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_figshare_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.figshare.httpx.AsyncClient", return_value=mock_client):
        documents = await FigshareConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_figshare_connector_rejects_non_digit_published_date_year() -> None:
    """A ``published_date`` that does not start with four digits must not yield a year.

    Figshare previously took ``published_date[:4]`` unconditionally, so values
    such as ``unpublished`` or ``TBA`` leaked garbage into ``metadata['year']``.
    Only dates matching ``^\\d{4}`` are accepted.
    """
    payload: list[dict[str, object]] = [
        {
            "title": "Undated Figure",
            "doi": "10.6084/m9.figshare.0",
            "published_date": "unpublished",
        }
    ]
    with patch("ingestion.figshare.httpx.AsyncClient", return_value=_figshare_client(payload)):
        documents = await FigshareConnector().search("figure", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["year"] == ""
    assert documents[0].text == ""


def _core_client(payload: object) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a fixed CORE JSON payload.

    Args:
        payload: The decoded JSON body the mocked client should return.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_core_connector_searches_and_normalizes_works() -> None:
    """CoreConnector normalizes works and prefers the display landing page."""
    payload: dict[str, object] = {
        "totalHits": 1,
        "results": [
            {
                "id": 171513974,
                "title": "Open Retrieval Survey",
                "abstract": "A survey of  retrieval   agents.",
                "doi": "10.1007/example.2024",
                "yearPublished": 2024,
                "authors": [{"name": "Ada Lovelace"}, {"name": "Alan Turing"}],
                "downloadUrl": "https://core.ac.uk/download/1.pdf",
                "links": [
                    {"type": "download", "url": "https://core.ac.uk/download/1.pdf"},
                    {"type": "display", "url": "https://core.ac.uk/works/171513974"},
                ],
            }
        ],
    }
    with patch("ingestion.core.httpx.AsyncClient", return_value=_core_client(payload)):
        documents = await CoreConnector().search("retrieval", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Open Retrieval Survey"
    assert document.text == "A survey of retrieval agents."
    assert document.source == "https://core.ac.uk/works/171513974"
    assert document.metadata["source_type"] == "core"
    assert document.metadata["doi"] == "10.1007/example.2024"
    assert document.metadata["year"] == "2024"
    assert document.metadata["authors"] == "Ada Lovelace, Alan Turing"


@pytest.mark.asyncio
async def test_core_connector_builds_descriptor_and_doi_source_without_abstract() -> None:
    """A work without an abstract falls back to a descriptor and DOI source."""
    payload: dict[str, object] = {
        "results": [
            {
                "title": "Dataset Only",
                "doi": "10.5281/core.999",
                "yearPublished": 2020,
                "authors": [{"name": "Grace Hopper"}],
            }
        ]
    }
    with patch("ingestion.core.httpx.AsyncClient", return_value=_core_client(payload)):
        documents = await CoreConnector().search("dataset", max_results=1)

    assert len(documents) == 1
    document = documents[0]
    assert document.text == "By Grace Hopper (2020)"
    assert document.source == "https://doi.org/10.5281/core.999"
    assert document.metadata["year"] == "2020"


@pytest.mark.asyncio
async def test_core_connector_extracts_year_from_date_string() -> None:
    """Date-shaped ``yearPublished`` strings should still yield the publication year."""
    payload: dict[str, object] = {
        "results": [
            {
                "title": "Date-Shaped CORE Work",
                "yearPublished": "2021-07-01",
                "authors": [{"name": "Ada Lovelace"}],
            }
        ]
    }
    with patch("ingestion.core.httpx.AsyncClient", return_value=_core_client(payload)):
        documents = await CoreConnector().search("date-shaped", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["year"] == "2021"
    assert documents[0].text == "By Ada Lovelace (2021)"


@pytest.mark.asyncio
async def test_core_connector_skips_works_without_title() -> None:
    """A work carrying no title is skipped, not crashed on."""
    payload: dict[str, object] = {"results": [{"abstract": "No title here.", "doi": "10.0/x"}]}
    with patch("ingestion.core.httpx.AsyncClient", return_value=_core_client(payload)):
        documents = await CoreConnector().search("anything", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_core_connector_rejects_blank_query() -> None:
    """A blank query yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.core.httpx.AsyncClient", return_value=mock_client):
        documents = await CoreConnector().search("   ", max_results=5)

    assert documents == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_core_connector_rejects_non_positive_max_results() -> None:
    """A non-positive max_results yields no documents and issues no request."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.core.httpx.AsyncClient", return_value=mock_client):
        documents = await CoreConnector().search("anything", max_results=0)

    assert documents == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_core_connector_sends_bearer_api_key_when_configured() -> None:
    """An optional API key is forwarded as a Bearer Authorization header."""
    payload: dict[str, object] = {
        "results": [{"title": "Keyed Work", "yearPublished": 2021, "abstract": "text"}]
    }
    mock_client = _core_client(payload)
    with patch("ingestion.core.httpx.AsyncClient", return_value=mock_client):
        documents = await CoreConnector(api_key="secret-core-key").search("keyed", max_results=1)

    assert len(documents) == 1
    headers = mock_client.get.call_args.kwargs["headers"]
    assert headers == {"Authorization": "Bearer secret-core-key"}


def _orcid_client(*payloads: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning fixed ORCID JSON payloads."""
    responses = [
        httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
        for payload in payloads
    ]
    mock_client = AsyncMock()
    if len(responses) == 1:
        mock_client.get.return_value = responses[0]
    else:
        mock_client.get.side_effect = responses
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _orcid_works_payload() -> dict[str, object]:
    """Return a representative ORCID works payload."""
    return {
        "group": [
            {
                "work-summary": [
                    {
                        "put-code": 12345,
                        "title": {"title": {"value": "Retrieval-Augmented Scholarship"}},
                        "type": "journal-article",
                        "publication-date": {"year": {"value": "2024"}},
                        "journal-title": {"value": "Journal of Scholarly AI"},
                        "url": {"value": "https://example.org/orcid-work"},
                        "external-ids": {
                            "external-id": [
                                {
                                    "external-id-type": "doi",
                                    "external-id-value": "https://doi.org/10.5555/orcid.rag",
                                    "external-id-url": {
                                        "value": "https://doi.org/10.5555/orcid.rag"
                                    },
                                }
                            ]
                        },
                    },
                    {
                        "put-code": 999,
                        "title": {"title": {"value": "Unrelated Plant Metabolomics"}},
                        "type": "dataset",
                        "publication-date": {"year": {"value": "2022"}},
                    },
                ]
            }
        ]
    }


@pytest.mark.asyncio
async def test_orcid_connector_searches_profiles_and_filters_works() -> None:
    """Keyword search finds ORCID profiles, fetches works, and filters by work metadata."""
    search_payload: dict[str, object] = {
        "expanded-result": [
            {
                "orcid-id": "0000-0002-1825-0097",
                "given-names": "Ada",
                "family-names": "Lovelace",
            }
        ]
    }
    mock_client = _orcid_client(search_payload, _orcid_works_payload())

    with patch("ingestion.orcid.httpx.AsyncClient", return_value=mock_client):
        documents = await OrcidConnector().search("retrieval scholarship", max_results=5)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Retrieval-Augmented Scholarship"
    assert document.text == (
        "Retrieval-Augmented Scholarship By Ada Lovelace in Journal of Scholarly AI "
        "type: journal-article DOI 10.5555/orcid.rag (2024)"
    )
    assert document.source == "https://example.org/orcid-work"
    assert document.metadata["source_type"] == "orcid"
    assert document.metadata["orcid"] == "0000-0002-1825-0097"
    assert document.metadata["doi"] == "10.5555/orcid.rag"
    assert document.metadata["year"] == "2024"
    assert document.metadata["authors"] == "Ada Lovelace"

    search_call = mock_client.get.call_args_list[0]
    assert search_call.args[0].endswith("/expanded-search/")
    assert search_call.kwargs["params"] == {"q": "retrieval scholarship", "rows": 5}
    assert mock_client.get.call_args_list[1].args[0].endswith("/0000-0002-1825-0097/works")


@pytest.mark.asyncio
async def test_orcid_connector_resolves_orcid_id_queries_directly() -> None:
    """A bare or URL ORCID iD bypasses profile search and returns works directly."""
    mock_client = _orcid_client(_orcid_works_payload())

    with patch("ingestion.orcid.httpx.AsyncClient", return_value=mock_client):
        documents = await OrcidConnector().search(
            "https://orcid.org/0000-0002-1825-0097",
            max_results=1,
        )

    assert len(documents) == 1
    assert documents[0].metadata["orcid"] == "0000-0002-1825-0097"
    assert documents[0].metadata["authors"] == "0000-0002-1825-0097"
    assert mock_client.get.await_count == 1
    assert mock_client.get.call_args.args[0].endswith("/0000-0002-1825-0097/works")


@pytest.mark.asyncio
async def test_orcid_connector_builds_doi_source_without_work_url() -> None:
    """When an ORCID work lacks a URL, the DOI link is used as source."""
    payload: dict[str, object] = {
        "group": [
            {
                "work-summary": {
                    "put-code": 7,
                    "title": {"title": {"value": "DOI-Only ORCID Work"}},
                    "publication-date": {"year": {"value": 2020}},
                    "external-ids": {
                        "external-id": {
                            "external-id-type": "doi",
                            "external-id-value": "10.1000/orcid-only",
                        }
                    },
                }
            }
        ]
    }
    mock_client = _orcid_client(payload)

    with patch("ingestion.orcid.httpx.AsyncClient", return_value=mock_client):
        documents = await OrcidConnector().search("0000-0002-1825-0097", max_results=1)

    assert len(documents) == 1
    assert documents[0].source == "https://doi.org/10.1000/orcid-only"
    assert documents[0].metadata["year"] == "2020"


@pytest.mark.asyncio
async def test_orcid_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.orcid.httpx.AsyncClient", return_value=mock_client):
        assert await OrcidConnector().search("   ", max_results=5) == []
        assert await OrcidConnector().search("retrieval", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_orcid_connector_skips_works_without_title() -> None:
    """ORCID work summaries without a title are skipped rather than surfaced empty."""
    payload: dict[str, object] = {"group": [{"work-summary": [{"put-code": 1, "type": "other"}]}]}
    mock_client = _orcid_client(payload)

    with patch("ingestion.orcid.httpx.AsyncClient", return_value=mock_client):
        documents = await OrcidConnector().search("0000-0002-1825-0097", max_results=5)

    assert documents == []


def _biorxiv_client(payload: dict[str, object]) -> AsyncMock:
    """Build an AsyncClient mock that returns a bioRxiv details payload."""
    response = httpx.Response(
        200,
        json=payload,
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_biorxiv_connector_searches_and_normalizes_preprints() -> None:
    """BioRxivConnector filters recent posts and normalizes matching preprints."""
    payload: dict[str, object] = {
        "collection": [
            {
                "title": "CRISPR base editing in neurons",
                "authors": "Doe, J.; Smith, A.",
                "doi": "10.1101/2024.01.01.123456",
                "date": "2024-01-02",
                "category": "neuroscience",
                "abstract": "A CRISPR study of neuronal base editing.",
                "server": "biorxiv",
            },
            {
                "title": "Unrelated plant metabolomics",
                "authors": "Lee, B.",
                "doi": "10.1101/2024.01.01.999999",
                "date": "2024-01-03",
                "category": "plant biology",
                "abstract": "Metabolite profiling in Arabidopsis.",
                "server": "biorxiv",
            },
        ]
    }
    mock_client = _biorxiv_client(payload)
    with patch("ingestion.biorxiv.httpx.AsyncClient", return_value=mock_client):
        documents = await BioRxivConnector().search("CRISPR neurons", max_results=5)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "CRISPR base editing in neurons"
    assert document.metadata["source_type"] == "biorxiv"
    assert document.metadata["doi"] == "10.1101/2024.01.01.123456"
    assert document.metadata["year"] == "2024"
    assert document.source == "https://www.biorxiv.org/content/10.1101/2024.01.01.123456"
    assert "CRISPR" in document.text


@pytest.mark.asyncio
async def test_biorxiv_connector_supports_medrxiv_server() -> None:
    """The connector can target the medRxiv server."""
    payload: dict[str, object] = {
        "collection": [
            {
                "title": "COVID vaccine effectiveness cohort",
                "authors": "Ng, C.",
                "doi": "10.1101/2021.03.01.212527",
                "date": "2021-03-02",
                "category": "epidemiology",
                "abstract": "A COVID vaccine effectiveness study.",
                "server": "medrxiv",
            }
        ]
    }
    with patch("ingestion.biorxiv.httpx.AsyncClient", return_value=_biorxiv_client(payload)):
        documents = await BioRxivConnector().search(
            "COVID vaccine", max_results=3, server="medrxiv"
        )

    assert len(documents) == 1
    assert documents[0].metadata["source_type"] == "medrxiv"
    assert documents[0].source.startswith("https://www.medrxiv.org/content/")


@pytest.mark.asyncio
async def test_biorxiv_connector_resolves_doi_queries() -> None:
    """A DOI-shaped query uses the DOI detail endpoint and skips text filtering."""
    payload: dict[str, object] = {
        "collection": [
            {
                "title": "Exact DOI Hit",
                "authors": "Ada, L.",
                "doi": "10.1101/2020.01.01.000001",
                "date": "2020-01-02",
                "category": "bioinformatics",
                "abstract": "",
                "server": "biorxiv",
            }
        ]
    }
    mock_client = _biorxiv_client(payload)
    with patch("ingestion.biorxiv.httpx.AsyncClient", return_value=mock_client):
        documents = await BioRxivConnector().search("10.1101/2020.01.01.000001", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["year"] == "2020"
    assert "By Ada, L." in documents[0].text
    assert "10.1101/2020.01.01.000001" in mock_client.get.call_args.args[0]


@pytest.mark.asyncio
async def test_biorxiv_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.biorxiv.httpx.AsyncClient", return_value=mock_client):
        assert await BioRxivConnector().search("   ", max_results=5) == []
        assert await BioRxivConnector().search("crispr", max_results=0) == []
        assert await BioRxivConnector().search("crispr", max_results=-1) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_biorxiv_connector_rejects_unsupported_server() -> None:
    """An unsupported server name raises ValueError before any HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with (
        patch("ingestion.biorxiv.httpx.AsyncClient", return_value=mock_client),
        pytest.raises(ValueError, match="Unsupported bioRxiv server"),
    ):
        await BioRxivConnector().search("crispr", server="arxiv")

    mock_client.get.assert_not_called()


def _ads_client(payload: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a fixed NASA ADS JSON payload.

    Args:
        payload: Decoded ADS search response body.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_ads_connector_searches_and_normalizes_records() -> None:
    """AdsConnector normalizes ADS ``response.docs`` into documents."""
    payload: dict[str, object] = {
        "response": {
            "docs": [
                {
                    "bibcode": "2024ApJ...900...1A",
                    "title": ["Exoplanet Transit Spectroscopy"],
                    "abstract": "  We measure atmospheric  features.  ",
                    "author": ["Ada, A.", "Bohr, B."],
                    "year": "2024",
                    "doi": ["10.3847/example"],
                    "pub": "ApJ",
                },
                {
                    "bibcode": "2023MNRAS.500.10B",
                    "title": ["Galaxy Formation"],
                    "abstract": "",
                    "author": ["Chen, C."],
                    "year": "2023",
                    "doi": [],
                    "pub": "MNRAS",
                },
            ]
        }
    }
    mock_client = _ads_client(payload)
    with patch("ingestion.ads.httpx.AsyncClient", return_value=mock_client):
        documents = await AdsConnector(api_key="ads-token").search("exoplanet", max_results=5)

    assert len(documents) == 2
    first = documents[0]
    assert first.title == "Exoplanet Transit Spectroscopy"
    assert first.text == "We measure atmospheric features."
    assert first.source == "https://ui.adsabs.harvard.edu/abs/2024ApJ...900...1A"
    assert first.metadata["source_type"] == "ads"
    assert first.metadata["doi"] == "10.3847/example"
    assert first.metadata["year"] == "2024"
    assert first.metadata["authors"] == "Ada, A., Bohr, B."
    assert first.metadata["bibcode"] == "2024ApJ...900...1A"
    assert "By Chen, C." in documents[1].text
    assert "(2023)" in documents[1].text
    params = mock_client.get.call_args.kwargs["params"]
    assert params["q"] == "exoplanet"
    assert params["rows"] == 5
    assert "bibcode" in params["fl"]
    assert mock_client.get.call_args.kwargs["headers"] == {"Authorization": "Bearer ads-token"}


@pytest.mark.asyncio
async def test_ads_connector_builds_doi_source_without_bibcode() -> None:
    """When bibcode is absent the DOI link is used as the source."""
    payload: dict[str, object] = {
        "response": {
            "docs": [
                {
                    "title": ["Untitled Bibcode"],
                    "abstract": "text",
                    "doi": ["10.1000/ads.1"],
                    "year": "2021",
                }
            ]
        }
    }
    with patch("ingestion.ads.httpx.AsyncClient", return_value=_ads_client(payload)):
        documents = await AdsConnector(api_key="ads-token").search("doi", max_results=1)

    assert len(documents) == 1
    assert documents[0].source == "https://doi.org/10.1000/ads.1"


@pytest.mark.asyncio
async def test_ads_connector_skips_records_without_title() -> None:
    """ADS hits without a usable title are skipped."""
    payload: dict[str, object] = {
        "response": {"docs": [{"bibcode": "2020ApJ", "title": [], "abstract": "x"}]}
    }
    with patch("ingestion.ads.httpx.AsyncClient", return_value=_ads_client(payload)):
        documents = await AdsConnector(api_key="ads-token").search("empty", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_ads_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.ads.httpx.AsyncClient", return_value=mock_client):
        assert await AdsConnector(api_key="ads-token").search("   ", max_results=5) == []
        assert await AdsConnector(api_key="ads-token").search("q", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_ads_connector_returns_empty_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ADS token is handled gracefully with no HTTP call."""
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.ads.httpx.AsyncClient", return_value=mock_client):
        documents = await AdsConnector().search("stars", max_results=5)

    assert documents == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_ads_connector_reads_token_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ADS_API_TOKEN`` from the environment is used when no key is passed."""
    monkeypatch.setenv("ADS_API_TOKEN", "env-ads-token")
    payload: dict[str, object] = {
        "response": {"docs": [{"title": ["From Env"], "year": "2020", "abstract": "a"}]}
    }
    mock_client = _ads_client(payload)
    with patch("ingestion.ads.httpx.AsyncClient", return_value=mock_client):
        documents = await AdsConnector().search("env", max_results=1)

    assert len(documents) == 1
    assert mock_client.get.call_args.kwargs["headers"] == {"Authorization": "Bearer env-ads-token"}


def _datacite_client(payload: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a fixed DataCite JSON payload.

    Args:
        payload: Decoded DataCite search response body.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_datacite_connector_searches_and_normalizes_dois() -> None:
    """DataCiteConnector normalizes JSON:API DOI resources into documents."""
    payload: dict[str, object] = {
        "data": [
            {
                "id": "10.5281/zenodo.123",
                "type": "dois",
                "attributes": {
                    "doi": "10.5281/zenodo.123",
                    "titles": [{"title": "Climate Dataset"}],
                    "creators": [{"name": "Ada, A."}, {"givenName": "Bob", "familyName": "Bohr"}],
                    "descriptions": [
                        {
                            "description": "  A curated  climate dataset.  ",
                            "descriptionType": "Abstract",
                        }
                    ],
                    "publicationYear": 2024,
                    "publisher": {"name": "Zenodo"},
                    "url": "https://zenodo.org/records/123",
                    "types": {"resourceTypeGeneral": "Dataset"},
                },
            },
            {
                "id": "10.1234/soft.1",
                "attributes": {
                    "doi": "10.1234/soft.1",
                    "titles": [{"title": "Analysis Toolkit"}],
                    "creators": [{"name": "Chen, C."}],
                    "descriptions": [],
                    "publicationYear": 2023,
                    "publisher": "Example Press",
                    "types": {"resourceTypeGeneral": "Software"},
                },
            },
        ]
    }
    mock_client = _datacite_client(payload)
    with patch("ingestion.datacite.httpx.AsyncClient", return_value=mock_client):
        documents = await DataCiteConnector().search("climate", max_results=5)

    assert len(documents) == 2
    first = documents[0]
    assert first.title == "Climate Dataset"
    assert first.text == "A curated climate dataset."
    assert first.source == "https://zenodo.org/records/123"
    assert first.metadata["source_type"] == "datacite"
    assert first.metadata["doi"] == "10.5281/zenodo.123"
    assert first.metadata["year"] == "2024"
    assert first.metadata["authors"] == "Ada, A., Bob Bohr"
    assert first.metadata["publisher"] == "Zenodo"
    assert first.metadata["resource_type"] == "Dataset"
    assert documents[1].source == "https://doi.org/10.1234/soft.1"
    assert "By Chen, C." in documents[1].text
    assert "via Example Press" in documents[1].text
    params = mock_client.get.call_args.kwargs["params"]
    assert params["query"] == "climate"
    assert params["page[size]"] == 5


@pytest.mark.asyncio
async def test_datacite_connector_prefers_abstract_description() -> None:
    """Abstract descriptionType is preferred over other descriptions."""
    payload: dict[str, object] = {
        "data": [
            {
                "attributes": {
                    "doi": "10.1/x",
                    "titles": [{"title": "T"}],
                    "descriptions": [
                        {"description": "Other notes", "descriptionType": "Other"},
                        {"description": "The abstract", "descriptionType": "Abstract"},
                    ],
                    "publicationYear": 2022,
                }
            }
        ]
    }
    with patch("ingestion.datacite.httpx.AsyncClient", return_value=_datacite_client(payload)):
        documents = await DataCiteConnector().search("t", max_results=1)

    assert documents[0].text == "The abstract"


@pytest.mark.asyncio
async def test_datacite_connector_skips_records_without_title() -> None:
    """DOI records without a usable title are skipped."""
    payload: dict[str, object] = {
        "data": [{"attributes": {"doi": "10.1/y", "titles": [], "publicationYear": 2020}}]
    }
    with patch("ingestion.datacite.httpx.AsyncClient", return_value=_datacite_client(payload)):
        documents = await DataCiteConnector().search("empty", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_datacite_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.datacite.httpx.AsyncClient", return_value=mock_client):
        assert await DataCiteConnector().search("   ", max_results=5) == []
        assert await DataCiteConnector().search("q", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_datacite_connector_accepts_float_like_publication_year() -> None:
    """Float-like publication years from DataCite must normalize to four digits.

    Some JSON serializers and upstream transformations represent
    ``publicationYear`` as ``2024.0`` or ``"2024.0"``. The old digit-only parser
    dropped those usable years entirely; integer-valued floats should preserve
    the publication year.
    """
    payload: dict[str, object] = {
        "data": [
            {
                "attributes": {
                    "doi": "10.1/float",
                    "titles": [{"title": "Float Year Dataset"}],
                    "publicationYear": 2024.0,
                }
            },
            {
                "attributes": {
                    "doi": "10.1/float-string",
                    "titles": [{"title": "Float String Year Dataset"}],
                    "publicationYear": "2023.0",
                }
            },
        ]
    }
    with patch("ingestion.datacite.httpx.AsyncClient", return_value=_datacite_client(payload)):
        documents = await DataCiteConnector().search("float years", max_results=2)

    assert [document.metadata["year"] for document in documents] == ["2024", "2023"]


def _opencitations_client(responses: list[httpx.Response]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning OpenCitations responses.

    Args:
        responses: Responses yielded by successive GET calls.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    mock_client = AsyncMock()
    mock_client.get.side_effect = responses
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_opencitations_connector_fetches_doi_metadata_and_counts() -> None:
    """OpenCitationsConnector normalizes Meta metadata and Index counts."""
    metadata_response = httpx.Response(
        200,
        json=[
            {
                "id": (
                    "doi:10.1038/227680a0 openalex:W2100837269 pmid:5432063 omid:br/06190356582"
                ),
                "title": (
                    "Cleavage Of Structural Proteins During The Assembly Of The Head "
                    "Of Bacteriophage T4"
                ),
                "author": "Laemmli, U. K. [omid:ra/061901010373]",
                "pub_date": "1970-08",
                "venue": ("Nature [issn:0028-0836 issn:1465-7392 omid:br/0626016512]"),
                "type": "journal article",
            }
        ],
        request=httpx.Request("GET", "http://test"),
    )
    citation_count_response = httpx.Response(
        200,
        json=[{"count": "19000"}],
        request=httpx.Request("GET", "http://test"),
    )
    reference_count_response = httpx.Response(
        200,
        json=[{"count": 19}],
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = _opencitations_client(
        [metadata_response, citation_count_response, reference_count_response]
    )

    with patch("ingestion.opencitations.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenCitationsConnector().search(
            "https://doi.org/10.1038/227680a0",
            max_results=5,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title.startswith("Cleavage Of Structural Proteins")
    assert document.text == "By Laemmli, U. K. in Nature [journal article] (1970)"
    assert document.source == "https://doi.org/10.1038/227680a0"
    assert document.metadata["source_type"] == "opencitations"
    assert document.metadata["doi"] == "10.1038/227680a0"
    assert document.metadata["year"] == "1970"
    assert document.metadata["authors"] == "Laemmli, U. K."
    assert document.metadata["venue"] == "Nature"
    assert document.metadata["type"] == "journal article"
    assert document.metadata["citation_count"] == "19000"
    assert document.metadata["reference_count"] == "19"

    metadata_call = mock_client.get.await_args_list[0]
    assert metadata_call.args[0].endswith("/doi:10.1038/227680a0")
    assert (
        mock_client.get.await_args_list[1].args[0].endswith("/citation-count/doi:10.1038/227680a0")
    )
    assert (
        mock_client.get.await_args_list[2].args[0].endswith("/reference-count/doi:10.1038/227680a0")
    )


@pytest.mark.asyncio
async def test_opencitations_connector_extracts_unique_doi_list_and_token_header() -> None:
    """Free text may contain a DOI list; duplicate DOIs are fetched once."""
    metadata_response = httpx.Response(
        200,
        json=[],
        request=httpx.Request("GET", "http://test"),
    )
    citation_count_response = httpx.Response(
        200,
        json=[],
        request=httpx.Request("GET", "http://test"),
    )
    reference_count_response = httpx.Response(
        200,
        json=[],
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = _opencitations_client(
        [metadata_response, citation_count_response, reference_count_response]
    )

    with patch("ingestion.opencitations.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenCitationsConnector(access_token="oc-token").search(  # noqa: S106
            "Compare DOI:10.1234/Alpha and https://doi.org/10.1234/alpha.",
            max_results=5,
        )

    assert documents == []
    assert mock_client.get.await_count == 3
    assert mock_client.get.await_args_list[0].args[0].endswith("/doi:10.1234/Alpha")
    assert mock_client.get.await_args_list[0].kwargs["headers"] == {"authorization": "oc-token"}


@pytest.mark.asyncio
async def test_opencitations_connector_rejects_blank_non_positive_and_non_doi() -> None:
    """Blank, non-positive, and non-DOI queries short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.opencitations.httpx.AsyncClient", return_value=mock_client):
        assert await OpenCitationsConnector().search("   ", max_results=5) == []
        assert await OpenCitationsConnector().search("10.1000/example", max_results=0) == []
        assert await OpenCitationsConnector().search("graph retrieval", max_results=5) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_opencitations_connector_keeps_metadata_when_count_fails() -> None:
    """Slow or failing Index counts must not drop usable Meta metadata."""
    metadata_response = httpx.Response(
        200,
        json=[
            {
                "id": "doi:10.5555/fail-count omid:br/1",
                "title": "Metadata Without Counts",
                "pub_date": "2022",
            }
        ],
        request=httpx.Request("GET", "http://test"),
    )
    failing_count_response = httpx.Response(
        503,
        json={"message": "temporarily unavailable"},
        request=httpx.Request("GET", "http://test"),
    )
    reference_count_response = httpx.Response(
        200,
        json=[{"count": "3"}],
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = _opencitations_client(
        [metadata_response, failing_count_response, reference_count_response]
    )

    with patch("ingestion.opencitations.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenCitationsConnector().search("10.5555/fail-count", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["citation_count"] == ""
    assert documents[0].metadata["reference_count"] == "3"


def _osf_client(payloads: list[object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning OSF JSON:API payloads."""
    responses = [
        httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
        for payload in payloads
    ]
    mock_client = AsyncMock()
    mock_client.get.side_effect = responses
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_osf_connector_searches_preprints_and_registrations() -> None:
    """OsfConnector queries both OSF endpoints and normalizes JSON:API records."""
    preprints_payload: dict[str, object] = {
        "data": [
            {
                "id": "pre123",
                "attributes": {
                    "title": "Open Science Preprint",
                    "description": "  OSF stores transparent   preprint workflows. ",
                    "date_published": "2025-04-12T10:00:00Z",
                    "category": "psychology",
                },
                "links": {
                    "html": "https://osf.io/preprints/psyarxiv/pre123",
                    "preprint_doi": "https://doi.org/10.31234/osf.io/pre123",
                },
                "embeds": {
                    "contributors": {
                        "data": [
                            {
                                "embeds": {
                                    "users": {"data": {"attributes": {"full_name": "Ada Lovelace"}}}
                                }
                            },
                            {"attributes": {"bibliographic": "Alan Turing"}},
                        ]
                    }
                },
            }
        ]
    }
    registrations_payload: dict[str, object] = {
        "data": [
            {
                "id": "reg456",
                "attributes": {
                    "title": "Registered Replication Protocol",
                    "description": "",
                    "date_registered": "2024-01-20T12:00:00Z",
                    "category": "project",
                    "doi": "10.17605/OSF.IO/REG456",
                },
                "links": {"html": "https://osf.io/reg456/"},
            }
        ]
    }
    mock_client = _osf_client([preprints_payload, registrations_payload])

    with patch("ingestion.osf.httpx.AsyncClient", return_value=mock_client):
        documents = await OsfConnector().search("open science", max_results=5)

    assert len(documents) == 2
    preprint = documents[0]
    assert preprint.title == "Open Science Preprint"
    assert preprint.text == "OSF stores transparent preprint workflows."
    assert preprint.source == "https://osf.io/preprints/psyarxiv/pre123"
    assert preprint.metadata["source_type"] == "osf"
    assert preprint.metadata["resource_type"] == "preprint"
    assert preprint.metadata["doi"] == "10.31234/osf.io/pre123"
    assert preprint.metadata["year"] == "2025"
    assert preprint.metadata["authors"] == "Ada Lovelace, Alan Turing"

    registration = documents[1]
    assert registration.metadata["resource_type"] == "registration"
    assert registration.metadata["doi"] == "10.17605/OSF.IO/REG456"
    assert registration.metadata["year"] == "2024"
    assert registration.text == "in project (2024)"

    first_call = mock_client.get.call_args_list[0]
    assert first_call.args[0] == "https://api.osf.io/v2/preprints/"
    assert first_call.kwargs["params"]["filter[search]"] == "open science"
    assert first_call.kwargs["params"]["page[size]"] == 5
    assert mock_client.get.call_args_list[1].args[0] == "https://api.osf.io/v2/registrations/"


@pytest.mark.asyncio
async def test_osf_connector_returns_empty_when_api_unavailable() -> None:
    """OSF network failures are handled gracefully with no exception."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError(
        "OSF API unavailable",
        request=httpx.Request("GET", "https://api.osf.io/v2/preprints/"),
    )
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.osf.httpx.AsyncClient", return_value=mock_client):
        documents = await OsfConnector().search("open science", max_results=3)

    assert documents == []
    assert mock_client.get.await_count == 2


@pytest.mark.asyncio
async def test_osf_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.osf.httpx.AsyncClient", return_value=mock_client):
        assert await OsfConnector().search("   ", max_results=5) == []
        assert await OsfConnector().search("open science", max_results=0) == []

    mock_client.get.assert_not_called()


def _unpaywall_client(responses: list[httpx.Response]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning Unpaywall API responses."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = responses
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_unpaywall_connector_extracts_dois_and_normalizes_oa_location() -> None:
    """UnpaywallConnector extracts DOI text and returns OA landing/PDF metadata."""
    response = httpx.Response(
        200,
        json={
            "doi": "10.1234/rag.oa",
            "doi_url": "https://doi.org/10.1234/rag.oa",
            "title": "Open Access Retrieval Study",
            "year": 2025,
            "published_date": "2025-03-14",
            "journal_name": "Journal of Open Retrieval",
            "publisher": "Example Publisher",
            "genre": "journal-article",
            "is_oa": True,
            "oa_status": "green",
            "z_authors": [{"given": "Ada", "family": "Lovelace"}, {"name": "Alan Turing"}],
            "best_oa_location": {
                "url_for_landing_page": "https://repository.example.org/rag-oa",
                "url_for_pdf": "https://repository.example.org/rag-oa.pdf",
                "host_type": "repository",
                "version": "acceptedVersion",
                "license": "cc-by",
            },
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = _unpaywall_client([response])

    with patch("ingestion.unpaywall.httpx.AsyncClient", return_value=mock_client):
        documents = await UnpaywallConnector(email="dev@example.org").search(
            "Read https://doi.org/10.1234/rag.oa, please.",
            max_results=3,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Open Access Retrieval Study"
    assert document.source == "https://repository.example.org/rag-oa"
    assert "OA status: green" in document.text
    assert "PDF: https://repository.example.org/rag-oa.pdf" in document.text
    assert document.metadata["source_type"] == "unpaywall"
    assert document.metadata["doi"] == "10.1234/rag.oa"
    assert document.metadata["year"] == "2025"
    assert document.metadata["authors"] == "Ada Lovelace, Alan Turing"
    assert document.metadata["is_oa"] == "true"
    assert document.metadata["landing_url"] == "https://repository.example.org/rag-oa"
    assert document.metadata["pdf_url"] == "https://repository.example.org/rag-oa.pdf"
    assert document.metadata["host_type"] == "repository"
    assert document.metadata["version"] == "acceptedVersion"
    assert document.metadata["license"] == "cc-by"

    call = mock_client.get.await_args
    assert call.args[0] == "https://api.unpaywall.org/v2/10.1234/rag.oa"
    assert call.kwargs["params"] == {"email": "dev@example.org"}


@pytest.mark.asyncio
async def test_unpaywall_connector_extracts_unique_dois_and_uses_fallback_location() -> None:
    """Free text DOI lists are de-duplicated and fallback OA locations are used."""
    first = httpx.Response(
        200,
        json={
            "doi": "10.5555/fallback",
            "title": "Fallback OA Location",
            "published_date": "2021-10-01",
            "is_oa": True,
            "oa_locations": [
                {
                    "url": "https://publisher.example.org/article",
                    "url_for_pdf": "https://publisher.example.org/article.pdf",
                    "host_type": "publisher",
                }
            ],
        },
        request=httpx.Request("GET", "http://test"),
    )
    second = httpx.Response(
        200,
        json={
            "doi": "10.5555/closed",
            "title": "Closed Access Record",
            "year": 2020,
            "is_oa": False,
            "doi_url": "https://doi.org/10.5555/closed",
        },
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = _unpaywall_client([first, second])

    with patch("ingestion.unpaywall.httpx.AsyncClient", return_value=mock_client):
        documents = await UnpaywallConnector(email="dev@example.org").search(
            "DOI:10.5555/fallback; duplicate 10.5555/FALLBACK and 10.5555/closed",
            max_results=5,
        )

    assert [document.metadata["doi"] for document in documents] == [
        "10.5555/fallback",
        "10.5555/closed",
    ]
    assert documents[0].source == "https://publisher.example.org/article"
    assert documents[0].metadata["year"] == "2021"
    assert documents[0].metadata["pdf_url"] == "https://publisher.example.org/article.pdf"
    assert documents[1].source == "https://doi.org/10.5555/closed"
    assert documents[1].metadata["is_oa"] == "false"
    assert "OA status: closed" in documents[1].text
    assert mock_client.get.await_count == 2


@pytest.mark.asyncio
async def test_unpaywall_connector_rejects_blank_non_positive_missing_email_and_non_doi() -> None:
    """Invalid queries and missing email configuration short-circuit without HTTP."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.unpaywall.httpx.AsyncClient", return_value=mock_client):
        assert await UnpaywallConnector(email="dev@example.org").search("   ", max_results=5) == []
        assert (
            await UnpaywallConnector(email="dev@example.org").search(
                "10.1234/example",
                max_results=0,
            )
            == []
        )
        assert await UnpaywallConnector(email="dev@example.org").search("graph retrieval") == []
        assert await UnpaywallConnector().search("10.1234/example") == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_unpaywall_connector_skips_failed_lookup() -> None:
    """An unavailable DOI lookup is skipped rather than failing the batch."""
    failing = httpx.Response(
        404,
        json={"error": True},
        request=httpx.Request("GET", "http://test"),
    )
    succeeding = httpx.Response(
        200,
        json={"doi": "10.7777/ok", "title": "Recovered OA Record", "is_oa": False},
        request=httpx.Request("GET", "http://test"),
    )
    mock_client = _unpaywall_client([failing, succeeding])

    with patch("ingestion.unpaywall.httpx.AsyncClient", return_value=mock_client):
        documents = await UnpaywallConnector(email="dev@example.org").search(
            "10.7777/missing 10.7777/ok",
            max_results=2,
        )

    assert len(documents) == 1
    assert documents[0].title == "Recovered OA Record"
    assert documents[0].metadata["doi"] == "10.7777/ok"


def _retraction_client(payload: object, *, status_code: int = 200) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning an OpenAlex retraction payload."""
    mock_client = AsyncMock()
    if status_code >= 400:
        response = httpx.Response(
            status_code,
            request=httpx.Request("GET", "http://test"),
        )
    else:
        response = httpx.Response(
            status_code,
            json=payload,
            request=httpx.Request("GET", "http://test"),
        )
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _retracted_work_payload() -> dict[str, object]:
    """Return a representative OpenAlex retracted work search payload."""
    return {
        "results": [
            {
                "id": "https://openalex.org/W2158048826",
                "doi": "https://doi.org/10.1038/nature00870",
                "title": "RETRACTED ARTICLE: Pluripotency of mesenchymal stem cells",
                "display_name": "RETRACTED ARTICLE: Pluripotency of mesenchymal stem cells",
                "publication_year": 2002,
                "is_retracted": True,
                "cited_by_count": 5532,
                "authorships": [
                    {"author": {"display_name": "Yuehua Jiang"}},
                    {"author": {"display_name": "Balkrishna Jahagirdar"}},
                ],
                "primary_location": {
                    "landing_page_url": "https://doi.org/10.1038/nature00870",
                    "source": {"display_name": "Nature"},
                },
                "abstract_inverted_index": {
                    "Retracted": [0],
                    "stem": [1],
                    "cell": [2],
                    "claim.": [3],
                },
            },
            {
                "id": "https://openalex.org/W999",
                "title": "Not actually retracted",
                "is_retracted": False,
            },
        ]
    }


@pytest.mark.asyncio
async def test_retraction_watch_connector_filters_retracted_openalex_works() -> None:
    """RetractionWatchConnector searches OpenAlex with is_retracted:true."""
    mock_client = _retraction_client(_retracted_work_payload())

    with patch("ingestion.retraction_watch.httpx.AsyncClient", return_value=mock_client):
        documents = await RetractionWatchConnector(mailto="dev@example.org").search(
            "stem cell",
            max_results=5,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "RETRACTED ARTICLE: Pluripotency of mesenchymal stem cells"
    assert document.text == "Retracted stem cell claim."
    assert document.source == "https://openalex.org/W2158048826"
    assert document.metadata["source_type"] == "retraction_watch"
    assert document.metadata["is_retracted"] == "true"
    assert document.metadata["doi"] == "10.1038/nature00870"
    assert document.metadata["year"] == "2002"
    assert document.metadata["authors"] == "Yuehua Jiang, Balkrishna Jahagirdar"
    assert document.metadata["journal"] == "Nature"
    assert document.metadata["openalex_id"] == "https://openalex.org/W2158048826"
    assert document.metadata["cited_by_count"] == "5532"

    call = mock_client.get.await_args
    assert call.args[0] == "https://api.openalex.org/works"
    assert call.kwargs["params"]["filter"] == "is_retracted:true"
    assert call.kwargs["params"]["search"] == "stem cell"
    assert call.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_retraction_watch_connector_synthesizes_text_without_abstract() -> None:
    """When OpenAlex omits an abstract, a retraction descriptor is synthesized."""
    payload: dict[str, object] = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "doi": "10.1000/retracted",
                "title": "Retracted Methods Paper",
                "publication_year": 2019,
                "is_retracted": True,
                "cited_by_count": 12,
                "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                "primary_location": {"source": {"display_name": "Fake Journal"}},
            }
        ]
    }
    mock_client = _retraction_client(payload)

    with patch("ingestion.retraction_watch.httpx.AsyncClient", return_value=mock_client):
        documents = await RetractionWatchConnector().search("methods", max_results=1)

    assert len(documents) == 1
    assert documents[0].text == (
        "Retracted work. By Ada Lovelace in Fake Journal (2019) "
        "DOI 10.1000/retracted cited_by_count=12"
    )


@pytest.mark.asyncio
async def test_retraction_watch_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.retraction_watch.httpx.AsyncClient", return_value=mock_client):
        assert await RetractionWatchConnector().search("   ", max_results=5) == []
        assert await RetractionWatchConnector().search("stem cell", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_retraction_watch_connector_skips_failed_lookup() -> None:
    """An unavailable OpenAlex response yields an empty list rather than raising."""
    mock_client = _retraction_client({}, status_code=503)

    with patch("ingestion.retraction_watch.httpx.AsyncClient", return_value=mock_client):
        documents = await RetractionWatchConnector().search("stem cell", max_results=3)

    assert documents == []


@pytest.mark.asyncio
async def test_retraction_watch_connector_skips_untitled_works() -> None:
    """Retracted OpenAlex works without a title are skipped."""
    payload: dict[str, object] = {
        "results": [{"id": "https://openalex.org/W2", "is_retracted": True}]
    }
    mock_client = _retraction_client(payload)

    with patch("ingestion.retraction_watch.httpx.AsyncClient", return_value=mock_client):
        documents = await RetractionWatchConnector().search("anything", max_results=5)

    assert documents == []


def _crossref_events_client(payload: object, *, status_code: int = 200) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a Crossref Event Data payload."""
    mock_client = AsyncMock()
    if status_code >= 400:
        response = httpx.Response(
            status_code,
            request=httpx.Request("GET", "http://test"),
        )
    else:
        response = httpx.Response(
            status_code,
            json=payload,
            request=httpx.Request("GET", "http://test"),
        )
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _crossref_event_payload() -> dict[str, object]:
    """Return a representative Crossref Event Data search payload."""
    return {
        "status": "ok",
        "message-type": "event-list",
        "message": {
            "total-results": 1,
            "items": [
                {
                    "id": "615cf92e-9922-4868-9b62-a51b8efd29ee",
                    "occurred_at": "2016-10-12T07:20:40.000Z",
                    "timestamp": "2017-02-20T07:20:40.000Z",
                    "subj_id": (
                        "https://reddit.com/r/math/comments/572xbh/"
                        "five_stages_of_accepting_constructive_mathematics/"
                    ),
                    "obj_id": "https://doi.org/10.1090/bull/1556",
                    "relation_type_id": "discusses",
                    "source_id": "reddit",
                    "subj": {
                        "pid": (
                            "https://reddit.com/r/math/comments/572xbh/"
                            "five_stages_of_accepting_constructive_mathematics/"
                        ),
                        "type": "post",
                        "title": (
                            "Five stages of accepting constructive mathematics, "
                            "by Andrej Bauer [abstract + link to PDF]"
                        ),
                        "issued": "2016-10-12T07:20:40.000Z",
                    },
                    "obj": {
                        "pid": "https://doi.org/10.1090/bull/1556",
                        "url": (
                            "http://www.ams.org/journals/bull/0000-000-00/"
                            "S0273-0979-2016-01556-4/home.html"
                        ),
                    },
                    "evidence-record": (
                        "https://evidence.eventdata.crossref.org/evidence/"
                        "2017022284421dfd-ddbe-4730-bc35-caf11d92231f"
                    ),
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_crossref_events_connector_searches_bibliographic_and_normalizes() -> None:
    """CrossrefEventsConnector normalizes Event Data items from bibliographic search."""
    mock_client = _crossref_events_client(_crossref_event_payload())

    with patch("ingestion.crossref_events.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefEventsConnector(mailto="dev@example.org").search(
            "constructive mathematics",
            max_results=3,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title.startswith("Five stages of accepting constructive mathematics")
    assert "Crossref Event Data mention." in document.text
    assert "Relation: discusses." in document.text
    assert document.source.startswith("https://evidence.eventdata.crossref.org/evidence/")
    assert document.metadata["source_type"] == "crossref_events"
    assert document.metadata["event_id"] == "615cf92e-9922-4868-9b62-a51b8efd29ee"
    assert document.metadata["relation_type"] == "discusses"
    assert document.metadata["source_id"] == "reddit"
    assert document.metadata["obj_doi"] == "10.1090/bull/1556"
    assert document.metadata["year"] == "2016"
    assert document.metadata["subj_title"].startswith("Five stages of accepting")

    call = mock_client.get.await_args
    assert call.args[0] == "https://api.eventdata.crossref.org/v1/events"
    assert call.kwargs["params"]["rows"] == 3
    assert call.kwargs["params"]["query.bibliographic"] == "constructive mathematics"
    assert call.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_crossref_events_connector_uses_obj_id_for_doi_queries() -> None:
    """DOI-shaped queries use obj-id instead of query.bibliographic."""
    mock_client = _crossref_events_client(_crossref_event_payload())

    with patch("ingestion.crossref_events.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefEventsConnector().search(
            "https://doi.org/10.1090/bull/1556",
            max_results=5,
        )

    assert len(documents) == 1
    call = mock_client.get.await_args
    assert call.kwargs["params"]["obj-id"] == "10.1090/bull/1556"
    assert "query.bibliographic" not in call.kwargs["params"]


@pytest.mark.asyncio
async def test_crossref_events_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.crossref_events.httpx.AsyncClient", return_value=mock_client):
        assert await CrossrefEventsConnector().search("   ", max_results=5) == []
        assert await CrossrefEventsConnector().search("machine learning", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_crossref_events_connector_handles_failed_lookup() -> None:
    """An unavailable Event Data response yields an empty list rather than raising."""
    mock_client = _crossref_events_client({}, status_code=503)

    with patch("ingestion.crossref_events.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefEventsConnector().search("machine learning", max_results=3)

    assert documents == []


@pytest.mark.asyncio
async def test_crossref_events_connector_synthesizes_title_without_subj_title() -> None:
    """Events without subject titles still normalize using relation and DOI metadata."""
    payload: dict[str, object] = {
        "message": {
            "items": [
                {
                    "id": "event-no-title",
                    "obj_id": "https://doi.org/10.5555/altmetric",
                    "relation_type_id": "references",
                    "source_id": "wikipedia",
                    "subj_id": "https://en.wikipedia.org/wiki/Example",
                }
            ]
        }
    }
    mock_client = _crossref_events_client(payload)

    with patch("ingestion.crossref_events.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefEventsConnector().search("altmetric", max_results=1)

    assert len(documents) == 1
    assert documents[0].title == "references on wikipedia for DOI 10.5555/altmetric"
    assert documents[0].metadata["subj_title"] == ""


def _dryad_client(payload: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a fixed Dryad JSON payload.

    Args:
        payload: Decoded Dryad search response body.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get = AsyncMock(return_value=response)
    return mock_client


def _dryad_dataset_payload() -> dict[str, object]:
    """Return a minimal Dryad search payload with one dataset."""
    return {
        "_embedded": {
            "stash:datasets": [
                {
                    "id": 102889,
                    "identifier": "doi:10.5061/dryad.hx3ffbgjj",
                    "title": "The climatic drivers of long-term population changes",
                    "publicationDate": "2023-01-19",
                    "fieldOfScience": "Natural sciences",
                    "sharingLink": "http://datadryad.org/dataset/doi:10.5061/dryad.hx3ffbgjj",
                    "authors": [
                        {"firstName": "Alejandro", "lastName": "de la Fuente"},
                        {"firstName": "Stephen", "lastName": "Williams"},
                    ],
                    "abstract": "<p>Climate-driven biodiversity erosion is escalating.</p>",
                }
            ]
        }
    }


@pytest.mark.asyncio
async def test_dryad_connector_searches_and_normalizes_datasets() -> None:
    """DryadConnector normalizes search datasets, strips HTML, and prefers sharingLink."""
    mock_client = _dryad_client(_dryad_dataset_payload())

    with patch("ingestion.dryad.httpx.AsyncClient", return_value=mock_client):
        documents = await DryadConnector().search("climate birds", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "The climatic drivers of long-term population changes"
    assert document.text == "Climate-driven biodiversity erosion is escalating."
    assert document.source == "http://datadryad.org/dataset/doi:10.5061/dryad.hx3ffbgjj"
    assert document.metadata["source_type"] == "dryad"
    assert document.metadata["doi"] == "10.5061/dryad.hx3ffbgjj"
    assert document.metadata["year"] == "2023"
    assert document.metadata["authors"] == "Alejandro de la Fuente, Stephen Williams"
    assert document.metadata["field_of_science"] == "Natural sciences"
    assert document.metadata["dryad_id"] == "102889"
    mock_client.get.assert_awaited_once()
    call_kwargs = mock_client.get.await_args.kwargs
    assert call_kwargs["params"]["q"] == "climate birds"
    assert call_kwargs["params"]["per_page"] == 3


@pytest.mark.asyncio
async def test_dryad_connector_builds_descriptor_and_doi_source_without_abstract() -> None:
    """A dataset without an abstract falls back to a descriptor and DOI source."""
    payload: dict[str, object] = {
        "_embedded": {
            "stash:datasets": [
                {
                    "id": 42,
                    "identifier": "doi:10.5061/dryad.abc123",
                    "title": "Supplemental Tables",
                    "publicationDate": "2020-06-01",
                    "fieldOfScience": "Biological sciences",
                    "authors": [{"firstName": "Ada", "lastName": "Lovelace"}],
                }
            ]
        }
    }
    with patch("ingestion.dryad.httpx.AsyncClient", return_value=_dryad_client(payload)):
        documents = await DryadConnector().search("tables", max_results=1)

    assert len(documents) == 1
    document = documents[0]
    assert document.text == "By Ada Lovelace (Biological sciences) (2020)"
    assert document.source == "https://doi.org/10.5061/dryad.abc123"
    assert document.metadata["year"] == "2020"


@pytest.mark.asyncio
async def test_dryad_connector_skips_datasets_without_title() -> None:
    """A dataset carrying no title is skipped, not crashed on."""
    payload: dict[str, object] = {
        "_embedded": {
            "stash:datasets": [
                {
                    "identifier": "doi:10.5061/dryad.empty",
                    "abstract": "No title here.",
                }
            ]
        }
    }
    with patch("ingestion.dryad.httpx.AsyncClient", return_value=_dryad_client(payload)):
        documents = await DryadConnector().search("anything", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_dryad_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.dryad.httpx.AsyncClient", return_value=mock_client):
        assert await DryadConnector().search("   ", max_results=5) == []
        assert await DryadConnector().search("q", max_results=0) == []

    mock_client.get.assert_not_called()


def _openalex_topics_client(
    payloads: list[tuple[str, object]],
    *,
    status_code: int = 200,
) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning OpenAlex topic payloads."""
    mock_client = AsyncMock()

    def _response_for(url: str) -> httpx.Response:
        for target_url, payload in payloads:
            if url == target_url:
                if status_code >= 400:
                    return httpx.Response(
                        status_code,
                        request=httpx.Request("GET", url),
                    )
                return httpx.Response(
                    status_code,
                    json=payload,
                    request=httpx.Request("GET", url),
                )
        return httpx.Response(
            status_code,
            json={},
            request=httpx.Request("GET", url),
        )

    async def _get(url: str, params: object = None) -> httpx.Response:
        return _response_for(url)

    mock_client.get.side_effect = _get
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _openalex_topics_search_payload() -> dict[str, object]:
    """Return a representative OpenAlex topics search payload."""
    return {
        "results": [
            {
                "id": "https://openalex.org/T11948",
                "display_name": "Machine Learning in Materials Science",
                "description": (
                    "Materials informatics and machine learning for property predictions."
                ),
                "works_count": 142968,
                "subfield": {"display_name": "Materials Chemistry"},
                "field": {"display_name": "Materials Science"},
                "domain": {"display_name": "Physical Sciences"},
            },
            {
                "id": "https://openalex.org/T12254",
                "display_name": "Machine Learning in Bioinformatics",
                "description": "",
                "works_count": 278481,
            },
            {"display_name": ""},
        ]
    }


@pytest.mark.asyncio
async def test_openalex_topics_connector_searches_and_normalizes_topics() -> None:
    """OpenAlexTopicsConnector searches the topics API for free-text queries."""
    mock_client = _openalex_topics_client(
        [("https://api.openalex.org/topics", _openalex_topics_search_payload())]
    )

    with patch("ingestion.openalex_topics.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexTopicsConnector(mailto="dev@example.org").search(
            "machine learning",
            max_results=5,
        )

    assert len(documents) == 2
    document = documents[0]
    assert document.title == "Machine Learning in Materials Science"
    assert document.text == ("Materials informatics and machine learning for property predictions.")
    assert document.source == "https://openalex.org/T11948"
    assert document.metadata["source_type"] == "openalex_topics"
    assert document.metadata["topic_id"] == "T11948"
    assert document.metadata["works_count"] == "142968"
    assert document.metadata["subfield"] == "Materials Chemistry"
    assert document.metadata["field"] == "Materials Science"
    assert document.metadata["domain"] == "Physical Sciences"

    fallback = documents[1]
    assert fallback.title == "Machine Learning in Bioinformatics"
    assert fallback.text == "OpenAlex topic Machine Learning in Bioinformatics with 278481 works."

    call = mock_client.get.await_args_list[0]
    assert call.args[0] == "https://api.openalex.org/topics"
    assert call.kwargs["params"]["search"] == "machine learning"
    assert call.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_openalex_topics_connector_resolves_topic_id_with_works_filter() -> None:
    """Topic-shaped queries resolve the topic and sample works via topics.id filter."""
    topic_payload = {
        "id": "https://openalex.org/T11948",
        "display_name": "Machine Learning in Materials Science",
        "description": "Topic cluster for materials informatics.",
        "works_count": 142968,
    }
    works_payload = {
        "results": [
            {"title": "Accelerated materials discovery with ML"},
            {"title": "High-throughput screening of perovskites"},
        ]
    }
    mock_client = _openalex_topics_client(
        [
            ("https://api.openalex.org/topics/T11948", topic_payload),
            ("https://api.openalex.org/works", works_payload),
        ]
    )

    with patch("ingestion.openalex_topics.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexTopicsConnector(mailto="dev@example.org").search(
            "T11948",
            max_results=2,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Machine Learning in Materials Science"
    assert "Sample works:" in document.text
    assert "Accelerated materials discovery with ML" in document.text
    assert document.metadata["sample_work_titles"].startswith("Accelerated materials discovery")

    topic_call = mock_client.get.await_args_list[0]
    works_call = mock_client.get.await_args_list[1]
    assert topic_call.args[0] == "https://api.openalex.org/topics/T11948"
    assert works_call.args[0] == "https://api.openalex.org/works"
    assert works_call.kwargs["params"]["filter"] == "topics.id:T11948"


@pytest.mark.asyncio
async def test_openalex_topics_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.openalex_topics.httpx.AsyncClient", return_value=mock_client):
        assert await OpenAlexTopicsConnector().search("   ", max_results=5) == []
        assert await OpenAlexTopicsConnector().search("machine learning", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_openalex_topics_connector_handles_failed_lookup() -> None:
    """An unavailable OpenAlex response yields an empty list rather than raising."""
    mock_client = _openalex_topics_client(
        [("https://api.openalex.org/topics", {})],
        status_code=503,
    )

    with patch("ingestion.openalex_topics.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexTopicsConnector().search("machine learning", max_results=3)

    assert documents == []


def _clinicaltrials_client(payload: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a fixed ClinicalTrials.gov payload.

    Args:
        payload: Decoded ClinicalTrials.gov search response body.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get = AsyncMock(return_value=response)
    return mock_client


def _clinicaltrials_study_payload() -> dict[str, object]:
    """Return a minimal ClinicalTrials.gov search payload with one study."""
    return {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00205335",
                        "briefTitle": "Free Test Strips and Blood Glucose Control",
                        "officialTitle": (
                            "The Impact of Increased Availability of Test Strips "
                            "on Blood Glucose Control in Patients With Diabetes"
                        ),
                    },
                    "statusModule": {
                        "overallStatus": "COMPLETED",
                        "startDateStruct": {"date": "2004-01"},
                    },
                    "descriptionModule": {
                        "briefSummary": (
                            "This study is designed to determine if there is an "
                            "impact on blood glucose control in patients who "
                            "receive free test strips."
                        ),
                    },
                    "conditionsModule": {"conditions": ["Diabetes"]},
                    "designModule": {
                        "studyType": "INTERVENTIONAL",
                        "phases": ["NA"],
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {
                            "name": "University of Wisconsin, Madison",
                        },
                    },
                }
            }
        ]
    }


@pytest.mark.asyncio
async def test_clinicaltrials_connector_searches_and_normalizes_studies() -> None:
    """ClinicalTrialsConnector normalizes API v2 studies and prefers brief titles."""
    mock_client = _clinicaltrials_client(_clinicaltrials_study_payload())

    with patch("ingestion.clinicaltrials.httpx.AsyncClient", return_value=mock_client):
        documents = await ClinicalTrialsConnector().search("diabetes", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Free Test Strips and Blood Glucose Control"
    assert "free test strips" in document.text
    assert document.source == "https://clinicaltrials.gov/study/NCT00205335"
    assert document.metadata["source_type"] == "clinicaltrials"
    assert document.metadata["nct_id"] == "NCT00205335"
    assert document.metadata["year"] == "2004"
    assert document.metadata["overall_status"] == "COMPLETED"
    assert document.metadata["conditions"] == "Diabetes"
    assert document.metadata["study_type"] == "INTERVENTIONAL"
    assert document.metadata["phases"] == "NA"
    assert document.metadata["lead_sponsor"] == "University of Wisconsin, Madison"
    mock_client.get.assert_awaited_once()
    call_kwargs = mock_client.get.await_args.kwargs
    assert call_kwargs["params"]["query.term"] == "diabetes"
    assert call_kwargs["params"]["pageSize"] == 3
    assert call_kwargs["params"]["format"] == "json"


@pytest.mark.asyncio
async def test_clinicaltrials_connector_builds_descriptor_without_summary() -> None:
    """A study without a brief summary falls back to a status/conditions descriptor."""
    payload: dict[str, object] = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00000001",
                        "briefTitle": "Example Trial Without Summary",
                    },
                    "statusModule": {
                        "overallStatus": "RECRUITING",
                        "startDateStruct": {"date": "2022-06"},
                    },
                    "conditionsModule": {"conditions": ["Asthma", "Allergy"]},
                    "designModule": {
                        "studyType": "INTERVENTIONAL",
                        "phases": ["PHASE2", "PHASE3"],
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {"name": "Example Sponsor"},
                    },
                }
            }
        ]
    }
    with patch(
        "ingestion.clinicaltrials.httpx.AsyncClient",
        return_value=_clinicaltrials_client(payload),
    ):
        documents = await ClinicalTrialsConnector().search("asthma", max_results=1)

    assert len(documents) == 1
    document = documents[0]
    assert document.text == (
        "Status: RECRUITING Conditions: Asthma, Allergy Type: INTERVENTIONAL "
        "Phases: PHASE2, PHASE3 Sponsor: Example Sponsor (2022)"
    )
    assert document.metadata["nct_id"] == "NCT00000001"
    assert document.metadata["year"] == "2022"


@pytest.mark.asyncio
async def test_clinicaltrials_connector_skips_studies_without_title_or_nct() -> None:
    """Studies missing a title or NCT ID are skipped, not crashed on."""
    payload: dict[str, object] = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT99999999",
                    },
                    "descriptionModule": {"briefSummary": "No title here."},
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {
                        "briefTitle": "Missing NCT identifier",
                    },
                }
            },
        ]
    }
    with patch(
        "ingestion.clinicaltrials.httpx.AsyncClient",
        return_value=_clinicaltrials_client(payload),
    ):
        documents = await ClinicalTrialsConnector().search("anything", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_clinicaltrials_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.clinicaltrials.httpx.AsyncClient", return_value=mock_client):
        assert await ClinicalTrialsConnector().search("   ", max_results=5) == []
        assert await ClinicalTrialsConnector().search("q", max_results=0) == []

    mock_client.get.assert_not_called()


def _crossref_funder_client(payload: object, *, status_code: int = 200) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a Crossref Funders payload.

    Args:
        payload: Decoded Crossref Funders response body.
        status_code: HTTP status returned by the mocked GET.

    Returns:
        An ``AsyncMock`` usable as an ``httpx.AsyncClient`` context manager.
    """
    request = httpx.Request("GET", "https://api.crossref.org/funders")
    response = httpx.Response(status_code, json=payload, request=request)
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _crossref_funder_list_payload() -> dict[str, object]:
    """Return a representative Crossref funder-list payload for tests."""
    return {
        "status": "ok",
        "message-type": "funder-list",
        "message": {
            "items-per-page": 2,
            "total-results": 2,
            "items": [
                {
                    "id": "100000001",
                    "location": "United States",
                    "name": "National Science Foundation",
                    "alt-names": ["NSF", "U.S. National Science Foundation"],
                    "uri": "https://doi.org/10.13039/100000001",
                    "replaces": [],
                    "replaced-by": [],
                },
                {
                    "id": "501100000780",
                    "location": "United Kingdom",
                    "name": "Engineering and Physical Sciences Research Council",
                    "alt-names": ["EPSRC"],
                    "uri": "https://doi.org/10.13039/501100000780",
                },
            ],
        },
    }


@pytest.mark.asyncio
async def test_crossref_funder_connector_searches_and_normalizes() -> None:
    """CrossrefFunderConnector normalizes funder-list items into documents."""
    mock_client = _crossref_funder_client(_crossref_funder_list_payload())

    with patch("ingestion.crossref_funder.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefFunderConnector(mailto="dev@example.org").search(
            "national science",
            max_results=5,
        )

    assert len(documents) == 2
    first = documents[0]
    assert first.title == "National Science Foundation"
    assert first.source == "https://doi.org/10.13039/100000001"
    assert first.metadata["source_type"] == "crossref_funder"
    assert first.metadata["funder_id"] == "100000001"
    assert first.metadata["location"] == "United States"
    assert first.metadata["alt_names"] == "NSF, U.S. National Science Foundation"
    assert "Funding organization: National Science Foundation" in first.text
    assert "Location: United States" in first.text
    assert "Also known as: NSF" in first.text
    call = mock_client.get.call_args
    assert call.args[0] == "https://api.crossref.org/funders"
    assert call.kwargs["params"]["query"] == "national science"
    assert call.kwargs["params"]["rows"] == 5
    assert call.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_crossref_funder_connector_resolves_funder_id() -> None:
    """Bare and DOI-shaped funder ids resolve via the single-funder endpoint."""
    payload: dict[str, object] = {
        "status": "ok",
        "message-type": "funder",
        "message": {
            "id": "100000001",
            "name": "National Science Foundation",
            "location": "United States",
            "alt-names": ["NSF"],
            "uri": "https://doi.org/10.13039/100000001",
            "work-count": 453732,
            "descendant-work-count": 558111,
        },
    }
    mock_client = _crossref_funder_client(payload)

    with patch("ingestion.crossref_funder.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefFunderConnector().search("100000001", max_results=3)

    assert len(documents) == 1
    document = documents[0]
    assert document.metadata["work_count"] == "453732"
    assert document.metadata["descendant_work_count"] == "558111"
    assert "Registered works: 453732" in document.text
    assert "Descendant works: 558111" in document.text
    assert mock_client.get.call_args.args[0] == "https://api.crossref.org/funders/100000001"

    mock_client = _crossref_funder_client(payload)
    with patch("ingestion.crossref_funder.httpx.AsyncClient", return_value=mock_client):
        doi_docs = await CrossrefFunderConnector().search(
            "https://doi.org/10.13039/100000001",
            max_results=1,
        )
    assert len(doi_docs) == 1
    assert mock_client.get.call_args.args[0] == "https://api.crossref.org/funders/100000001"


@pytest.mark.asyncio
async def test_crossref_funder_connector_skips_nameless_and_caps_rows() -> None:
    """Funders without a name are skipped and max_results caps the list."""
    payload: dict[str, object] = {
        "message": {
            "items": [
                {"id": "1", "name": ""},
                {"id": "2", "name": "Alpha Fund", "uri": "https://doi.org/10.13039/2"},
                {"id": "3", "name": "Beta Fund", "uri": "https://doi.org/10.13039/3"},
            ]
        }
    }
    mock_client = _crossref_funder_client(payload)

    with patch("ingestion.crossref_funder.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefFunderConnector().search("fund", max_results=1)

    assert len(documents) == 1
    assert documents[0].title == "Alpha Fund"


@pytest.mark.asyncio
async def test_crossref_funder_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.crossref_funder.httpx.AsyncClient", return_value=mock_client):
        assert await CrossrefFunderConnector().search("   ", max_results=5) == []
        assert await CrossrefFunderConnector().search("NSF", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_crossref_funder_connector_handles_failed_lookup() -> None:
    """An unavailable Crossref Funders response yields an empty list."""
    mock_client = _crossref_funder_client({}, status_code=503)

    with patch("ingestion.crossref_funder.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefFunderConnector().search("national science", max_results=3)

    assert documents == []


_PMC_OA_TGZ_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OA>
  <responseDate>2026-07-30 12:00:00</responseDate>
  <request id="PMC13900">oa.fcgi?id=PMC13900</request>
  <records returned-count="1" total-count="1">
    <record
      id="PMC13900"
      citation="Breast Cancer Res. 2001 Nov 2; 3(1):55-60"
      license="none"
      retracted="no"
    >
      <link
        format="tgz"
        updated="2025-06-04 14:25:31"
        href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/08/e0/PMC13900.tar.gz"
      />
    </record>
  </records>
</OA>
"""

_PMC_OA_PDF_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OA>
  <responseDate>2026-07-30 12:00:00</responseDate>
  <request id="PMC5334499">oa.fcgi?id=PMC5334499</request>
  <records returned-count="1" total-count="1">
    <record
      id="PMC5334499"
      citation="World J Radiol. 2017 Feb 28; 9(2):27-33"
      license="CC BY-NC"
      retracted="no"
    >
      <link
        format="tgz"
        updated="2021-12-16 16:16:38"
        href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/8e/71/PMC5334499.tar.gz"
      />
      <link
        format="pdf"
        updated="2017-03-03 06:05:17"
        href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/8e/71/WJR-9-27.PMC5334499.pdf"
      />
    </record>
  </records>
</OA>
"""

_PMC_OA_ERROR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OA>
  <responseDate>2026-07-30 12:00:00</responseDate>
  <request>oa.fcgi?id=PMC999999999</request>
  <error code="idDoesNotExist">identifier 'PMC999999999' does not exist</error>
</OA>
"""


def _pmc_oa_client(responses: list[httpx.Response]) -> AsyncMock:
    """Build a mock AsyncClient that returns OA XML responses in order."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=responses)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_pmc_oa_connector_resolves_package_links_for_pmcid() -> None:
    """PmcOaPackageConnector resolves a PMCID to HTTPS package metadata."""
    response = httpx.Response(
        200,
        text=_PMC_OA_TGZ_XML,
        request=httpx.Request("GET", "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"),
    )
    mock_client = _pmc_oa_client([response])

    with patch("ingestion.pmc_oa.httpx.AsyncClient", return_value=mock_client):
        documents = await PmcOaPackageConnector(email="dev@example.org").search(
            "PMC13900",
            max_results=3,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.metadata["source_type"] == "pmc_oa"
    assert document.metadata["pmcid"] == "PMC13900"
    assert document.metadata["year"] == "2001"
    assert document.metadata["retracted"] == "no"
    assert document.metadata["package_url"].endswith("/pub/pmc/oa_package/08/e0/PMC13900.tar.gz")
    assert document.metadata["package_url"].startswith("https://")
    assert document.metadata["pdf_url"] == ""
    assert document.metadata["formats"] == "tgz"
    assert "PMC13900.tar.gz" in document.source
    assert "Breast Cancer Res." in document.title

    call = mock_client.get.await_args_list[0]
    assert call.args[0] == "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    assert call.kwargs["params"]["id"] == "PMC13900"
    assert call.kwargs["params"]["email"] == "dev@example.org"


@pytest.mark.asyncio
async def test_pmc_oa_connector_prefers_pdf_and_extracts_unique_pmcids() -> None:
    """PDF links are preferred as source; duplicate PMCIDs are looked up once."""
    first = httpx.Response(
        200,
        text=_PMC_OA_PDF_XML,
        request=httpx.Request("GET", "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"),
    )
    second = httpx.Response(
        200,
        text=_PMC_OA_TGZ_XML,
        request=httpx.Request("GET", "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"),
    )
    mock_client = _pmc_oa_client([first, second])

    with patch("ingestion.pmc_oa.httpx.AsyncClient", return_value=mock_client):
        documents = await PmcOaPackageConnector().search(
            "See PMC5334499 and https://pmc.ncbi.nlm.nih.gov/articles/PMC5334499/ plus pmc:13900",
            max_results=5,
        )

    assert len(documents) == 2
    assert documents[0].metadata["pmcid"] == "PMC5334499"
    assert documents[0].metadata["pdf_url"].endswith("WJR-9-27.PMC5334499.pdf")
    assert documents[0].metadata["package_url"].endswith("PMC5334499.tar.gz")
    assert documents[0].metadata["formats"] == "pdf,tgz"
    assert documents[0].metadata["license"] == "CC BY-NC"
    assert documents[0].source.endswith("WJR-9-27.PMC5334499.pdf")
    assert documents[1].metadata["pmcid"] == "PMC13900"
    assert mock_client.get.await_count == 2


@pytest.mark.asyncio
async def test_pmc_oa_connector_rejects_blank_non_positive_and_non_pmcid() -> None:
    """Blank, non-positive, and non-PMCID queries short-circuit with no HTTP."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.pmc_oa.httpx.AsyncClient", return_value=mock_client):
        assert await PmcOaPackageConnector().search("   ", max_results=5) == []
        assert await PmcOaPackageConnector().search("PMC13900", max_results=0) == []
        assert (
            await PmcOaPackageConnector().search("open access breast cancer", max_results=5) == []
        )

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_pmc_oa_connector_skips_missing_and_failed_lookups() -> None:
    """Missing PMCIDs and HTTP failures are skipped so one miss does not fail a batch."""
    missing = httpx.Response(
        200,
        text=_PMC_OA_ERROR_XML,
        request=httpx.Request("GET", "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"),
    )
    failing = httpx.Response(
        503,
        text="unavailable",
        request=httpx.Request("GET", "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"),
    )
    succeeding = httpx.Response(
        200,
        text=_PMC_OA_TGZ_XML,
        request=httpx.Request("GET", "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"),
    )
    mock_client = _pmc_oa_client([missing, failing, succeeding])

    with patch("ingestion.pmc_oa.httpx.AsyncClient", return_value=mock_client):
        documents = await PmcOaPackageConnector().search(
            "PMC999999999 PMC888888888 PMC13900",
            max_results=5,
        )

    assert len(documents) == 1
    assert documents[0].metadata["pmcid"] == "PMC13900"


def _openalex_authors_client(
    payloads: list[tuple[str, object]],
    *,
    status_code: int = 200,
) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning OpenAlex author payloads."""
    mock_client = AsyncMock()

    def _response_for(url: str) -> httpx.Response:
        for target_url, payload in payloads:
            if url == target_url:
                if status_code >= 400:
                    return httpx.Response(
                        status_code,
                        request=httpx.Request("GET", url),
                    )
                return httpx.Response(
                    status_code,
                    json=payload,
                    request=httpx.Request("GET", url),
                )
        return httpx.Response(
            status_code,
            json={},
            request=httpx.Request("GET", url),
        )

    async def _get(url: str, params: object = None) -> httpx.Response:
        return _response_for(url)

    mock_client.get.side_effect = _get
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _openalex_authors_search_payload() -> dict[str, object]:
    """Return a representative OpenAlex authors search payload."""
    return {
        "results": [
            {
                "id": "https://openalex.org/A2208157607",
                "display_name": "Geoffrey E. Hinton",
                "orcid": "https://orcid.org/0000-0003-0660-5270",
                "works_count": 312,
                "cited_by_count": 498231,
                "last_known_institutions": [
                    {"display_name": "University of Toronto"},
                ],
                "summary_stats": {"h_index": 142, "i10_index": 287},
            },
            {"display_name": ""},
        ]
    }


@pytest.mark.asyncio
async def test_openalex_authors_connector_searches_and_normalizes_authors() -> None:
    """OpenAlexAuthorsConnector searches the authors API for free-text queries."""
    mock_client = _openalex_authors_client(
        [("https://api.openalex.org/authors", _openalex_authors_search_payload())]
    )

    with patch("ingestion.openalex_authors.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexAuthorsConnector(mailto="dev@example.org").search(
            "Geoffrey Hinton",
            max_results=5,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Geoffrey E. Hinton"
    assert "University of Toronto" in document.text
    assert document.source == "https://openalex.org/A2208157607"
    assert document.metadata["source_type"] == "openalex_authors"
    assert document.metadata["author_id"] == "A2208157607"
    assert document.metadata["orcid"] == "0000-0003-0660-5270"
    assert document.metadata["works_count"] == "312"
    assert document.metadata["cited_by_count"] == "498231"
    assert document.metadata["h_index"] == "142"

    call = mock_client.get.await_args_list[0]
    assert call.args[0] == "https://api.openalex.org/authors"
    assert call.kwargs["params"]["search"] == "Geoffrey Hinton"
    assert call.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_openalex_authors_connector_resolves_author_id() -> None:
    """Author-shaped queries resolve the author directly."""
    author_payload = {
        "id": "https://openalex.org/A2208157607",
        "display_name": "Geoffrey E. Hinton",
        "works_count": 312,
        "cited_by_count": 498231,
        "summary_stats": {"h_index": 142},
    }
    mock_client = _openalex_authors_client(
        [("https://api.openalex.org/authors/A2208157607", author_payload)]
    )

    with patch("ingestion.openalex_authors.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexAuthorsConnector().search("A2208157607", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["author_id"] == "A2208157607"
    assert (
        mock_client.get.await_args_list[0].args[0] == "https://api.openalex.org/authors/A2208157607"
    )


@pytest.mark.asyncio
async def test_openalex_authors_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.openalex_authors.httpx.AsyncClient", return_value=mock_client):
        assert await OpenAlexAuthorsConnector().search("   ", max_results=5) == []
        assert await OpenAlexAuthorsConnector().search("Geoffrey Hinton", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_openalex_authors_connector_handles_failed_lookup() -> None:
    """An unavailable OpenAlex response yields an empty list rather than raising."""
    mock_client = _openalex_authors_client(
        [("https://api.openalex.org/authors", {})],
        status_code=503,
    )

    with patch("ingestion.openalex_authors.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexAuthorsConnector().search("Geoffrey Hinton", max_results=3)

    assert documents == []


def _biorxiv_collections_client(payload: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a bioRxiv collection payload."""
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _biorxiv_collections_payload() -> dict[str, object]:
    """Return a representative bioRxiv details API collection payload."""
    return {
        "messages": [{"status": "ok", "count": 2}],
        "collection": [
            {
                "doi": "10.1101/2024.01.16.575895",
                "title": "Single-cell atlas of tumor microenvironment",
                "authors": "Smith, J.; Doe, A.",
                "date": "2024-01-20",
                "category": "cell biology",
                "abstract": "We profile immune cells in solid tumors.",
                "server": "biorxiv",
            },
            {
                "doi": "10.1101/2024.02.01.123456",
                "title": "Unrelated neuroscience preprint",
                "authors": "Lee, K.",
                "date": "2024-02-01",
                "category": "neuroscience",
                "abstract": "Synaptic plasticity in cortex.",
                "server": "biorxiv",
            },
            {"title": ""},
        ],
    }


@pytest.mark.asyncio
async def test_biorxiv_collections_connector_filters_recent_collection() -> None:
    """BioRxivCollectionsConnector filters the collection array by query tokens."""
    mock_client = _biorxiv_collections_client(_biorxiv_collections_payload())

    with patch("ingestion.biorxiv_collections.httpx.AsyncClient", return_value=mock_client):
        documents = await BioRxivCollectionsConnector().search(
            "tumor microenvironment",
            max_results=5,
            server="biorxiv",
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Single-cell atlas of tumor microenvironment"
    assert "immune cells" in document.text
    assert document.metadata["source_type"] == "biorxiv_collections"
    assert document.metadata["doi"] == "10.1101/2024.01.16.575895"
    assert document.metadata["category"] == "cell biology"
    assert document.metadata["year"] == "2024"
    assert document.source.endswith("10.1101/2024.01.16.575895")

    call = mock_client.get.await_args_list[0]
    assert call.args[0] == "https://api.biorxiv.org/details/biorxiv/100"


@pytest.mark.asyncio
async def test_biorxiv_collections_connector_fetches_category_collection() -> None:
    """Category-shaped queries use the category query parameter on date ranges."""
    mock_client = _biorxiv_collections_client(_biorxiv_collections_payload())

    with patch("ingestion.biorxiv_collections.httpx.AsyncClient", return_value=mock_client):
        documents = await BioRxivCollectionsConnector().search(
            "cell_biology",
            max_results=5,
        )

    assert len(documents) == 1
    assert documents[0].metadata["category"] == "cell biology"
    call = mock_client.get.await_args_list[0]
    assert call.kwargs["params"]["category"] == "cell_biology"


@pytest.mark.asyncio
async def test_biorxiv_collections_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.biorxiv_collections.httpx.AsyncClient", return_value=mock_client):
        assert await BioRxivCollectionsConnector().search("   ", max_results=5) == []
        assert await BioRxivCollectionsConnector().search("cell biology", max_results=0) == []

    mock_client.get.assert_not_called()


def _wikidata_scholarly_client(
    search_payload: object | None = None,
    entities_payload: object | None = None,
    sparql_payload: object | None = None,
    *,
    status_code: int = 200,
) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning Wikidata API payloads."""
    mock_client = AsyncMock()

    async def _get(url: str, params: object = None, headers: object = None) -> httpx.Response:
        if status_code >= 400:
            return httpx.Response(status_code, request=httpx.Request("GET", url))
        if url == WIKIDATA_SCHOLARLY_SPARQL_URL:
            return httpx.Response(
                status_code,
                json=sparql_payload or {},
                request=httpx.Request("GET", url),
            )
        action = ""
        if isinstance(params, dict):
            action = str(params.get("action", ""))
        if action == "wbgetentities":
            payload = entities_payload or {}
        elif action == "wbsearchentities":
            payload = search_payload or {}
        else:
            payload = entities_payload or search_payload or {}
        return httpx.Response(
            status_code,
            json=payload,
            request=httpx.Request("GET", url),
        )

    mock_client.get.side_effect = _get
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _wikidata_search_payload() -> dict[str, object]:
    """Return a representative wbsearchentities payload."""
    return {
        "search": [
            {"id": "Q210272", "label": "Attention Is All You Need", "description": "2017 paper"},
            {"id": "Q42", "label": "Douglas Adams", "description": "Author"},
        ]
    }


def _wikidata_entities_payload() -> dict[str, object]:
    """Return a representative wbgetentities payload."""
    return {
        "entities": {
            "Q210272": {
                "labels": {"en": {"value": "Attention Is All You Need"}},
                "descriptions": {"en": {"value": "2017 transformer architecture paper"}},
                "claims": {
                    "P356": [
                        {
                            "mainsnak": {
                                "datavalue": {"value": "10.48550/arXiv.1706.03762"},
                            }
                        }
                    ],
                    "P31": [
                        {
                            "mainsnak": {
                                "datavalue": {"value": {"id": "Q13442814"}},
                            }
                        }
                    ],
                    "P577": [
                        {
                            "mainsnak": {
                                "datavalue": {
                                    "value": {"time": "+2017-06-12T00:00:00Z"},
                                }
                            }
                        }
                    ],
                },
                "sitelinks": {"enwiki": {"title": "Attention Is All You Need"}},
            }
        }
    }


@pytest.mark.asyncio
async def test_wikidata_scholarly_connector_searches_and_normalizes_entities() -> None:
    """WikidataScholarlyConnector searches entities and enriches with wbgetentities."""
    mock_client = _wikidata_scholarly_client(
        search_payload=_wikidata_search_payload(),
        entities_payload=_wikidata_entities_payload(),
    )

    with patch("ingestion.wikidata_scholarly.httpx.AsyncClient", return_value=mock_client):
        documents = await WikidataScholarlyConnector().search(
            "attention is all you need",
            max_results=5,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Attention Is All You Need"
    assert "transformer architecture" in document.text
    assert document.source == "https://www.wikidata.org/wiki/Q210272"
    assert document.metadata["source_type"] == "wikidata_scholarly"
    assert document.metadata["wikidata_id"] == "Q210272"
    assert document.metadata["doi"] == "10.48550/arXiv.1706.03762"
    assert document.metadata["year"] == "2017"
    assert document.metadata["scholarly"] == "true"
    assert document.metadata["wikipedia_title"] == "Attention Is All You Need"

    search_call = mock_client.get.await_args_list[0]
    assert search_call.kwargs["params"]["action"] == "wbsearchentities"
    entities_call = mock_client.get.await_args_list[1]
    assert entities_call.kwargs["params"]["action"] == "wbgetentities"


@pytest.mark.asyncio
async def test_wikidata_scholarly_connector_resolves_qid_directly() -> None:
    """QID-shaped queries fetch the entity directly via wbgetentities."""
    mock_client = _wikidata_scholarly_client(entities_payload=_wikidata_entities_payload())

    with patch("ingestion.wikidata_scholarly.httpx.AsyncClient", return_value=mock_client):
        documents = await WikidataScholarlyConnector().search("Q210272", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["wikidata_id"] == "Q210272"
    call = mock_client.get.await_args_list[0]
    assert call.kwargs["params"]["action"] == "wbgetentities"
    assert call.kwargs["params"]["ids"] == "Q210272"


@pytest.mark.asyncio
async def test_wikidata_scholarly_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.wikidata_scholarly.httpx.AsyncClient", return_value=mock_client):
        assert await WikidataScholarlyConnector().search("   ", max_results=5) == []
        assert await WikidataScholarlyConnector().search("transformer", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_wikidata_scholarly_connector_falls_back_to_sparql() -> None:
    """When entity search returns no ids, the connector falls back to scholarly SPARQL."""
    sparql_payload = {
        "results": {
            "bindings": [
                {
                    "item": {"value": "https://www.wikidata.org/entity/Q210272"},
                    "itemLabel": {"value": "Attention Is All You Need"},
                    "itemDescription": {"value": "2017 paper on transformers"},
                    "doi": {"value": "10.48550/arXiv.1706.03762"},
                }
            ]
        }
    }
    mock_client = _wikidata_scholarly_client(
        search_payload={"search": []},
        sparql_payload=sparql_payload,
    )

    with patch("ingestion.wikidata_scholarly.httpx.AsyncClient", return_value=mock_client):
        documents = await WikidataScholarlyConnector().search("transformer paper", max_results=3)

    assert len(documents) == 1
    assert documents[0].metadata["wikidata_id"] == "Q210272"
    assert documents[0].metadata["doi"] == "10.48550/arXiv.1706.03762"
    sparql_call = mock_client.get.await_args_list[1]
    assert sparql_call.args[0] == "https://query-scholarly.wikidata.org/sparql"


def _openalex_concepts_client(
    payloads: list[tuple[str, object]],
    *,
    status_code: int = 200,
) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning OpenAlex concept payloads."""
    mock_client = AsyncMock()

    def _response_for(url: str) -> httpx.Response:
        for target_url, payload in payloads:
            if url == target_url:
                if status_code >= 400:
                    return httpx.Response(
                        status_code,
                        request=httpx.Request("GET", url),
                    )
                return httpx.Response(
                    status_code,
                    json=payload,
                    request=httpx.Request("GET", url),
                )
        return httpx.Response(
            status_code,
            json={},
            request=httpx.Request("GET", url),
        )

    async def _get(url: str, params: object = None) -> httpx.Response:
        return _response_for(url)

    mock_client.get.side_effect = _get
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _openalex_concepts_search_payload() -> dict[str, object]:
    """Return a representative OpenAlex concepts search payload."""
    return {
        "results": [
            {
                "id": "https://openalex.org/C119857082",
                "display_name": "Machine learning",
                "description": ("scientific study of algorithms and statistical models"),
                "works_count": 5185484,
                "cited_by_count": 83307510,
                "level": 1,
                "wikidata": "https://www.wikidata.org/wiki/Q2539",
            },
            {
                "id": "https://openalex.org/C41008148",
                "display_name": "Computer science",
                "description": "",
                "works_count": 12000000,
                "cited_by_count": 50000000,
            },
            {"display_name": ""},
        ]
    }


@pytest.mark.asyncio
async def test_openalex_concepts_connector_searches_and_normalizes_concepts() -> None:
    """OpenAlexConceptsConnector searches the concepts API for free-text queries."""
    mock_client = _openalex_concepts_client(
        [("https://api.openalex.org/concepts", _openalex_concepts_search_payload())]
    )

    with patch("ingestion.openalex_concepts.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexConceptsConnector(mailto="dev@example.org").search(
            "machine learning",
            max_results=5,
        )

    assert len(documents) == 2
    document = documents[0]
    assert document.title == "Machine learning"
    assert document.text == ("scientific study of algorithms and statistical models")
    assert document.source == "https://openalex.org/C119857082"
    assert document.metadata["source_type"] == "openalex_concepts"
    assert document.metadata["concept_id"] == "C119857082"
    assert document.metadata["works_count"] == "5185484"
    assert document.metadata["cited_by_count"] == "83307510"
    assert document.metadata["level"] == "1"
    assert document.metadata["wikidata"] == "Q2539"

    fallback = documents[1]
    assert fallback.title == "Computer science"
    assert "5185484" not in fallback.text
    assert "12000000" in fallback.text

    call = mock_client.get.await_args_list[0]
    assert call.args[0] == "https://api.openalex.org/concepts"
    assert call.kwargs["params"]["search"] == "machine learning"
    assert call.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_openalex_concepts_connector_resolves_concept_id_with_works_filter() -> None:
    """Concept-shaped queries resolve the concept and sample works via concepts.id filter."""
    concept_payload = {
        "id": "https://openalex.org/C119857082",
        "display_name": "Machine learning",
        "description": "Topic cluster for machine learning.",
        "works_count": 5185484,
        "cited_by_count": 83307510,
    }
    works_payload = {
        "results": [
            {"title": "Deep learning for tabular data"},
            {"title": "Gradient boosting at scale"},
        ]
    }
    mock_client = _openalex_concepts_client(
        [
            ("https://api.openalex.org/concepts/C119857082", concept_payload),
            ("https://api.openalex.org/works", works_payload),
        ]
    )

    with patch("ingestion.openalex_concepts.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexConceptsConnector(mailto="dev@example.org").search(
            "C119857082",
            max_results=2,
        )

    assert len(documents) == 1
    assert documents[0].metadata["concept_id"] == "C119857082"
    assert "Sample works:" in documents[0].text
    assert "Deep learning for tabular data" in documents[0].text
    works_call = mock_client.get.await_args_list[1]
    assert works_call.args[0] == "https://api.openalex.org/works"
    assert works_call.kwargs["params"]["filter"] == "concepts.id:119857082"


@pytest.mark.asyncio
async def test_openalex_concepts_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.openalex_concepts.httpx.AsyncClient", return_value=mock_client):
        assert await OpenAlexConceptsConnector().search("   ", max_results=5) == []
        assert await OpenAlexConceptsConnector().search("machine learning", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_openalex_concepts_connector_handles_failed_lookup() -> None:
    """An unavailable OpenAlex response yields an empty list rather than raising."""
    mock_client = _openalex_concepts_client(
        [("https://api.openalex.org/concepts", {})],
        status_code=503,
    )

    with patch("ingestion.openalex_concepts.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexConceptsConnector().search("machine learning", max_results=3)

    assert documents == []


def _ssrn_client(payload: object, *, status_code: int = 200) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a Crossref works payload."""
    request = httpx.Request("GET", "https://api.crossref.org/works")
    response = httpx.Response(status_code, json=payload, request=request)
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _ssrn_work_list_payload() -> dict[str, object]:
    """Return a representative SSRN-filtered Crossref work-list payload."""
    return {
        "status": "ok",
        "message-type": "work-list",
        "message": {
            "items": [
                {
                    "title": ["Corporate Governance Codes in Financial Communication"],
                    "DOI": "10.2139/ssrn.3537853",
                    "abstract": "<jats:p>SSRN preprint on governance reporting.</jats:p>",
                    "author": [
                        {"given": "Marc Steffen", "family": "Rapp"},
                        {"given": "Marco O.", "family": "Sperling"},
                    ],
                    "issued": {"date-parts": [[2020]]},
                    "container-title": ["SSRN Electronic Journal"],
                    "resource": {"primary": {"URL": "https://www.ssrn.com/abstract=3537853"}},
                },
                {
                    "title": ["Untitled SSRN work"],
                    "DOI": "10.2139/ssrn.9999999",
                },
            ]
        },
    }


@pytest.mark.asyncio
async def test_ssrn_connector_searches_and_normalizes() -> None:
    """SsrnConnector normalizes SSRN-filtered Crossref works into documents."""
    mock_client = _ssrn_client(_ssrn_work_list_payload())

    with patch("ingestion.ssrn.httpx.AsyncClient", return_value=mock_client):
        documents = await SsrnConnector(mailto="dev@example.org").search(
            "corporate governance",
            max_results=5,
        )

    assert len(documents) == 2
    first = documents[0]
    assert first.title == "Corporate Governance Codes in Financial Communication"
    assert first.text == "SSRN preprint on governance reporting."
    assert first.source == "https://doi.org/10.2139/ssrn.3537853"
    assert first.metadata["source_type"] == "ssrn"
    assert first.metadata["doi"] == "10.2139/ssrn.3537853"
    assert first.metadata["year"] == "2020"
    assert first.metadata["authors"] == "Marc Steffen Rapp, Marco O. Sperling"
    assert first.metadata["container"] == "SSRN Electronic Journal"
    assert first.metadata["ssrn_url"] == "https://www.ssrn.com/abstract=3537853"
    call = mock_client.get.call_args
    assert call.args[0] == "https://api.crossref.org/works"
    assert call.kwargs["params"]["query"] == "corporate governance"
    assert call.kwargs["params"]["filter"] == "prefix:10.2139"
    assert call.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_ssrn_connector_resolves_ssrn_doi() -> None:
    """SSRN DOI-shaped queries resolve via the single-work endpoint."""
    payload: dict[str, object] = {
        "status": "ok",
        "message-type": "work",
        "message": {
            "title": ["Corporate Governance Codes in Financial Communication"],
            "DOI": "10.2139/ssrn.3537853",
            "issued": {"date-parts": [[2020]]},
            "author": [{"given": "Marc Steffen", "family": "Rapp"}],
            "resource": {"primary": {"URL": "https://www.ssrn.com/abstract=3537853"}},
        },
    }
    mock_client = _ssrn_client(payload)

    with patch("ingestion.ssrn.httpx.AsyncClient", return_value=mock_client):
        documents = await SsrnConnector().search(
            "https://doi.org/10.2139/ssrn.3537853",
            max_results=1,
        )

    assert len(documents) == 1
    assert documents[0].metadata["doi"] == "10.2139/ssrn.3537853"
    call = mock_client.get.call_args
    assert call.args[0] == "https://api.crossref.org/works/10.2139/ssrn.3537853"


@pytest.mark.asyncio
async def test_ssrn_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.ssrn.httpx.AsyncClient", return_value=mock_client):
        assert await SsrnConnector().search("   ", max_results=5) == []
        assert await SsrnConnector().search("corporate governance", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_ssrn_connector_handles_failed_lookup() -> None:
    """An unavailable Crossref works response yields an empty list."""
    mock_client = _ssrn_client({}, status_code=503)

    with patch("ingestion.ssrn.httpx.AsyncClient", return_value=mock_client):
        documents = await SsrnConnector().search("corporate governance", max_results=3)

    assert documents == []


def _crossref_members_client(payload: object, *, status_code: int = 200) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a Crossref members payload."""
    request = httpx.Request("GET", "https://api.crossref.org/members")
    response = httpx.Response(status_code, json=payload, request=request)
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _crossref_members_list_payload() -> dict[str, object]:
    """Return a representative Crossref member-list payload for tests."""
    return {
        "status": "ok",
        "message-type": "member-list",
        "message": {
            "items-per-page": 2,
            "total-results": 2,
            "items": [
                {
                    "id": 78,
                    "primary-name": "Elsevier BV",
                    "location": "Amsterdam, NX, Netherlands",
                    "prefixes": ["10.1016", "10.2139"],
                    "names": ["Elsevier BV", "Elsevier"],
                    "counts": {"total-dois": 25169491},
                },
                {
                    "id": 311,
                    "primary-name": "Public Library of Science",
                    "location": "United States",
                    "prefixes": ["10.1371"],
                    "names": ["Public Library of Science", "PLOS"],
                    "counts": {"total-dois": 250000},
                },
            ],
        },
    }


@pytest.mark.asyncio
async def test_crossref_members_connector_searches_and_normalizes() -> None:
    """CrossrefMembersConnector normalizes member-list items into documents."""
    mock_client = _crossref_members_client(_crossref_members_list_payload())

    with patch("ingestion.crossref_members.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefMembersConnector(mailto="dev@example.org").search(
            "elsevier",
            max_results=5,
        )

    assert len(documents) == 2
    first = documents[0]
    assert first.title == "Elsevier BV"
    assert first.source == "https://api.crossref.org/members/78"
    assert first.metadata["source_type"] == "crossref_members"
    assert first.metadata["member_id"] == "78"
    assert first.metadata["location"] == "Amsterdam, NX, Netherlands"
    assert first.metadata["prefixes"] == "10.1016, 10.2139"
    assert first.metadata["total_dois"] == "25169491"
    assert "Crossref member: Elsevier BV" in first.text
    assert "DOI prefixes: 10.1016" in first.text
    call = mock_client.get.call_args
    assert call.args[0] == "https://api.crossref.org/members"
    assert call.kwargs["params"]["query"] == "elsevier"
    assert call.kwargs["params"]["rows"] == 5
    assert call.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_crossref_members_connector_resolves_member_id() -> None:
    """Bare member ids resolve via the single-member endpoint."""
    payload: dict[str, object] = {
        "status": "ok",
        "message-type": "member",
        "message": {
            "id": 78,
            "primary-name": "Elsevier BV",
            "location": "Amsterdam, NX, Netherlands",
            "prefixes": ["10.1016"],
            "names": ["Elsevier BV"],
            "counts": {"total-dois": 25169491},
        },
    }
    mock_client = _crossref_members_client(payload)

    with patch("ingestion.crossref_members.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefMembersConnector().search("78", max_results=3)

    assert len(documents) == 1
    assert documents[0].metadata["member_id"] == "78"
    call = mock_client.get.call_args
    assert call.args[0] == "https://api.crossref.org/members/78"


@pytest.mark.asyncio
async def test_crossref_members_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.crossref_members.httpx.AsyncClient", return_value=mock_client):
        assert await CrossrefMembersConnector().search("   ", max_results=5) == []
        assert await CrossrefMembersConnector().search("elsevier", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_crossref_members_connector_handles_failed_lookup() -> None:
    """An unavailable Crossref members response yields an empty list."""
    mock_client = _crossref_members_client({}, status_code=503)

    with patch("ingestion.crossref_members.httpx.AsyncClient", return_value=mock_client):
        documents = await CrossrefMembersConnector().search("elsevier", max_results=3)

    assert documents == []


def _openalex_institutions_client(
    payloads: list[tuple[str, object]],
    *,
    status_code: int = 200,
) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning OpenAlex institution payloads."""
    mock_client = AsyncMock()

    def _response_for(url: str) -> httpx.Response:
        for target_url, payload in payloads:
            if url == target_url:
                if status_code >= 400:
                    return httpx.Response(
                        status_code,
                        request=httpx.Request("GET", url),
                    )
                return httpx.Response(
                    status_code,
                    json=payload,
                    request=httpx.Request("GET", url),
                )
        return httpx.Response(
            status_code,
            json={},
            request=httpx.Request("GET", url),
        )

    async def _get(url: str, params: object = None) -> httpx.Response:
        return _response_for(url)

    mock_client.get.side_effect = _get
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _openalex_institutions_search_payload() -> dict[str, object]:
    """Return a representative OpenAlex institutions search payload."""
    return {
        "results": [
            {
                "id": "https://openalex.org/I136199984",
                "display_name": "Harvard University",
                "type": "education",
                "country_code": "US",
                "works_count": 707672,
                "cited_by_count": 145282587,
                "ror": "https://ror.org/03vek6s52",
                "ids": {"wikidata": "https://www.wikidata.org/wiki/Q13371"},
                "geo": {"city": "Cambridge", "country": "United States"},
                "summary_stats": {"h_index": 2785},
            },
            {
                "id": "https://openalex.org/I999999999",
                "display_name": "",
            },
        ]
    }


@pytest.mark.asyncio
async def test_openalex_institutions_connector_searches_and_normalizes() -> None:
    """OpenAlexInstitutionsConnector searches the institutions API for free-text queries."""
    mock_client = _openalex_institutions_client(
        [("https://api.openalex.org/institutions", _openalex_institutions_search_payload())]
    )

    with patch(
        "ingestion.openalex_institutions.httpx.AsyncClient",
        return_value=mock_client,
    ):
        documents = await OpenAlexInstitutionsConnector(mailto="dev@example.org").search(
            "harvard",
            max_results=5,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Harvard University"
    assert "OpenAlex institution Harvard University" in document.text
    assert "Cambridge" in document.text
    assert document.source == "https://openalex.org/I136199984"
    assert document.metadata["source_type"] == "openalex_institutions"
    assert document.metadata["institution_id"] == "I136199984"
    assert document.metadata["country_code"] == "US"
    assert document.metadata["works_count"] == "707672"
    assert document.metadata["ror"] == "03vek6s52"
    assert document.metadata["wikidata"] == "Q13371"
    assert document.metadata["h_index"] == "2785"

    call = mock_client.get.await_args_list[0]
    assert call.args[0] == "https://api.openalex.org/institutions"
    assert call.kwargs["params"]["search"] == "harvard"
    assert call.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_openalex_institutions_connector_resolves_institution_id() -> None:
    """Institution-shaped queries resolve via the single-institution endpoint."""
    institution_payload = {
        "id": "https://openalex.org/I136199984",
        "display_name": "Harvard University",
        "type": "education",
        "country_code": "US",
        "works_count": 707672,
        "cited_by_count": 145282587,
    }
    mock_client = _openalex_institutions_client(
        [("https://api.openalex.org/institutions/I136199984", institution_payload)]
    )

    with patch(
        "ingestion.openalex_institutions.httpx.AsyncClient",
        return_value=mock_client,
    ):
        documents = await OpenAlexInstitutionsConnector(mailto="dev@example.org").search(
            "I136199984",
            max_results=1,
        )

    assert len(documents) == 1
    assert documents[0].metadata["institution_id"] == "I136199984"
    call = mock_client.get.await_args_list[0]
    assert call.args[0] == "https://api.openalex.org/institutions/I136199984"


@pytest.mark.asyncio
async def test_openalex_institutions_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.openalex_institutions.httpx.AsyncClient", return_value=mock_client):
        assert await OpenAlexInstitutionsConnector().search("   ", max_results=5) == []
        assert await OpenAlexInstitutionsConnector().search("harvard", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_openalex_institutions_connector_handles_failed_lookup() -> None:
    """An unavailable OpenAlex response yields an empty list rather than raising."""
    mock_client = _openalex_institutions_client(
        [("https://api.openalex.org/institutions", {})],
        status_code=503,
    )

    with patch("ingestion.openalex_institutions.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenAlexInstitutionsConnector().search("harvard", max_results=3)

    assert documents == []


def _openaire_projects_client(payload: object, *, status_code: int = 200) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning an OpenAIRE projects payload."""
    request = httpx.Request("GET", "https://api.openaire.eu/search/projects")
    response = httpx.Response(status_code, json=payload, request=request)
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _openaire_projects_search_payload() -> dict[str, object]:
    """Return a representative OpenAIRE projects search payload."""
    return {
        "response": {
            "results": {
                "result": [
                    {
                        "metadata": {
                            "oaf:entity": {
                                "oaf:project": {
                                    "originalId": {"$": "snsf________::214458"},
                                    "code": {"$": "214458"},
                                    "title": {"$": "Quantum-inspired machine learning research"},
                                    "summary": {
                                        "$": "A fellowship on quantum-inspired machine learning."
                                    },
                                    "keywords": {"$": "Physics"},
                                    "startdate": {"$": "2023-07-01"},
                                    "enddate": {"$": "2025-10-31"},
                                    "fundedamount": {"$": 159791.0},
                                    "fundingtree": {
                                        "funder": {
                                            "name": {"$": "Swiss National Science Foundation"},
                                        }
                                    },
                                }
                            }
                        }
                    },
                    {
                        "metadata": {
                            "oaf:entity": {
                                "oaf:project": {
                                    "title": {"$": ""},
                                }
                            }
                        }
                    },
                ]
            }
        }
    }


@pytest.mark.asyncio
async def test_openaire_projects_connector_searches_and_normalizes() -> None:
    """OpenaireProjectsConnector normalizes OpenAIRE project records into documents."""
    mock_client = _openaire_projects_client(_openaire_projects_search_payload())

    with patch("ingestion.openaire_projects.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenaireProjectsConnector().search(
            "machine learning",
            max_results=5,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Quantum-inspired machine learning research"
    assert document.text == "A fellowship on quantum-inspired machine learning."
    assert document.source == "snsf________::214458"
    assert document.metadata["source_type"] == "openaire_projects"
    assert document.metadata["project_id"] == "snsf________::214458"
    assert document.metadata["code"] == "214458"
    assert document.metadata["funder"] == "Swiss National Science Foundation"
    assert document.metadata["start_date"] == "2023-07-01"
    assert document.metadata["funded_amount"] == "159791"

    call = mock_client.get.call_args
    assert call.args[0] == "https://api.openaire.eu/search/projects"
    assert call.kwargs["params"]["keywords"] == "machine learning"
    assert call.kwargs["params"]["format"] == "json"


@pytest.mark.asyncio
async def test_openaire_projects_connector_resolves_grant_id() -> None:
    """Grant-shaped queries resolve via the grantID parameter."""
    mock_client = _openaire_projects_client(_openaire_projects_search_payload())

    with patch("ingestion.openaire_projects.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenaireProjectsConnector().search("214458", max_results=1)

    assert len(documents) == 1
    call = mock_client.get.call_args
    assert call.kwargs["params"]["grantID"] == "214458"


@pytest.mark.asyncio
async def test_openaire_projects_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.openaire_projects.httpx.AsyncClient", return_value=mock_client):
        assert await OpenaireProjectsConnector().search("   ", max_results=5) == []
        assert await OpenaireProjectsConnector().search("machine learning", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_openaire_projects_connector_handles_failed_lookup() -> None:
    """An unavailable OpenAIRE projects response yields an empty list."""
    mock_client = _openaire_projects_client({}, status_code=503)

    with patch("ingestion.openaire_projects.httpx.AsyncClient", return_value=mock_client):
        documents = await OpenaireProjectsConnector().search("machine learning", max_results=3)

    assert documents == []


def _datacite_related_client(payload: dict[str, object]) -> AsyncMock:
    """Build a mocked httpx.AsyncClient returning a DataCite related search payload."""
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_datacite_related_connector_searches_and_enriches_related_identifiers() -> None:
    """DataciteRelatedConnector enriches DOI records with relatedIdentifiers."""
    payload: dict[str, object] = {
        "data": [
            {
                "id": "10.5281/zenodo.21761055",
                "type": "dois",
                "attributes": {
                    "doi": "10.5281/zenodo.21761055",
                    "titles": [{"title": "Climate Dataset"}],
                    "creators": [{"name": "Ada, A."}],
                    "descriptions": [
                        {
                            "description": "A curated climate dataset.",
                            "descriptionType": "Abstract",
                        }
                    ],
                    "publicationYear": 2026,
                    "publisher": {"name": "Zenodo"},
                    "url": "https://zenodo.org/doi/10.5281/zenodo.21761055",
                    "types": {"resourceTypeGeneral": "Dataset"},
                    "relatedIdentifiers": [
                        {
                            "relationType": "IsVersionOf",
                            "relatedIdentifierType": "DOI",
                            "relatedIdentifier": "10.5281/zenodo.21761054",
                        }
                    ],
                },
            },
            {
                "id": "10.1234/empty.title",
                "type": "dois",
                "attributes": {"titles": []},
            },
        ]
    }
    mock_client = _datacite_related_client(payload)

    with patch("ingestion.datacite_related.httpx.AsyncClient", return_value=mock_client):
        documents = await DataciteRelatedConnector().search("climate", max_results=5)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Climate Dataset"
    assert "Related identifiers:" in document.text
    assert "IsVersionOf DOI 10.5281/zenodo.21761054" in document.text
    assert document.metadata["source_type"] == "datacite_related"
    assert document.metadata["related_identifiers"] == ("IsVersionOf DOI 10.5281/zenodo.21761054")
    assert document.metadata["related_identifier_count"] == "1"

    call = mock_client.get.call_args
    assert call.kwargs["params"]["query"] == "climate"


@pytest.mark.asyncio
async def test_datacite_related_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit with no HTTP call."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.datacite_related.httpx.AsyncClient", return_value=mock_client):
        assert await DataciteRelatedConnector().search("   ", max_results=5) == []
        assert await DataciteRelatedConnector().search("climate", max_results=0) == []

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_datacite_related_connector_handles_failed_lookup() -> None:
    """An unavailable DataCite response yields an empty list."""
    response = httpx.Response(503, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ingestion.datacite_related.httpx.AsyncClient", return_value=mock_client):
        documents = await DataciteRelatedConnector().search("climate", max_results=3)

    assert documents == []


def _openalex_sources_client(
    responses: list[tuple[str, dict[str, object]]],
) -> AsyncMock:
    """Build an AsyncClient mock that routes OpenAlex sources URLs."""

    async def _get(url: str, params: dict[str, object] | None = None) -> MagicMock:
        del params
        response = MagicMock()
        for prefix, payload in responses:
            if str(url).startswith(prefix):
                response.raise_for_status = MagicMock()
                response.json = MagicMock(return_value=payload)
                return response
        response.raise_for_status = MagicMock(side_effect=httpx.HTTPError("missing"))
        return response

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False
    mock_client.get = AsyncMock(side_effect=_get)
    return mock_client


def _openalex_sources_search_payload() -> dict[str, object]:
    """Return a sample OpenAlex sources search payload."""
    return {
        "results": [
            {
                "id": "https://openalex.org/S137773608",
                "display_name": "Nature",
                "type": "journal",
                "host_organization_name": "Springer Nature",
                "issn_l": "0028-0836",
                "issn": ["0028-0836", "1476-4687"],
                "is_oa": False,
                "works_count": 500000,
                "cited_by_count": 20000000,
                "homepage_url": "https://www.nature.com",
                "summary_stats": {"h_index": 1331},
            }
        ]
    }


@pytest.mark.asyncio
async def test_openalex_sources_connector_searches_and_normalizes() -> None:
    """OpenAlexSourcesConnector searches the sources API for free-text queries."""
    mock_client = _openalex_sources_client(
        [("https://api.openalex.org/sources", _openalex_sources_search_payload())]
    )
    with patch(
        "ingestion.openalex_sources.httpx.AsyncClient",
        return_value=mock_client,
    ):
        documents = await OpenAlexSourcesConnector(mailto="dev@example.org").search(
            "Nature",
            max_results=3,
        )

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Nature"
    assert document.metadata["source_type"] == "openalex_sources"
    assert document.metadata["openalex_source_id"] == "S137773608"
    assert document.metadata["issn_l"] == "0028-0836"
    assert "OpenAlex source Nature." in document.text
    mock_client.get.assert_awaited()
    assert mock_client.get.await_args.kwargs["params"]["mailto"] == "dev@example.org"


@pytest.mark.asyncio
async def test_openalex_sources_connector_resolves_source_id() -> None:
    """Bare OpenAlex source ids resolve via the direct sources endpoint."""
    payload = _openalex_sources_search_payload()["results"][0]
    mock_client = _openalex_sources_client(
        [("https://api.openalex.org/sources/S137773608", payload)]
    )
    with patch(
        "ingestion.openalex_sources.httpx.AsyncClient",
        return_value=mock_client,
    ):
        documents = await OpenAlexSourcesConnector().search("S137773608", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["openalex_source_id"] == "S137773608"


@pytest.mark.asyncio
async def test_openalex_sources_connector_skips_sources_without_name() -> None:
    """Sources missing display_name are skipped."""
    payload = {"results": [{"id": "https://openalex.org/S1", "display_name": ""}]}
    mock_client = _openalex_sources_client([("https://api.openalex.org/sources", payload)])
    with patch(
        "ingestion.openalex_sources.httpx.AsyncClient",
        return_value=mock_client,
    ):
        documents = await OpenAlexSourcesConnector().search("x", max_results=5)

    assert documents == []


@pytest.mark.asyncio
async def test_openalex_sources_connector_rejects_blank_and_non_positive() -> None:
    """Blank queries and non-positive max_results short-circuit without HTTP."""
    mock_client = AsyncMock()
    with patch("ingestion.openalex_sources.httpx.AsyncClient", return_value=mock_client):
        assert await OpenAlexSourcesConnector().search(" ", max_results=5) == []
        assert await OpenAlexSourcesConnector().search("Nature", max_results=0) == []
    mock_client.get.assert_not_called()
