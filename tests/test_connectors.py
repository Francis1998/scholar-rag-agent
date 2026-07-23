"""Tests for ingestion connectors."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.ads import AdsConnector
from ingestion.arxiv import ArxivConnector
from ingestion.biorxiv import BioRxivConnector
from ingestion.core import CoreConnector
from ingestion.crossref import CrossrefConnector
from ingestion.datacite import DataCiteConnector
from ingestion.dblp import DblpConnector
from ingestion.doaj import DoajConnector
from ingestion.europepmc import EuropePmcConnector
from ingestion.figshare import FigshareConnector
from ingestion.hal import HalConnector
from ingestion.openaire import OpenAireConnector
from ingestion.openalex import OpenAlexConnector
from ingestion.opencitations import OpenCitationsConnector
from ingestion.orcid import OrcidConnector
from ingestion.pdf import PDFConnector
from ingestion.pubmed import PubMedConnector
from ingestion.semantic_scholar import SemanticScholarConnector
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
