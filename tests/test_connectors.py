"""Tests for ingestion connectors."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.arxiv import ArxivConnector
from ingestion.crossref import CrossrefConnector
from ingestion.openalex import OpenAlexConnector
from ingestion.pdf import PDFConnector
from ingestion.pubmed import PubMedConnector
from ingestion.semantic_scholar import SemanticScholarConnector

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
