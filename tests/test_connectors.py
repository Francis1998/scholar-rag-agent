"""Tests for ingestion connectors."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.arxiv import ArxivConnector
from ingestion.openalex import OpenAlexConnector
from ingestion.pdf import PDFConnector
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
