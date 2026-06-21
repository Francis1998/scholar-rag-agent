"""Tests for installed console scripts and package layout."""

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.ingest_papers import main as ingest_main

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


def test_scripts_package_is_importable() -> None:
    """The scripts package is installed for console script entry points."""
    module = importlib.import_module("scripts.ingest_papers")
    assert callable(module.main)


def test_scholar_rag_ingest_main_prints_document(
    tmp_path: Path,
    capsys: "CaptureFixture[str]",
) -> None:
    """The ingest CLI prints normalized document metadata for a local file."""
    fixture = tmp_path / "paper.txt"
    fixture.write_text("GraphRAG connects retrieval and agents.", encoding="utf-8")
    sys.argv = ["scholar-rag-ingest", str(fixture)]
    ingest_main()
    captured = capsys.readouterr()
    assert "GraphRAG connects retrieval and agents." in captured.out
