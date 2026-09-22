"""Tests for installed console scripts and package layout."""

import importlib
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hatchling.builders.wheel import WheelBuilder
from scripts.ingest_papers import main as ingest_main

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the distribution with the configured backend, without downloading tools."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path_factory.mktemp("wheel")
    (wheel,) = WheelBuilder(str(project_root)).build(directory=str(output_dir))
    return Path(wheel)


def run_from_wheel(wheel: Path, working_dir: Path, script: str) -> None:
    """Import the wheel and installed dependencies without editable import hooks."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import sys\nsys.path[:0] = sys.argv[1:]\n" + textwrap.dedent(script),
            str(wheel),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        ],
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_built_wheel_runtime_imports(built_wheel: Path, tmp_path: Path) -> None:
    """The wheel supplies configuration and its API/router consumers without src/."""
    run_from_wheel(
        built_wheel,
        tmp_path,
        """
        from pathlib import Path

        import api.dependencies
        import config
        import llm.router
        from llm.schemas import TaskType

        for module in (config, api.dependencies, llm.router):
            assert Path(module.__file__).is_relative_to(sys.argv[1]), module.__file__

        settings = config.Settings(
            database_path=Path("runtime.sqlite3"),
            default_model="fake",
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            moonshot_api_key=None,
        )
        router = llm.router.build_model_router(settings)
        assert router.route(TaskType.DEFAULT).provider_name == "fake"
        container = api.dependencies.create_container(settings)
        assert container.document_store.list_chunks() == []
        assert settings.database_path.is_file()
        """,
    )


def test_built_wheel_console_entry_points(built_wheel: Path, tmp_path: Path) -> None:
    """All console targets load from the wheel, with the API server mocked."""
    run_from_wheel(
        built_wheel,
        tmp_path,
        """
        import contextlib
        import importlib
        import io
        import json
        import os
        from importlib.metadata import distribution
        from pathlib import Path
        from unittest.mock import patch

        for key in tuple(os.environ):
            if key.startswith("SCHOLAR_RAG_") or key in {
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                "MOONSHOT_API_KEY", "SEMANTIC_SCHOLAR_API_KEY",
            }:
                del os.environ[key]
        os.environ["SCHOLAR_RAG_DEFAULT_MODEL"] = "fake"
        os.environ["SCHOLAR_RAG_DATABASE_PATH"] = "console.sqlite3"

        entry_points = {
            entry.name: entry
            for entry in distribution("scholar-rag-agent").entry_points
            if entry.group == "console_scripts"
        }
        assert set(entry_points) == {
            "scholar-rag-api", "scholar-rag-ingest", "scholar-rag-eval",
        }
        for entry in entry_points.values():
            module = importlib.import_module(entry.module)
            assert Path(module.__file__).is_relative_to(sys.argv[1]), module.__file__
            assert callable(entry.load())

        with patch("uvicorn.run") as serve:
            entry_points["scholar-rag-api"].load()()
        serve.assert_called_once_with(
            "api.main:app", host="127.0.0.1", port=8000, reload=False,
        )

        fixture = Path("paper.txt")
        fixture.write_text("GraphRAG connects retrieval and agents.", encoding="utf-8")
        sys.argv = ["scholar-rag-ingest", str(fixture)]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            entry_points["scholar-rag-ingest"].load()()
        assert json.loads(output.getvalue())["text"] == fixture.read_text(encoding="utf-8")

        sys.argv = ["scholar-rag-eval"]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            entry_points["scholar-rag-eval"].load()()
        assert "\\trrf" in output.getvalue()

        sys.argv = ["scholar-rag-eval", "--demo", "--retriever", "bm25", "--k", "1"]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            entry_points["scholar-rag-eval"].load()()
        report = json.loads(output.getvalue())
        assert report["schema_version"] == 1
        assert report["chunk_count"] == 6
        assert report["case_count"] == 4
        assert report["retrievers"][0]["name"] == "bm25"
        """,
    )


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
