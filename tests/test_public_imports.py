"""Public imports must not depend on modules previously collected by pytest."""

import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import pytest


def run_isolated(tmp_path: Path, script: str) -> None:
    """Use this checkout and installed dependencies, without ambient settings or hooks."""
    bootstrap = """
        import os
        import sys
        from pathlib import Path
        from urllib.parse import unquote, urlsplit

        sys.path[:0] = sys.argv[1:]
        _database_path = Path(os.environ["SCHOLAR_RAG_DATABASE_PATH"]).resolve()
        _database_connections = []

        def guard_io(event, arguments):
            if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
                raise AssertionError(f"Import checks must remain offline: {event}")
            if event == "sqlite3.connect":
                database = arguments[0]
                if isinstance(database, str) and database.startswith("file:"):
                    uri = urlsplit(database)
                    assert not uri.netloc and not uri.fragment, database
                    assert uri.query in {"mode=ro", "mode=rw"}, database
                    database = unquote(uri.path, errors="strict")
                assert Path(database).resolve() == _database_path, arguments[0]
                _database_connections.append(arguments[0])

        sys.addaudithook(guard_io)
    """
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            textwrap.dedent(bootstrap) + "\n" + textwrap.dedent(script),
            str(Path(__file__).resolve().parents[1] / "src"),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        ],
        cwd=tmp_path,
        env={
            "HOME": str(tmp_path),
            "SCHOLAR_RAG_DATABASE_PATH": str(tmp_path / "runtime.sqlite3"),
            "SCHOLAR_RAG_DEFAULT_MODEL": "fake",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_database_audit_accepts_exact_file_uris_and_rejects_other_targets(tmp_path: Path) -> None:
    run_isolated(
        tmp_path,
        """
        import sqlite3
        from contextlib import closing

        _database_path = _database_path.with_name("runtime %?# \\u7814\\u7a76.sqlite3")
        with closing(sqlite3.connect(_database_path)) as connection:
            connection.execute("CREATE TABLE marker (id INTEGER)")
        for mode in ("rw", "ro"):
            with closing(sqlite3.connect(
                _database_path.as_uri() + f"?mode={mode}", uri=True
            )) as connection:
                assert connection.execute("SELECT count(*) FROM marker").fetchone() == (0,)

        other = _database_path.with_name("not-authorized.sqlite3")
        for target, is_uri in (
            (str(other), False),
            (other.as_uri() + "?mode=rw", True),
            (_database_path.as_uri() + "?mode=rwc", True),
        ):
            try:
                sqlite3.connect(target, uri=is_uri)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"Unexpected database target was allowed: {target}")
        assert not other.exists()
        assert len(_database_connections) == 3
        """,
    )


@pytest.mark.parametrize(
    "first_import",
    [
        pytest.param("from retrieval.citations import CitationGrounder", id="citations-first"),
        pytest.param("from retrieval.hybrid import HybridRetriever", id="hybrid-first"),
        pytest.param(
            "from retrieval.claim_verification_gate import ClaimVerificationGate",
            id="claim-verification-first",
        ),
        pytest.param(
            "from retrieval.citation_groundedness_score import CitationGroundednessScorer",
            id="citation-scorer-first",
        ),
        pytest.param("from api.application import create_app", id="application-first"),
        pytest.param("from api.dependencies import create_container", id="dependencies-first"),
        pytest.param("from agent import AgentRunner", id="package-export-first"),
        pytest.param("import agent\nAgentRunner = agent.AgentRunner", id="package-attribute-first"),
        pytest.param("from agent.runner import AgentRunner", id="runner-module-first"),
    ],
)
def test_public_library_import_orders(tmp_path: Path, first_import: str) -> None:
    run_isolated(
        tmp_path,
        'os.environ["SCHOLAR_RAG_MAX_HOPS"] = "invalid-ambient-setting"\n'
        + first_import
        + "\n"
        + textwrap.dedent(
            """
            from retrieval.citations import CitationGrounder
            from retrieval.hybrid import HybridRetriever
            from retrieval.claim_verification_gate import ClaimVerificationGate
            from retrieval.citation_groundedness_score import CitationGroundednessScorer
            from retrieval.models import Chunk, SearchResult
            from agent.models import Claim
            from agent import AgentRunner
            from agent.runner import AgentRunner as DirectRunner
            from api.application import create_app
            from api.dependencies import create_container
            import agent

            assert agent.AgentRunner is AgentRunner is DirectRunner
            assert agent.__all__ == ["AgentRunner"]
            assert callable(HybridRetriever)
            assert callable(create_app) and callable(create_container)
            chunk = Chunk(
                chunk_id="c1", document_id="d1", title="Synthetic note",
                text="Graph retrieval connects passages.", source="synthetic:imports",
            )
            result = SearchResult(chunk=chunk, score=1.0, retriever="fixture")
            answer = CitationGrounder().ground(
                chunk.text, [Claim(text=chunk.text, chunk_ids=["c1"])], [chunk],
            )
            assert not answer.ungrounded
            assert [citation.chunk_id for citation in answer.citations] == ["c1"]
            assert ClaimVerificationGate().verify(chunk.text, [result]).supported_count == 1
            assert CitationGroundednessScorer().score(
                "Graph retrieval connects passages [1].", [result],
            ).grounded_count == 1
            assert "api.main" not in sys.modules
            assert _database_connections == []
            assert not _database_path.exists()
            """
        ),
    )


def test_lazy_package_preserves_exports_submodules_and_attribute_errors(tmp_path: Path) -> None:
    run_isolated(
        tmp_path,
        """
        import agent
        from agent import models
        from types import ModuleType

        assert isinstance(models, ModuleType)
        assert agent.models is models is sys.modules["agent.models"]
        assert agent.__all__ == ["AgentRunner"]
        assert "agent.runner" not in sys.modules
        assert "agent.executor" not in sys.modules

        def check_unknown_attributes():
            for name in ("MissingAgent", "__missing_attribute__"):
                try:
                    getattr(agent, name)
                except AttributeError as error:
                    assert str(error) == f"module 'agent' has no attribute {name!r}"
                else:
                    raise AssertionError(f"Unexpected package attribute: {name}")
                assert not hasattr(agent, name)
                assert getattr(agent, name, None) is None

        check_unknown_attributes()
        assert "agent.runner" not in sys.modules
        from agent import AgentRunner
        from agent import executor, runner

        assert isinstance(runner, ModuleType) and isinstance(executor, ModuleType)
        assert agent.runner is runner is sys.modules["agent.runner"]
        assert agent.executor is executor is sys.modules["agent.executor"]
        assert agent.AgentRunner is AgentRunner is runner.AgentRunner
        assert vars(agent)["AgentRunner"] is AgentRunner
        namespace = {}
        exec("from agent import *", namespace)
        assert set(namespace) == {"__builtins__", "AgentRunner"}
        assert namespace["AgentRunner"] is AgentRunner
        check_unknown_attributes()
        assert _database_connections == []
        """,
    )


@pytest.mark.parametrize(
    "first_import",
    [
        pytest.param("from retrieval.citations import CitationGrounder", id="citations-first"),
        pytest.param("from api.application import create_app", id="factory-first"),
        pytest.param("from api.main import app", id="entrypoint-first"),
    ],
)
def test_api_entrypoint_still_initializes_its_explicit_database(
    tmp_path: Path, first_import: str
) -> None:
    run_isolated(
        tmp_path,
        first_import
        + "\n"
        + textwrap.dedent(
            """
            import api.application
            import api.main
            from api.main import app
            from agent import AgentRunner
            from retrieval.citations import CitationGrounder

            assert api.main.create_app is api.application.create_app
            assert api.main.app is app
            assert isinstance(app.state.container.runner, AgentRunner)
            assert "/health" in app.openapi()["paths"]
            assert _database_connections
            assert _database_path.is_file()
            assert not Path(".scholar-rag-agent.sqlite3").exists()
            """
        ),
    )


def test_package_typing_rejects_unknown_exports_and_preserves_known_types(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        from typing import assert_type
        import agent
        from agent import AgentRunner
        from agent.runner import AgentRunner as DirectRunner

        def supported_types(
            exported: AgentRunner, qualified: agent.AgentRunner, direct: DirectRunner
        ) -> None:
            assert_type(exported, DirectRunner)
            assert_type(qualified, DirectRunner)
            assert_type(direct, DirectRunner)

        assert_type(AgentRunner.__name__, str)
        assert_type(agent.AgentRunner.__name__, str)
        assert_type(DirectRunner.__name__, str)
        agent.AgentRuner
        agent.MissingRunner
        from agent import AgentRuner
        from agent import MissingRunner
        """
    )
    run_isolated(
        tmp_path,
        f"source = {source!r}\n"
        + textwrap.dedent(
            """
            from mypy import api as mypy_api

            os.environ["MYPYPATH"] = sys.argv[1]
            output, errors, status = mypy_api.run([
                "--config-file", str(Path(sys.argv[1]).parent / "pyproject.toml"),
                "--strict", "--no-incremental", "--cache-dir", os.devnull,
                "--show-error-codes", "--no-error-summary", "--no-pretty",
                "--command", source,
            ])
            assert not errors, errors
            assert status == 1, f"Expected rejection of unknown exports; got {status}: {output}"
            assert output.count(": error:") == 4, output
            assert output.count("[attr-defined]") == 4, output
            assert output.count("AgentRuner") == 2, output
            assert output.count("MissingRunner") == 2, output
            assert _database_connections == []
            """
        ),
    )
