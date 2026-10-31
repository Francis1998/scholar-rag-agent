"""Execute the published routing preflight against synthetic sources without network access."""

import json
import os
import re
import socket
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from agent.evidence import EvidenceBundle
from config import Settings
from llm.providers import HTTPProviderAdapter

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/guides/MODEL_ROUTING_PREFLIGHT_GUIDE.md"
TASKS = ("reasoning", "speed", "cost", "default")
EXPECTED_ROUTES = {
    "all keyed": ("anthropic", "gemini", "kimi", "openai"),
    "fake default, all keyed": ("anthropic", "gemini", "kimi", "fake"),
    "fake default, only OpenAI keyed": ("fake",) * 4,
    "only default Gemini": ("gemini",) * 4,
    "missing default, OpenAI keyed": ("openai",) * 4,
    "only unrelated Kimi keyed": ("fake", "fake", "kimi", "fake"),
    "no keys": ("fake",) * 4,
}


@pytest.fixture(autouse=True)
def isolate_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Iterator[None]:
    """Make accidental ambient reads fail, and reject even swallowed network attempts."""
    monkeypatch.chdir(tmp_path)
    ambient = tmp_path / "ambient.sqlite3"
    sentinel = b"Do not open or modify this ambient database"
    ambient.write_bytes(sentinel)
    poison = {
        "SCHOLAR_RAG_DATABASE_PATH": str(ambient),
        "SCHOLAR_RAG_DEFAULT_MODEL": "invalid-ambient-provider",
        "SCHOLAR_RAG_MAX_HOPS": "invalid-ambient-limit",
        **dict.fromkeys(
            ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"),
            "unused-ambient-credential",
        ),
        "SEMANTIC_SCHOLAR_API_KEY": "unused-ambient-metadata-credential",
    }
    monkeypatch.setattr(os, "environ", poison)
    dotenv = tmp_path / ".env"
    dotenv_bytes = b"SCHOLAR_RAG_DEFAULT_MODEL=invalid-dotenv-provider\n"
    dotenv.write_bytes(dotenv_bytes)
    sync_network = Mock(side_effect=AssertionError("Network forbidden in published examples"))
    async_network = AsyncMock(side_effect=AssertionError("Network forbidden in published examples"))
    live_generate = AsyncMock(side_effect=AssertionError("Live generation forbidden"))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", sync_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_network)
    monkeypatch.setattr(HTTPProviderAdapter, "generate", live_generate)
    monkeypatch.setattr(socket.socket, "connect", sync_network)
    monkeypatch.setattr(socket.socket, "connect_ex", sync_network)
    monkeypatch.setattr(socket, "getaddrinfo", sync_network)
    yield
    sync_network.assert_not_called()
    async_network.assert_not_called()
    live_generate.assert_not_called()
    assert ambient.read_bytes() == sentinel
    assert dotenv.read_bytes() == dotenv_bytes
    assert not (tmp_path / "unused-preflight.sqlite3").exists()
    captured = capsys.readouterr()
    assert "unused-ambient" not in captured.out + captured.err
    assert "synthetic-not-a-credential" not in captured.out + captured.err


def run_example(index: int) -> dict[str, object]:
    blocks = re.findall(r"^```python\n(.*?)^```$", GUIDE.read_text(), re.DOTALL | re.MULTILINE)
    assert len(blocks) == 4, "Every published Python block must have an execution test"
    namespace: dict[str, object] = {}
    exec(compile(blocks[index], str(GUIDE), "exec"), namespace)  # noqa: S102
    return namespace


def test_family_and_model_example_validates_without_building_live_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_example(0)
    assert capsys.readouterr().out.splitlines() == [
        "API ID rejected as a provider family",
        "family=openai; requested_model=gpt-6.1-terra; selected_adapter=fake",
        "direct constructor keeps its own model; no generation attempted",
    ]


def test_matrix_example_reports_every_real_route_and_requested_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    namespace = run_example(1)
    assert namespace["routes"] == {
        label: dict(zip(TASKS, expected, strict=True))
        for label, expected in EXPECTED_ROUTES.items()
    }
    models = {
        "anthropic": "claude-sonnet-5-5",
        "gemini": "gemini-3.8-flash",
        "kimi": "kimi-k3.1-pro",
        "openai": "gpt-6-astra",
        "fake": None,
    }
    expected_lines = []
    for label, expected in EXPECTED_ROUTES.items():
        expected_lines.append(label)
        expected_lines.extend(
            f"  {task}: {provider}; requested_model={models[provider]}"
            for task, provider in zip(TASKS, expected, strict=True)
        )
    expected_lines.append("custom direct-router adapter: fake; generation calls: 0")
    assert capsys.readouterr().out.splitlines() == expected_lines


def test_precedence_example_preserves_existing_router_and_restores_environment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = os.environ
    run_example(2)
    assert os.environ is original
    assert capsys.readouterr().out.splitlines() == [
        "explicit > environment > dotenv > defaults",
        "_env_file=None does not disable environment",
        "old router=environment-model; rebuilt router=explicit-model",
    ]


def test_fake_example_saves_actual_frozen_evidence_and_no_credentials(tmp_path: Path) -> None:
    namespace = run_example(3)
    output = tmp_path / "routing-preflight-artifact"
    assert {path.name for path in output.iterdir()} == {"bundle.json", "bundle.md", "checks.json"}
    checks = json.loads((output / "checks.json").read_text())
    assert checks == {
        "data": "synthetic, not scientific evidence",
        "routes": dict.fromkeys(TASKS, "fake"),
        "state": "DONE",
        "provider": "fake",
        "requested_model": None,
        "captured_sources": 1,
        "live_generation_calls": 0,
        "http_network_calls": 0,
        "scientific_validation": False,
    }
    bundle = EvidenceBundle.model_validate_json((output / "bundle.json").read_bytes())
    assert bundle.generation.provider == "fake"
    assert bundle.generation.model_name is None
    assert bundle.generation.task_type.value == "reasoning"
    assert bundle.snapshot.sources[0].chunk.text == namespace["text"]
    assert bundle.run_id in (output / "bundle.md").read_text()
    assert list(tmp_path.glob("*.sqlite3")) == [tmp_path / "ambient.sqlite3"]
    settings = namespace["settings"]
    assert isinstance(settings, Settings)
    assert not settings.database_path.exists()
    for artifact in output.iterdir():
        assert b"unused-ambient" not in artifact.read_bytes()
        assert b"synthetic-not-a-credential" not in artifact.read_bytes()
        assert b"API_KEY" not in artifact.read_bytes()


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_fake_example_refuses_existing_artifact_paths(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "routing-preflight-artifact"
    if kind == "file":
        output.write_text("Keep this artifact")
    elif kind == "directory":
        output.mkdir()
        (output / "keep.txt").write_text("Keep this artifact")
    else:
        output.symlink_to(tmp_path / "not-created")
    with pytest.raises(FileExistsError):
        run_example(3)
    if kind == "file":
        assert output.read_text() == "Keep this artifact"
    elif kind == "directory":
        assert (output / "keep.txt").read_text() == "Keep this artifact"
        assert len(list(output.iterdir())) == 1
    else:
        assert output.is_symlink()
        assert not (tmp_path / "not-created").exists()
