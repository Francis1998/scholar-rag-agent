"""Exercise the documented offline demo, persistent artifacts, and focused GIF generator."""

import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_evidence_gif import create_gif
from scripts.demo_evidence_export import DEMO_TEXT, run_demo

from tests.test_evidence_export import no_live_call


def test_demo_creates_auditable_files_and_gif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("httpx.AsyncClient.send", no_live_call)
    credential_names = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "MOONSHOT_API_KEY",
    )
    for name in credential_names:
        monkeypatch.setenv(name, "synthetic-unused-environment-value")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    (tmp_path / ".env").write_text(
        "\n".join(f"{name}=synthetic-unused-dotenv-value" for name in credential_names),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "demo"
    transcript = run_demo(output_dir)
    bundle = json.loads((output_dir / "bundle.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "bundle.md").read_text(encoding="utf-8")
    assert bundle["snapshot"]["sources"][0]["chunk"]["text"] == DEMO_TEXT
    assert bundle["generation"]["provider"] == "fake"
    assert "synthetic-unused" not in (output_dir / "bundle.json").read_text(encoding="utf-8")
    assert DEMO_TEXT in markdown
    assert "Remaining corpus chunks: 0" in transcript
    assert "JSON after restart: byte-identical" in transcript
    assert "Markdown after restart: byte-identical" in transcript
    assert (output_dir / "transcript.txt").read_text(encoding="utf-8") == transcript
    before = (output_dir / "bundle.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output_dir)
    assert (output_dir / "bundle.json").read_bytes() == before

    gif = output_dir / "evidence-export.gif"
    create_gif(output_dir / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.n_frames == 4
        assert image.size == (1120, 540)
        assert image.info["loop"] == 0
        for index in range(4):
            image.seek(index)
            assert image.info["duration"] == 3500


def test_gif_rejects_unrelated_transcripts(tmp_path: Path) -> None:
    transcript = tmp_path / "unrelated.txt"
    transcript.write_text("Not the evidence demo.", encoding="utf-8")
    with pytest.raises(ValueError, match="four panels"):
        create_gif(transcript, tmp_path / "not-created.gif")
    assert not (tmp_path / "not-created.gif").exists()
