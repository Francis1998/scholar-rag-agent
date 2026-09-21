"""The benchmark showcase renders real, explicitly synthetic local measurements."""

from pathlib import Path

import httpx
import pytest
from PIL import Image
from scripts.create_retrieval_benchmark_gif import create_gif
from scripts.demo_retrieval_benchmark import PANEL_TITLES, run_demo

from evaluation.benchmark import BenchmarkReport, dataset_fingerprint, load_dataset


def test_demo_is_offline_measured_and_reproducibly_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*_args: object, **_kwargs: object) -> None:
        pytest.fail("The benchmark demo must not construct HTTP clients")

    monkeypatch.setattr(httpx.Client, "__init__", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", no_network)
    monkeypatch.setenv("OPENAI_API_KEY", "unused-synthetic-secret")
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid-unused-setting")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(tmp_path / "must-not-exist.sqlite3"))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "demo"
    transcript = run_demo(output)
    baseline = BenchmarkReport.model_validate_json((output / "baseline.json").read_bytes())
    failing = BenchmarkReport.model_validate_json((output / "gate-fail.json").read_bytes())
    passing = BenchmarkReport.model_validate_json((output / "gate-pass.json").read_bytes())
    dataset = load_dataset(output / "dataset.json")
    assert baseline.dataset_sha256 == dataset_fingerprint(dataset)
    assert baseline.dataset_sha256 == failing.dataset_sha256 == passing.dataset_sha256
    assert baseline.chunk_count == 6 and baseline.case_count == 4
    assert baseline.passed and not failing.passed and passing.passed
    assert "bm25: PASS\nhybrid: FAIL" in transcript
    assert "No models" in transcript
    assert not (tmp_path / "must-not-exist.sqlite3").exists()
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    for file in output.iterdir():
        assert "unused-synthetic-secret" not in file.read_text(encoding="utf-8")
    before = (output / "baseline.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    assert (output / "baseline.json").read_bytes() == before

    gif = output / "benchmark.gif"
    create_gif(output / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.size == (1120, 540)
        assert image.n_frames == 4
        assert image.info["loop"] == 0
        frames = []
        for index in range(4):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
        assert len(set(frames)) == 4


@pytest.mark.parametrize("text", ["unrelated", "One\n\nTwo\n\nThree\n\nFour"])
def test_gif_rejects_unrelated_transcripts(tmp_path: Path, text: str) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="retrieval benchmark demo"):
        create_gif(transcript, tmp_path / "not-created.gif")
    assert not (tmp_path / "not-created.gif").exists()


def test_gif_rejects_overflow_and_existing_outputs(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(
        "\n\n".join([PANEL_TITLES[0] + "\n" + "\n".join(["line"] * 20), *PANEL_TITLES[1:]]),
        encoding="utf-8",
    )
    gif = tmp_path / "not-created.gif"
    with pytest.raises(ValueError, match="does not fit"):
        create_gif(transcript, gif)
    assert not gif.exists()
    transcript.write_text("\n\n".join(PANEL_TITLES), encoding="utf-8")
    create_gif(transcript, gif)
    before = gif.read_bytes()
    with pytest.raises(FileExistsError):
        create_gif(transcript, gif)
    assert gif.read_bytes() == before
