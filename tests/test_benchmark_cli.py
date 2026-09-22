"""Exercise the installed retrieval benchmark command, without model calls."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _dataset() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "synthetic ranking fixture",
        "chunks": [
            {
                "chunk_id": "red",
                "document_id": "fruit",
                "title": "Fruit",
                "text": "red red apple",
                "source": "synthetic:fruit",
            },
            {
                "chunk_id": "blue",
                "document_id": "water",
                "title": "Water",
                "text": "blue blue ocean",
                "source": "synthetic:water",
            },
            {
                "chunk_id": "grey",
                "document_id": "stone",
                "title": "Stone",
                "text": "grey grey rock",
                "source": "synthetic:stone",
            },
        ],
        "cases": [
            {"case_id": "fruit", "query": "red apple", "relevant_chunk_ids": ["red", "blue"]},
            {"case_id": "water", "query": "blue ocean", "relevant_chunk_ids": ["blue"]},
        ],
    }


def _write_dataset(tmp_path: Path, data: dict[str, object] | None = None) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data if data is not None else _dataset()), encoding="utf-8")
    return path


def _run(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
        "MOONSHOT_API_KEY": "",
        "SCHOLAR_RAG_DATABASE_PATH": str(tmp_path / "must-not-exist.sqlite3"),
        "SCHOLAR_RAG_MAX_HOPS": "deliberately-invalid-unused-settings",
    }
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.evaluate_retrieval", *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_no_arguments_preserve_original_smoke_output(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "\trrf" in result.stdout
    assert not (tmp_path / "must-not-exist.sqlite3").exists()


def test_benchmark_reports_actual_per_case_and_aggregate_metrics(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    result = _run(tmp_path, "--dataset", str(dataset), "--retriever", "bm25", "--k", "1")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["dataset_name"] == "synthetic ranking fixture"
    assert len(report["dataset_sha256"]) == 64
    assert report["chunk_count"] == 3
    assert report["case_count"] == 2
    assert report["k"] == 1
    assert report["passed"] is True
    scores = report["retrievers"][0]
    assert scores["name"] == "bm25"
    assert scores["mean_hit_at_k"] == 1.0
    assert scores["mean_recall_at_k"] == 0.75
    assert scores["cases"][0] == {
        "case_id": "fruit",
        "hit_at_k": 1.0,
        "recall_at_k": 0.5,
        "retrieved_chunk_ids": ["red"],
    }
    assert "red red apple" not in result.stdout
    assert not (tmp_path / "must-not-exist.sqlite3").exists()


def test_demo_compares_both_retrievers_deterministically(tmp_path: Path) -> None:
    first = _run(tmp_path, "--demo", "--k", "1")
    second = _run(tmp_path, "--demo", "--k", "1")
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["case_count"] >= 3
    assert report["chunk_count"] > report["case_count"]
    assert [item["name"] for item in report["retrievers"]] == ["bm25", "hybrid"]
    assert all(len(item["cases"]) == report["case_count"] for item in report["retrievers"])


@pytest.mark.parametrize(
    ("threshold", "value", "expected_code"),
    [
        ("--min-recall", "0.75", 0),
        ("--min-recall", "0.750001", 1),
        ("--min-hit-rate", "1", 0),
    ],
)
def test_thresholds_use_unrounded_metrics_and_inclusive_boundary(
    tmp_path: Path, threshold: str, value: str, expected_code: int
) -> None:
    dataset = _write_dataset(tmp_path)
    result = _run(
        tmp_path, "--dataset", str(dataset), "--retriever", "bm25", "--k", "1", threshold, value
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is (expected_code == 0)
    assert report["retrievers"][0]["passed"] is (expected_code == 0)
    if expected_code:
        assert "threshold" in result.stderr.lower()


def test_failing_threshold_still_saves_complete_json_report(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    output = tmp_path / "report.json"
    result = _run(
        tmp_path,
        "--dataset",
        str(dataset),
        "--retriever",
        "bm25",
        "--k",
        "1",
        "--min-recall",
        "1",
        "--output",
        str(output),
    )
    assert result.returncode == 1
    assert output.read_text(encoding="utf-8") == result.stdout
    assert json.loads(result.stdout)["passed"] is False


def test_output_refuses_to_overwrite_input_or_an_existing_report(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    original = dataset.read_bytes()
    result = _run(tmp_path, "--dataset", str(dataset), "--output", str(dataset))
    assert result.returncode == 2
    assert "exist" in result.stderr.lower()
    assert dataset.read_bytes() == original
    assert not result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ("--demo", "--k", "0"),
        ("--demo", "--k", "101"),
        ("--demo", "--k", "1.5"),
        ("--demo", "--min-recall", "-0.01"),
        ("--demo", "--min-recall", "1.01"),
        ("--demo", "--min-recall", "nan"),
        ("--demo", "--min-hit-rate", "inf"),
        ("--demo", "--retriever", "live"),
        ("--k", "2"),
        ("--output", "report.json"),
    ],
)
def test_invalid_arguments_fail_instead_of_running_smoke(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    result = _run(tmp_path, *arguments)
    assert result.returncode == 2
    assert not result.stdout
    assert "error:" in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", "1"),
        ("name", " "),
        ("cases", []),
        ("chunks", []),
        ("unexpected", "not accepted"),
    ],
)
def test_invalid_dataset_fields_are_rejected(tmp_path: Path, field: str, value: object) -> None:
    data = _dataset()
    data[field] = value
    dataset = _write_dataset(tmp_path, data)
    result = _run(tmp_path, "--dataset", str(dataset))
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.parametrize(
    "cases",
    [
        [{"case_id": "missing", "query": "apple", "relevant_chunk_ids": ["absent"]}],
        [{"case_id": "duplicate", "query": "apple", "relevant_chunk_ids": ["red", "red"]}],
        [{"case_id": "empty", "query": "apple", "relevant_chunk_ids": []}],
        [{"case_id": "blank", "query": " ", "relevant_chunk_ids": ["red"]}],
        [
            {"case_id": "same", "query": "apple", "relevant_chunk_ids": ["red"]},
            {"case_id": "same", "query": "ocean", "relevant_chunk_ids": ["blue"]},
        ],
    ],
)
def test_invalid_labels_cannot_produce_success_shaped_reports(
    tmp_path: Path, cases: list[dict[str, object]]
) -> None:
    data = _dataset()
    data["cases"] = cases
    dataset = _write_dataset(tmp_path, data)
    result = _run(tmp_path, "--dataset", str(dataset))
    assert result.returncode == 2
    assert not result.stdout


def test_duplicate_chunk_ids_are_rejected(tmp_path: Path) -> None:
    data = _dataset()
    chunks = data["chunks"]
    assert isinstance(chunks, list)
    chunks.append(chunks[0])
    result = _run(tmp_path, "--dataset", str(_write_dataset(tmp_path, data)))
    assert result.returncode == 2
    assert "unique" in result.stderr


@pytest.mark.parametrize(
    "content",
    [
        b'{"schema_version":1,"schema_version":2}',
        b'{"schema_version":NaN}',
        b"{not json}",
        b"\xff",
        b" " * (8 * 1024 * 1024 + 1),
    ],
    ids=["duplicate-key", "nonfinite-json", "invalid-json", "invalid-utf8", "too-large"],
)
def test_malformed_or_oversized_files_fail_explicitly(tmp_path: Path, content: bytes) -> None:
    dataset = tmp_path / "bad.json"
    dataset.write_bytes(content)
    result = _run(tmp_path, "--dataset", str(dataset))
    assert result.returncode == 2
    assert not result.stdout
    assert "error:" in result.stderr


def test_missing_dataset_is_an_error(tmp_path: Path) -> None:
    result = _run(tmp_path, "--dataset", str(tmp_path / "missing.json"))
    assert result.returncode == 2
    assert not result.stdout


def test_canonical_fingerprint_tracks_data_not_json_whitespace(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path)
    first = _run(tmp_path, "--dataset", str(path), "--retriever", "bm25")
    path.write_text(json.dumps(_dataset(), indent=4, sort_keys=True), encoding="utf-8")
    second = _run(tmp_path, "--dataset", str(path), "--retriever", "bm25")
    assert json.loads(first.stdout)["dataset_sha256"] == json.loads(second.stdout)["dataset_sha256"]
    data = _dataset()
    data["name"] = "different fixture"
    path.write_text(json.dumps(data), encoding="utf-8")
    third = _run(tmp_path, "--dataset", str(path), "--retriever", "bm25")
    assert json.loads(first.stdout)["dataset_sha256"] != json.loads(third.stdout)["dataset_sha256"]
