"""Tests for DatasetMentionIndexer."""

from retrieval.dataset_mention import DatasetMentionIndexer


def test_empty_text_ok() -> None:
    report = DatasetMentionIndexer().index("", "")
    assert report.datasets == ()
    assert report.reasons == ("no title or abstract text",)


def test_lexicon_extracts_common_datasets() -> None:
    report = DatasetMentionIndexer().index(
        "Fine-tuning on ImageNet and CIFAR-100",
        "We also evaluate on SQuAD 2.0, GLUE, MIMIC-IV, and PubMedQA.",
    )
    assert "ImageNet" in report.datasets
    assert "CIFAR" in report.datasets
    assert "SQuAD" in report.datasets
    assert "GLUE" in report.datasets
    assert "MIMIC" in report.datasets
    assert "PubMedQA" in report.datasets
    assert all("lexicon match" in reason for reason in report.reasons)
    assert len(report.datasets) == len(report.reasons)


def test_nearby_heuristic_captures_unknown_dataset_token() -> None:
    report = DatasetMentionIndexer().index(
        "Benchmarks",
        "We introduce the FooBar dataset for retrieval.",
    )
    assert "FooBar" in report.datasets
    assert any("nearby dataset cue" in reason for reason in report.reasons)


def test_nearby_heuristic_can_be_disabled() -> None:
    report = DatasetMentionIndexer(nearby_heuristic=False).index(
        "Benchmarks",
        "We introduce the FooBar dataset for retrieval.",
    )
    assert report.datasets == ()
    assert report.reasons == ("no dataset mentions found",)


def test_dedupes_lexicon_aliases() -> None:
    report = DatasetMentionIndexer().index(
        "CIFAR and CIFAR-10 and cifar100",
        "Training uses the CIFAR dataset.",
    )
    assert report.datasets.count("CIFAR") == 1


def test_no_false_positive_on_unrelated_abstract() -> None:
    report = DatasetMentionIndexer().index(
        "Optics survey",
        "We review laser cavities and photonic crystals without benchmarks.",
    )
    assert report.datasets == ()
    assert report.reasons == ("no dataset mentions found",)


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = DatasetMentionIndexer.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "PapersWithCode" in doc
