"""Tests for FigureCaptionIndexer."""

from ingestion.figure_caption_indexer import FigureCaptionIndexer


def test_extracts_figure_and_table_captions() -> None:
    text = (
        "Introduction\n"
        "Figure 1: Overview of the retrieval pipeline.\n"
        "Some body text.\n"
        "Table 2. Ablation results on the benchmark.\n"
        "Fig. 3: Attention heatmaps across layers.\n"
    )
    captions = FigureCaptionIndexer().extract(text)
    assert [(c.kind, c.number, c.caption) for c in captions] == [
        ("figure", 1, "Overview of the retrieval pipeline."),
        ("table", 2, "Ablation results on the benchmark."),
        ("figure", 3, "Attention heatmaps across layers."),
    ]
    assert captions[0].char_offset == text.index("Figure 1:")
    assert captions[1].char_offset == text.index("Table 2.")
    assert captions[2].char_offset == text.index("Fig. 3:")


def test_empty_text_returns_empty_list() -> None:
    indexer = FigureCaptionIndexer()
    assert indexer.extract("") == []
    assert indexer.extract("   \n\t  ") == []
    assert indexer.extract("No captions here, just prose.") == []


def test_extract_is_deterministic() -> None:
    text = "Figure 1: First.\nTable 1: Second.\nFigure 2: Third.\n"
    indexer = FigureCaptionIndexer()
    first = indexer.extract(text)
    second = indexer.extract(text)
    assert first == second
    assert [c.char_offset for c in first] == sorted(c.char_offset for c in first)


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = FigureCaptionIndexer.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "PaperQA" in doc or "Unstructured" in doc
