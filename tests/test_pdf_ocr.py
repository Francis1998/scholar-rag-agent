"""Tests for PdfOcrHook and OCR backends."""

from pathlib import Path

import pytest

from ingestion.pdf_ocr import NullOcrBackend, PdfOcrHook


class FakeOcrBackend:
    """Test double that returns a fixed OCR string."""

    def __init__(self, text: str = "ocr-text") -> None:
        self.text = text
        self.calls: list[str] = []

    def ocr_pdf(self, path: Path | str) -> str:
        self.calls.append(str(path))
        return self.text


def test_rejects_negative_min_chars() -> None:
    with pytest.raises(ValueError, match="min_chars"):
        PdfOcrHook(min_chars=-1)


def test_null_backend_returns_empty() -> None:
    assert NullOcrBackend().ocr_pdf("scan.pdf") == ""


def test_needs_ocr_threshold() -> None:
    hook = PdfOcrHook(min_chars=10)
    assert hook.needs_ocr("short") is True
    assert hook.needs_ocr("0123456789") is False
    assert hook.needs_ocr("   abc   ") is True
    assert hook.needs_ocr("abcdefghij") is False


def test_extract_or_ocr_keeps_sufficient_text() -> None:
    fake = FakeOcrBackend("from-ocr")
    hook = PdfOcrHook(fake, min_chars=5)
    text = hook.extract_or_ocr("paper.pdf", "enough text here")
    assert text == "enough text here"
    assert fake.calls == []


def test_extract_or_ocr_selects_ocr_when_too_short() -> None:
    fake = FakeOcrBackend("scanned body")
    hook = PdfOcrHook(fake, min_chars=40)
    text = hook.extract_or_ocr("/data/scan.pdf", "hi")
    assert text == "scanned body"
    assert fake.calls == ["/data/scan.pdf"]


def test_default_backend_is_null() -> None:
    hook = PdfOcrHook(min_chars=40)
    assert hook.extract_or_ocr("x.pdf", "") == ""


def test_whitespace_only_triggers_ocr() -> None:
    fake = FakeOcrBackend("ocr")
    hook = PdfOcrHook(fake, min_chars=1)
    assert hook.needs_ocr("   \n\t  ") is True
    assert hook.extract_or_ocr("a.pdf", "   \n\t  ") == "ocr"


def test_docstring_mentions_frontier_models() -> None:
    doc = PdfOcrHook.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
