"""Optional OCR fallback hook for local PDF text extraction.

``PDFConnector`` uses pypdf character extraction, which fails on scanned pages.
``PdfOcrHook`` detects too-short extracted text and delegates to an
:class:`OcrBackend`. ``NullOcrBackend`` is the default no-op (returns empty
string) so no heavy OCR dependency is required. Swap in Tesseract/PaddleOCR
adapters later. Local ingestion hook for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 pipelines (not a DOI connector).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class OcrBackend(Protocol):
    """Protocol for PDF OCR backends."""

    def ocr_pdf(self, path: Path | str) -> str:
        """Return OCR text for the PDF at ``path``."""


class NullOcrBackend:
    """No-op OCR backend that always returns an empty string."""

    def ocr_pdf(self, path: Path | str) -> str:
        """Ignore ``path`` and return empty text."""
        return ""


class PdfOcrHook:
    """Decide when pypdf text is too short and optionally run OCR.

    ``needs_ocr`` is true when stripped extracted text length is below
    ``min_chars``. ``extract_or_ocr`` returns extracted text when sufficient,
    otherwise the backend OCR result. Inputs are not mutated. Local hook for
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 ingestion pipelines
    (not a DOI connector).
    """

    def __init__(
        self,
        backend: OcrBackend | None = None,
        *,
        min_chars: int = 40,
    ) -> None:
        """Create a PDF OCR hook.

        Args:
            backend: OCR implementation (defaults to :class:`NullOcrBackend`).
            min_chars: Minimum stripped character count before OCR is skipped.

        Raises:
            ValueError: If ``min_chars`` is negative.
        """
        if min_chars < 0:
            raise ValueError("min_chars must be >= 0")
        self._backend: OcrBackend = backend if backend is not None else NullOcrBackend()
        self._min_chars = min_chars

    @property
    def min_chars(self) -> int:
        """Return the configured short-text threshold."""
        return self._min_chars

    def needs_ocr(self, text: str) -> bool:
        """Return whether ``text`` is too short and OCR should run."""
        return len(text.strip()) < self._min_chars

    def extract_or_ocr(self, path: Path | str, extracted_text: str) -> str:
        """Return ``extracted_text`` or OCR output when text is too short.

        When OCR is not needed, returns ``extracted_text`` unchanged (aside from
        callers that may strip). When OCR is needed, returns
        ``backend.ocr_pdf(path)``.
        """
        if not self.needs_ocr(extracted_text):
            return extracted_text
        return self._backend.ocr_pdf(path)
