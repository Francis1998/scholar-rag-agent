# PDF OCR Hook Guide

![PDF OCR hook demo](../assets/pdf-ocr-hook.gif)

`PdfOcrHook` wraps local PDF text extraction with an optional OCR fallback.
`PDFConnector` uses pypdf character extraction, which returns little or no text
for scanned pages. When stripped extracted text is shorter than `min_chars`
(default `40`), the hook calls an `OcrBackend`. `NullOcrBackend` is the default
no-op (returns `""`) so no heavy OCR dependency is required — plug in
Tesseract/PaddleOCR later. Local ingestion hook for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 pipelines (not a DOI connector).

## Usage

```python
from ingestion.pdf_ocr import NullOcrBackend, PdfOcrHook

hook = PdfOcrHook(NullOcrBackend(), min_chars=40)
if hook.needs_ocr(extracted):
    text = hook.extract_or_ocr(pdf_path, extracted)
else:
    text = extracted
```

See unit tests for the short-text threshold and a `FakeOcrBackend` example.
