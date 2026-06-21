"""Local PDF ingestion connector."""

from pathlib import Path

from ingestion.chunking import stable_id
from retrieval.models import Document


class PDFConnector:
    """Extract text from local PDF files."""

    def load(self, path: Path | str) -> Document:
        """Return a normalized document from a PDF path."""
        pdf_path = Path(path)
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError(f"failed to parse PDF {pdf_path}: {exc}") from exc
        title = pdf_path.stem.replace("_", " ").strip() or "Untitled PDF"
        return Document(
            document_id=stable_id(str(pdf_path.resolve()), "doc"),
            title=title,
            text=text,
            source=str(pdf_path),
            metadata={"source_type": "pdf"},
        )
