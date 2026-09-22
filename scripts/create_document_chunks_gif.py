"""Illustrate the measured synthetic chunk-reader transcript, not a browser recording."""

import argparse
from pathlib import Path

from scripts.create_evidence_gif import render_panels
from scripts.demo_document_chunks import PANEL_TITLES


def create_gif(transcript_path: Path, output_path: Path) -> None:
    """Require this demo's actual panel shape and never overwrite another artifact."""
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new filename.")
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != len(PANEL_TITLES) or any(
        panel.splitlines()[:1] != [title] for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected the four panels produced by the chunk-reader demo.")
    render_panels(
        panels, output_path, banner="SCHOLAR RAG / STORED CHUNK EVIDENCE / SYNTHETIC + OFFLINE"
    )


def main() -> None:
    """Generate only the requested illustration from an inspectable saved transcript."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/document-chunks.gif"))
    arguments = parser.parse_args()
    create_gif(arguments.transcript, arguments.output)
    print(f"Rendered {arguments.output} from {arguments.transcript}")


if __name__ == "__main__":
    main()
