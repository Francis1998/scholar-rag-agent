"""Illustrate the actual synthetic offline document-scope demo transcript."""

import argparse
from pathlib import Path

from scripts.create_evidence_gif import render_panels
from scripts.demo_document_scope import PANEL_TITLES


def create_gif(transcript_path: Path, output_path: Path) -> None:
    """Reuse the measured-transcript renderer without regenerating other assets."""
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != len(PANEL_TITLES) or any(
        panel.splitlines()[:1] != [title] for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected the four panels produced by the scope demo.")
    render_panels(panels, output_path, banner="SCHOLAR RAG / DOCUMENT SCOPE / SYNTHETIC + OFFLINE")


def main() -> None:
    """Render the guide GIF from previously measured demo output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/document-scope.gif"))
    arguments = parser.parse_args()
    create_gif(arguments.transcript, arguments.output)
    print(f"Rendered {arguments.output} from {arguments.transcript}")


if __name__ == "__main__":
    main()
