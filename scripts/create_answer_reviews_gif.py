"""Render measured review-demo output using the existing evidence illustration renderer."""

import argparse
from pathlib import Path

from scripts.create_evidence_gif import render_panels
from scripts.demo_answer_reviews import PANEL_TITLES


def create_gif(transcript_path: Path, output_path: Path) -> None:
    """Reject unrelated panels and overwrites; never invent source results or UI frames."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new filename.")
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != len(PANEL_TITLES) or any(
        panel.partition("\n")[0] != title for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected the four panels produced by demo_answer_reviews.")
    render_panels(panels, output_path, banner="SCHOLAR RAG / SAVED ANSWER REVIEWS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/answer-reviews.gif"))
    arguments = parser.parse_args()
    create_gif(arguments.transcript, arguments.output)
    print(f"Rendered {arguments.output} from {arguments.transcript}")


if __name__ == "__main__":
    main()
