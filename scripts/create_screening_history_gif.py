"""Render an output illustration from the executed synthetic history transcript."""

import argparse
from pathlib import Path

from scripts.create_evidence_gif import render_panels
from scripts.demo_screening_history import PANEL_TITLES


def create_gif(transcript_path: Path, output_path: Path) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new filename.")
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != len(PANEL_TITLES) or any(
        panel.splitlines()[0] != title for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected four executed screening-history panels.")
    render_panels(
        panels, output_path, banner="SCHOLAR RAG / SCREENING HISTORY / SYNTHETIC + OFFLINE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_gif(args.transcript, args.output)
    print(f"Rendered actual-output illustration: {args.output}")


if __name__ == "__main__":
    main()
