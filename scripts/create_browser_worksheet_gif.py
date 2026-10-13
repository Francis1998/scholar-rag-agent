"""Render an explicitly labeled illustration of executed native-form demo output."""

import argparse
from pathlib import Path

from PIL import Image

from scripts.create_evidence_gif import render_panels
from scripts.demo_browser_worksheet import PANEL_TITLES


def create_gif(transcript_path: Path, output_path: Path) -> None:
    """Reuse the measured-panel renderer, with a still preview and no fabricated screenshot."""
    preview = output_path.with_suffix(".png")
    if output_path.exists() or preview.exists():
        raise FileExistsError("Refusing to overwrite GIF or static preview; choose new paths.")
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != 4 or any(
        panel.splitlines()[0] != title for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected the four executed browser worksheet demo panels.")
    render_panels(panels, output_path, banner="SCHOLAR RAG / NATIVE WORKSHEET / SYNTHETIC OFFLINE")
    with Image.open(output_path) as image:
        image.convert("RGB").save(preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_gif(args.transcript, args.output)
    print(f"Saved labeled synthetic-output illustration and static preview: {args.output}")


if __name__ == "__main__":
    main()
