"""Render measured native-preview output as an honestly labeled, reproducible illustration."""

import argparse
from pathlib import Path

from PIL import Image

from scripts.create_evidence_gif import render_panels
from scripts.demo_retrieval_workbench import PANEL_TITLES


def create_gif(transcript_path: Path, output_path: Path) -> None:
    preview = output_path.with_suffix(".png")
    if any(path.exists() or path.is_symlink() for path in (output_path, preview)):
        raise FileExistsError("Refusing to overwrite GIF or static preview; choose new paths.")
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != 4 or any(
        panel.splitlines()[0] != title for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected the four executed retrieval workbench demo panels.")
    render_panels(
        panels, output_path, banner="SCHOLAR RAG / RETRIEVAL WORKBENCH / SYNTHETIC OFFLINE"
    )
    with Image.open(output_path) as image:
        image.convert("RGB").save(preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_gif(args.transcript, args.output)
    print(f"Saved labeled synthetic-output GIF and static preview: {args.output}")


if __name__ == "__main__":
    main()
