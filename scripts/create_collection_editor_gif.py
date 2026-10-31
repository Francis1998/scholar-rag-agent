"""Render measured collection-editing output, explicitly not browser footage."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.create_evidence_gif import render_panels
from scripts.demo_collection_editing import PANEL_TITLES


def create_preview(transcript_path: Path, output_path: Path, still_path: Path) -> None:
    for path in (output_path, still_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new filename.")
    if output_path.resolve() == still_path.resolve():
        raise ValueError("GIF and still require different paths.")
    with transcript_path.open(encoding="utf-8") as source:
        transcript = source.read(16_385)
    panels = transcript.strip().split("\n\n")
    if (
        len(transcript) > 16_384
        or len(panels) != 4
        or any(
            panel.splitlines()[:1] != [title]
            for panel, title in zip(panels, PANEL_TITLES, strict=True)
        )
    ):
        raise ValueError("Expected four bounded measured collection editing panels.")
    with TemporaryDirectory(prefix="collection-editor-preview-") as temporary:
        rendered = Path(temporary) / "preview.gif"
        render_panels(panels, rendered, banner="SCHOLAR RAG / NATIVE COLLECTION EDITOR")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        still_path.parent.mkdir(parents=True, exist_ok=True)
        with rendered.open("rb") as source, output_path.open("xb") as target:
            target.write(source.read())
        with Image.open(rendered) as image, still_path.open("xb") as target:
            image.convert("RGB").save(target, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--still", required=True, type=Path)
    arguments = parser.parse_args()
    create_preview(arguments.transcript, arguments.output, arguments.still)
    print(f"Measured-output illustration, not browser footage: {arguments.output}")


if __name__ == "__main__":
    main()
