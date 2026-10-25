"""Illustrate the executed screening transcript using the existing offline Pillow renderer."""

import argparse
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.create_evidence_gif import render_panels
from scripts.demo_human_screening_workbench import PANEL_TITLES


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
            panel.splitlines()[:1] != [title] or len(panel.splitlines()) < 2
            for panel, title in zip(panels, PANEL_TITLES, strict=True)
        )
    ):
        raise ValueError("Expected four bounded measured human-screening panels.")
    with TemporaryDirectory(prefix="human-screening-preview-") as temporary:
        rendered = Path(temporary) / "preview.gif"
        render_panels(panels, rendered, banner="SCHOLAR RAG / HUMAN SCREENING WORKBENCH")
        with Image.open(rendered) as image, BytesIO() as still:
            image.convert("RGB").save(still, format="PNG")
            still_bytes = still.getvalue()
        for path, data in ((output_path, rendered.read_bytes()), (still_path, still_bytes)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--still", required=True, type=Path)
    arguments = parser.parse_args()
    create_preview(arguments.transcript, arguments.output, arguments.still)
    print(f"Measured-output illustration: {arguments.output}; opt-in preview: {arguments.still}")


if __name__ == "__main__":
    main()
