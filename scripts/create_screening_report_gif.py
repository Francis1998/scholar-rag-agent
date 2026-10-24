"""Illustrate executed screening-report output using the existing Pillow panel renderer."""

import argparse
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.create_evidence_gif import render_panels
from scripts.demo_screening_report import PANEL_TITLES


def create_preview(transcript_path: Path, output_path: Path, still_path: Path) -> None:
    for path in (output_path, still_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new filename.")
    if output_path.resolve() == still_path.resolve():
        raise ValueError("The GIF and static preview need distinct paths.")
    with transcript_path.open(encoding="utf-8") as transcript:
        content = transcript.read(16385)
    if len(content) > 16384:
        raise ValueError("The screening report transcript exceeds its character limit.")
    panels = content.strip().split("\n\n")
    if len(panels) != 4 or any(
        panel.splitlines()[:1] != [title] or len(panel.splitlines()) < 2
        for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected four executed screening report transcript panels.")
    with TemporaryDirectory(prefix="screening-report-preview-") as temporary:
        rendered = Path(temporary) / "preview.gif"
        render_panels(panels, rendered, banner="SCHOLAR RAG / HUMAN SCREENING PROGRESS")
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
    args = parser.parse_args()
    create_preview(args.transcript, args.output, args.still)
    print(f"Saved executed-output illustration: {args.output}; still: {args.still}")


if __name__ == "__main__":
    main()
