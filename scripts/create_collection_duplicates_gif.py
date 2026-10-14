"""Render four measured duplicate-inspection panels with the existing Pillow renderer."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.create_evidence_gif import render_panels
from scripts.demo_collection_duplicates import PANEL_TITLES


def create_gif(transcript_path: Path, output_path: Path) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new filename.")
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != len(PANEL_TITLES) or any(
        panel.splitlines()[:1] != [title] for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected four panels from the collection-duplicates demo.")
    with TemporaryDirectory(prefix="scholar-duplicates-gif-") as temporary:
        rendered = Path(temporary) / "illustration.gif"
        render_panels(
            panels, rendered, banner="SCHOLAR RAG / DUPLICATE CANDIDATES / SYNTHETIC + OFFLINE"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as output:
            output.write(rendered.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    create_gif(arguments.transcript, arguments.output)
    print(f"Rendered {arguments.output} from {arguments.transcript}")


if __name__ == "__main__":
    main()
