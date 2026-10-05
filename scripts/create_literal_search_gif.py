"""Illustrate the measured synthetic literal-search transcript, not a live UI recording."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.create_evidence_gif import render_panels
from scripts.demo_literal_search import PANEL_TITLES


def create_gif(transcript_path: Path, output_path: Path) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new filename.")
    with transcript_path.open(encoding="utf-8") as transcript:
        content = transcript.read(16385)
    if len(content) > 16384:
        raise ValueError("The literal-search transcript exceeds its character limit.")
    panels = content.strip().split("\n\n")
    if len(panels) != len(PANEL_TITLES) or any(
        panel.splitlines()[:1] != [title] or len(panel.splitlines()) < 2
        for panel, title in zip(panels, PANEL_TITLES, strict=True)
    ):
        raise ValueError("Expected the four panels produced by the literal-search demo.")
    with TemporaryDirectory(prefix="scholar-literal-gif-") as temporary:
        rendered = Path(temporary) / "illustration.gif"
        render_panels(
            panels, rendered, banner="SCHOLAR RAG / LITERAL PASSAGE SEARCH / SYNTHETIC + OFFLINE"
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
