"""Assemble four actual local-browser frames; reuse the existing bounded GIF renderer."""

import argparse
from pathlib import Path

from scripts.create_corpus_explorer_gif import render_browser_frames
from scripts.demo_frozen_annotation import FRAME_PAGES


def create_preview(frames_dir: Path, output_path: Path, still_path: Path) -> None:
    for path in (output_path, still_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new filename.")
    if output_path.resolve() == still_path.resolve():
        raise ValueError("GIF and still require different paths.")
    frames = [frames_dir / f"{name}.png" for name in FRAME_PAGES]
    render_browser_frames(frames, output_path)
    still_path.parent.mkdir(parents=True, exist_ok=True)
    with still_path.open("xb") as output:
        output.write(frames[0].read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--still", required=True, type=Path)
    arguments = parser.parse_args()
    create_preview(arguments.frames_dir, arguments.output, arguments.still)
    print(
        f"Actual browser screenshot sequence: {arguments.output}; static preview: {arguments.still}"
    )


if __name__ == "__main__":
    main()
