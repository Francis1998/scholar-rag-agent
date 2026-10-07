"""Assemble actual browser screenshots of the synthetic explorer, not a mock UI."""

import argparse
from pathlib import Path

from PIL import Image

from scripts.demo_corpus_explorer import FRAME_PAGES

FRAME_SIZE = (1280, 960)


def create_gif(frames_dir: Path, output_path: Path) -> None:
    """Validate full-size, distinct browser frames and preserve their actual pixels."""
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new filename.")
    frames = []
    for name in FRAME_PAGES:
        with Image.open(frames_dir / f"{name}.png") as image:
            if image.size != FRAME_SIZE or image.format != "PNG":
                raise ValueError("Expected four 1280x960 PNG browser screenshots.")
            frames.append(image.convert("RGB"))
    if len({frame.tobytes() for frame in frames}) != len(FRAME_PAGES):
        raise ValueError("Browser screenshots must show four distinct workflow states.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=3000,
        loop=0,
        disposal=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    create_gif(arguments.frames_dir, arguments.output)
    print(f"Saved actual browser-frame animation: {arguments.output}")


if __name__ == "__main__":
    main()
