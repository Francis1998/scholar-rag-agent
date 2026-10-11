"""Assemble real browser frames of the synthetic saved-run workflow with the shared renderer."""

import argparse
from pathlib import Path

from scripts.create_corpus_explorer_gif import render_browser_frames
from scripts.demo_saved_run_browser import FRAME_PAGES


def create_gif(frames_dir: Path, output_path: Path) -> None:
    """Preserve actual screenshot pixels; do not synthesize a UI or replace an artifact."""
    render_browser_frames([frames_dir / f"{name}.png" for name in FRAME_PAGES], output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    create_gif(arguments.frames_dir, arguments.output)
    print(f"Saved actual browser-frame animation: {arguments.output}")


if __name__ == "__main__":
    main()
