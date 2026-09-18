"""Render a labeled illustration from an actual offline evidence-demo transcript."""

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: str,
) -> None:
    if draw.textbbox(position, text, font=font)[2] > 1068:
        raise ValueError("Transcript text exceeds the illustration width; shorten or resize it.")
    draw.text(position, text, font=font, fill=color)


def create_gif(transcript_path: Path, output_path: Path) -> None:
    """Render this demo only, without regenerating unrelated repository assets."""
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != 4:
        raise ValueError("Expected the four panels produced by demo_evidence_export.")
    render_panels(panels, output_path, banner="SCHOLAR RAG / PORTABLE EVIDENCE")


def render_panels(panels: list[str], output_path: Path, *, banner: str) -> None:
    """Render four measured transcript panels with explicit illustration provenance."""
    if len(panels) != 4:
        raise ValueError("Expected four transcript panels.")
    title_font = ImageFont.load_default(size=28)
    body_font = ImageFont.load_default(size=21)
    small_font = ImageFont.load_default(size=16)
    frames = []
    for index, panel in enumerate(panels, start=1):
        title, *lines = panel.splitlines()
        frame = Image.new("RGB", (1120, 540), "#0f172a")
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle((24, 24, 1096, 516), radius=18, fill="#1e293b")
        _draw_text(draw, (50, 44), banner, small_font, "#38bdf8")
        _draw_text(draw, (50, 90), title, title_font, "#e2e8f0")
        y = 154
        for line in lines:
            for wrapped in textwrap.wrap(line, width=90, break_long_words=True):
                _draw_text(draw, (50, y), wrapped, body_font, "#cbd5e1")
                y += 31
        if y > 451:
            raise ValueError("Transcript does not fit the illustration; shorten or resize it.")
        draw.line((50, 456, 1068, 456), fill="#475569", width=1)
        _draw_text(
            draw,
            (50, 476),
            "Generated illustration from real synthetic-demo output; not a screen recording.",
            small_font,
            "#94a3b8",
        )
        _draw_text(draw, (1020, 476), f"{index}/4", small_font, "#34d399")
        frames.append(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=3500,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    """Build the committed GIF from inspectable API-demo output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/evidence-export.gif"))
    arguments = parser.parse_args()
    create_gif(arguments.transcript, arguments.output)
    print(f"Rendered {arguments.output} from {arguments.transcript}")


if __name__ == "__main__":
    main()
