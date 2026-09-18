"""Render a labeled illustration from an actual offline evidence-demo transcript."""

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_gif(transcript_path: Path, output_path: Path) -> None:
    """Render this demo only, without regenerating unrelated repository assets."""
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != 4:
        raise ValueError("Expected the four panels produced by demo_evidence_export.")
    title_font = ImageFont.load_default(size=28)
    body_font = ImageFont.load_default(size=21)
    small_font = ImageFont.load_default(size=16)
    frames = []
    for index, panel in enumerate(panels, start=1):
        title, *lines = panel.splitlines()
        frame = Image.new("RGB", (1120, 540), "#0f172a")
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle((24, 24, 1096, 516), radius=18, fill="#1e293b")
        draw.text((50, 44), "SCHOLAR RAG / PORTABLE EVIDENCE", font=small_font, fill="#38bdf8")
        draw.text((50, 90), title, font=title_font, fill="#e2e8f0")
        y = 154
        for line in lines:
            for wrapped in textwrap.wrap(line, width=90, break_long_words=True):
                draw.text((50, y), wrapped, font=body_font, fill="#cbd5e1")
                y += 31
        if y > 451:
            raise ValueError("Transcript does not fit the illustration; shorten or resize it.")
        draw.line((50, 456, 1068, 456), fill="#475569", width=1)
        draw.text(
            (50, 476),
            "Generated illustration from real synthetic-demo output; not a screen recording.",
            font=small_font,
            fill="#94a3b8",
        )
        draw.text((1020, 476), f"{index}/4", font=small_font, fill="#34d399")
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
