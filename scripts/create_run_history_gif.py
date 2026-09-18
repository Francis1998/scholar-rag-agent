"""Render the actual synthetic/offline run-history demo transcript, not a UI recording."""

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_TITLES = (
    "1. Produce synthetic offline run records",
    "2. Reopen SQLite and discover forgotten IDs",
    "3. Continue by creation order, then filter",
    "4. Follow the discovered evidence links",
)


def create_gif(transcript_path: Path, output_path: Path) -> None:
    """Illustrate four actual transcript panels; reject overflow or existing artifacts."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new filename.")
    panels = transcript_path.read_text(encoding="utf-8").strip().split("\n\n")
    if len(panels) != len(_TITLES) or any(
        panel.partition("\n")[0] != title for panel, title in zip(panels, _TITLES, strict=True)
    ):
        raise ValueError("Expected the four panels produced by the run-history demo.")
    title_font = ImageFont.load_default(size=28)
    body_font = ImageFont.load_default(size=21)
    small_font = ImageFont.load_default(size=16)
    frames = []
    for index, panel in enumerate(panels, start=1):
        title, *lines = panel.splitlines()
        frame = Image.new("RGB", (1120, 600), "#0f172a")
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle((24, 24, 1096, 576), radius=18, fill="#1e293b")
        draw.text(
            (50, 44), "SCHOLAR RAG / PERSISTED RUN DISCOVERY", font=small_font, fill="#38bdf8"
        )
        draw.text((50, 90), title, font=title_font, fill="#e2e8f0")
        y = 154
        for line in lines:
            for wrapped in textwrap.wrap(line, width=90, break_long_words=True):
                draw.text((50, y), wrapped, font=body_font, fill="#cbd5e1")
                y += 31
        if y > 500:
            raise ValueError("Transcript does not fit the illustration; shorten or resize it.")
        draw.line((50, 516, 1068, 516), fill="#475569", width=1)
        draw.text(
            (50, 540),
            "Actual synthetic/offline demo output; generated illustration, not a UI recording.",
            font=small_font,
            fill="#94a3b8",
        )
        draw.text((1020, 540), f"{index}/4", font=small_font, fill="#34d399")
        frames.append(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=4000,
        loop=0,
        optimize=True,
        disposal=2,
        comment=b"Actual synthetic/offline run-history output; illustration, not a UI recording.",
    )


def main() -> None:
    """Generate only the requested run-history GIF from an inspectable transcript."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/run-history.gif"))
    arguments = parser.parse_args()
    create_gif(arguments.transcript, arguments.output)
    print(f"Rendered {arguments.output} from {arguments.transcript}")


if __name__ == "__main__":
    main()
