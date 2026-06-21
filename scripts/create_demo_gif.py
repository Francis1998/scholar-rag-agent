"""Create the repository demo GIF asset."""

from pathlib import Path

from PIL import Image, ImageDraw

FRAMES = [
    "1. Ingest paper fixture",
    "2. Query agent",
    "3. Observe -> Decide -> Act",
    "4. Return cited answer",
]


def main() -> None:
    """Render a compact animated GIF for README/docs."""
    output_path = Path("docs/assets/demo.gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []
    for frame_text in FRAMES:
        image = Image.new("RGB", (640, 240), color=(15, 23, 42))
        draw = ImageDraw.Draw(image)
        draw.text((36, 48), "Scholar RAG Agent Demo", fill=(226, 232, 240))
        draw.text((36, 108), frame_text, fill=(125, 211, 252))
        draw.text((36, 164), "planning trace + grounded citations", fill=(203, 213, 225))
        images.append(image)
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=900, loop=0)


if __name__ == "__main__":
    main()
