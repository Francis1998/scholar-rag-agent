"""Create polished repository demo GIF assets."""

from pathlib import Path

from PIL import Image, ImageDraw

CANVAS_SIZE = (900, 420)
BACKGROUND = (15, 23, 42)
PANEL = (30, 41, 59)
ACCENT = (56, 189, 248)
ACCENT_GREEN = (52, 211, 153)
ACCENT_YELLOW = (250, 204, 21)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)

Story = list[tuple[str, str, list[str]]]

STORIES: dict[str, Story] = {
    "demo.gif": [
        (
            "1. Ingest papers",
            "PDF, arXiv, and Semantic Scholar sources become normalized chunks.",
            ["PDF upload", "arXiv query", "S2 metadata", "chunk IDs"],
        ),
        (
            "2. Ask a research question",
            "The query analyzer classifies factual, synthesis, comparison, or hypothesis intent.",
            ["intent=synthesis", "entities=GraphRAG", "scope<=50 docs"],
        ),
        (
            "3. Observe -> Decide -> Act",
            "The planner writes a rationale trace before retrieval tools run.",
            ["PLANNING", "RETRIEVING", "REASONING", "ANSWERING"],
        ),
        (
            "4. Return cited answer",
            "Claims are mapped back to source chunk IDs and unsupported claims are flagged.",
            ["claim -> chunk-01", "citation", "[UNGROUNDED] guard"],
        ),
    ],
    "use_cases.gif": [
        (
            "Problem: keyword search misses evidence",
            "Different papers describe the same idea with different vocabulary.",
            ["BM25 exact terms", "dense semantics", "HyDE expansion", "RRF fusion"],
        ),
        (
            "Problem: evidence is spread across papers",
            "A useful answer may require following entities through methods and findings.",
            ["spaCy entities", "relationship graph", "depth<=3", "visited set"],
        ),
        (
            "Problem: summaries are hard to trust",
            "Every claim must map back to retrieved chunks before it reaches the user.",
            ["claim check", "source chunk", "citation", "warning"],
        ),
        (
            "Result: faster grounded synthesis",
            "Researchers get audit-friendly answers for reviews, grants, and prior art.",
            ["review", "NIW evidence", "hypothesis", "comparison"],
        ),
    ],
    "planning_trace.gif": [
        (
            "Observe",
            "Classify the request and extract entities before touching retrieval tools.",
            ["intent=comparison", "entities=3", "constraints parsed"],
        ),
        (
            "Decide",
            "Decompose the question into sub-tasks with rationale persisted as JSON.",
            ["supporting evidence", "counter evidence", "contrast findings"],
        ),
        (
            "Act",
            "Execute bounded retrieval, reranking, LLM generation, and validation.",
            ["hybrid search", "GraphRAG", "rerank", "Pydantic schema"],
        ),
        (
            "Audit",
            "Replay the run from SQLite events when a reviewer asks how an answer was formed.",
            ["timestamp", "agent_id", "run_id", "payload"],
        ),
    ],
    "grounded_answer.gif": [
        (
            "Retrieved context",
            "The executor passes chunk IDs and snippets into the generation contract.",
            ["chunk-a12", "chunk-b44", "chunk-c09"],
        ),
        (
            "Validated output",
            "Pydantic schemas parse model output before the answer can be returned.",
            ["answer", "claims[]", "citations[]"],
        ),
        (
            "Grounding check",
            "Each claim is compared against retrieved chunk support.",
            ["claim terms", "chunk terms", "support IDs"],
        ),
        (
            "Conservative response",
            "Unsupported claims are flagged instead of being presented as evidence.",
            ["cited answer", "warnings[]", "[UNGROUNDED]"],
        ),
    ],
}


def _draw_card(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int) -> None:
    """Draw a panel card with a simple border."""
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=18,
        fill=PANEL,
        outline=(51, 65, 85),
        width=2,
    )


def _draw_progress(draw: ImageDraw.ImageDraw, active_index: int, total_steps: int) -> None:
    """Draw a compact progress rail."""
    start_x = 84
    y = 342
    gap = 170
    for index in range(total_steps):
        color = ACCENT_GREEN if index <= active_index else (71, 85, 105)
        draw.ellipse((start_x + index * gap, y, start_x + 22 + index * gap, y + 22), fill=color)
        if index < total_steps - 1:
            line_color = ACCENT_GREEN if index < active_index else (71, 85, 105)
            draw.line(
                (start_x + 22 + index * gap, y + 11, start_x + gap + index * gap, y + 11),
                fill=line_color,
                width=4,
            )


def _draw_badges(draw: ImageDraw.ImageDraw, badges: list[str]) -> None:
    """Draw supporting badges for each frame."""
    x = 80
    y = 238
    for badge in badges:
        badge_width = 28 + len(badge) * 8
        draw.rounded_rectangle(
            (x, y, x + badge_width, y + 34),
            radius=12,
            fill=(8, 47, 73),
            outline=(14, 116, 144),
        )
        draw.text((x + 14, y + 9), badge, fill=(186, 230, 253))
        x += badge_width + 12
        if x > 700:
            x = 80
            y += 46


def _render_frame(title: str, subtitle: str, badges: list[str], active_index: int) -> Image.Image:
    """Render one animated GIF frame."""
    image = Image.new("RGB", CANVAS_SIZE, color=BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_card(draw, 48, 42, 804, 310)
    draw.text((80, 74), "Scholar RAG Agent", fill=ACCENT)
    draw.text((80, 118), title, fill=TEXT)
    draw.text((80, 160), subtitle, fill=MUTED)
    draw.line((80, 210, 810, 210), fill=(51, 65, 85), width=2)
    _draw_badges(draw, badges)
    _draw_progress(draw, active_index=active_index, total_steps=4)
    draw.text((80, 382), "Agentic RAG for grounded scientific synthesis", fill=ACCENT_YELLOW)
    return image


def _save_story(output_path: Path, story: Story) -> None:
    """Save one story as an animated GIF."""
    frames = [
        _render_frame(title, subtitle, badges, active_index=index)
        for index, (title, subtitle, badges) in enumerate(story)
    ]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1250,
        loop=0,
        optimize=True,
    )


def main() -> None:
    """Render animated GIF assets for the README and docs."""
    output_directory = Path("docs/assets")
    output_directory.mkdir(parents=True, exist_ok=True)
    for file_name, story in STORIES.items():
        _save_story(output_directory / file_name, story)


if __name__ == "__main__":
    main()
