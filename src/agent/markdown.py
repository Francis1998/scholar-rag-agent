"""Literal Markdown blocks shared by evidence downloads."""

import re


def literal_block(text: str, language: str = "text") -> str:
    """Keep arbitrary HTML/Markdown inert, including embedded closing fences."""
    fence_length = max((len(match[0]) + 1 for match in re.finditer(r"`+", text)), default=3)
    fence = "`" * max(3, fence_length)
    newline = "" if text.endswith("\n") else "\n"
    return f"{fence}{language}\n{text}{newline}{fence}\n"
