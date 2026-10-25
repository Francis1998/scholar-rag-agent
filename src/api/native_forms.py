"""Strict bounded wire parsing shared by script-free native forms."""

import re
from urllib.parse import parse_qsl

from fastapi import Request


class FormTooLargeError(ValueError):
    pass


def parse_fields(
    raw: bytes, *, max_fields: int, max_field_characters: int = 16_384
) -> dict[str, str]:
    wire = raw.decode("ascii")
    if re.search(r"%(?![0-9a-fA-F]{2})", wire):
        raise ValueError("Malformed percent encoding.")
    pairs = parse_qsl(
        wire,
        keep_blank_values=True,
        strict_parsing=True,
        encoding="utf-8",
        errors="strict",
        max_num_fields=max_fields,
    )
    if len(dict(pairs)) != len(pairs) or any(
        len(name) > 32 or len(value) > max_field_characters or "\0" in name for name, value in pairs
    ):
        raise ValueError("Invalid form fields.")
    return dict(pairs)


async def read_body(request: Request, *, max_bytes: int) -> bytes:
    lengths = request.headers.getlist("content-length")
    if len(lengths) > 1 or (
        lengths and (not lengths[0].isascii() or not lengths[0].isdecimal() or len(lengths[0]) > 19)
    ):
        raise ValueError("Invalid body length.")
    if lengths and int(lengths[0]) > max_bytes:
        raise FormTooLargeError()
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > max_bytes:
            raise FormTooLargeError()
        content.extend(chunk)
    if lengths and int(lengths[0]) != len(content):
        raise ValueError("Body length mismatch.")
    return bytes(content)


def require_urlencoded(request: Request) -> None:
    content_types = request.headers.getlist("content-type")
    if (
        len(content_types) != 1
        or not re.fullmatch(
            r'application/x-www-form-urlencoded(?:\s*;\s*charset=(?:utf-8|"utf-8"))?',
            content_types[0].strip().lower(),
        )
        or "content-encoding" in request.headers
    ):
        raise ValueError("Expected UTF-8 URL-encoded form without content encoding.")
