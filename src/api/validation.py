"""Preserve validation details without echoing unencodable diagnostic values."""

import json

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError


def sanitized_validation_content(exc: RequestValidationError) -> str | None:
    """Encode repaired diagnostics, or defer to the normal handler when all are safe."""
    detail: list[dict[str, object]] = jsonable_encoder(exc.errors())
    sanitized = False
    for index, error in enumerate(detail):
        try:
            json.dumps(error, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            detail[index] = {key: error[key] for key in ("loc", "type", "msg")}
            sanitized = True
    if not sanitized:
        return None
    # Locations can themselves contain an invalid Unicode object key.
    return json.dumps({"detail": detail}, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
