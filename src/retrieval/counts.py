"""Validation shared by core retrieval result-count boundaries."""


def validate_result_count(value: int, *, name: str = "limit") -> None:
    """Require a nonnegative Python integer without coercion or an upper cap."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer (excluding booleans)")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
