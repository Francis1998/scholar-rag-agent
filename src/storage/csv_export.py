"""Shared reversible spreadsheet text encoding for local CSV downloads."""


def csv_cell(value: str | int | float | bool | None) -> str | int | float:
    # Uniform prefixes also protect whitespace/control-led formulas and preserve leading zeros.
    if isinstance(value, str):
        return "'" + value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value
