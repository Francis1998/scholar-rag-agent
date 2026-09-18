"""Validated, immutable document selection shared by API and retrieval boundaries."""

from collections.abc import Iterable
from typing import Annotated, TypedDict

from pydantic import AfterValidator, BeforeValidator, Field, StringConstraints, TypeAdapter

MAX_DOCUMENT_IDS = 100
MAX_DOCUMENT_ID_LENGTH = 128
DocumentIdsInput = list[str] | tuple[str, ...]
DocumentId = Annotated[
    str,
    StringConstraints(
        strict=True, strip_whitespace=True, min_length=1, max_length=MAX_DOCUMENT_ID_LENGTH
    ),
]


def _require_ordered_ids(value: object) -> object:
    if not isinstance(value, (list, tuple)):
        raise ValueError("document_ids must be a list or tuple of strings.")
    return value


def _deduplicate(document_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(document_ids))


DocumentIds = Annotated[
    tuple[DocumentId, ...],
    Field(min_length=1, max_length=MAX_DOCUMENT_IDS),
    BeforeValidator(_require_ordered_ids),
    AfterValidator(_deduplicate),
]
_DOCUMENT_IDS = TypeAdapter(DocumentIds)


def normalize_document_ids(document_ids: DocumentIdsInput | None) -> tuple[str, ...] | None:
    """Detach caller-owned IDs; only None means unscoped, never an empty collection."""
    return None if document_ids is None else _DOCUMENT_IDS.validate_python(document_ids)


class ScopeArguments(TypedDict, total=False):
    """Omit the keyword entirely for legacy unscoped component signatures."""

    document_ids: tuple[str, ...]


def scope_arguments(document_ids: tuple[str, ...] | None) -> ScopeArguments:
    """Forward a requested scope without falling back if a component rejects it."""
    return {} if document_ids is None else {"document_ids": document_ids}


def documents_within_scope(
    document_ids: tuple[str, ...] | None, candidate_ids: Iterable[str]
) -> bool:
    """Check provenance without silently dropping mis-scoped candidates."""
    if document_ids is None:
        return True
    allowed = frozenset(document_ids)
    return all(document_id in allowed for document_id in candidate_ids)


def ensure_document_scope(
    document_ids: tuple[str, ...] | None, candidate_ids: Iterable[str]
) -> None:
    """Fail before generation if an explicitly scoped custom component leaks evidence."""
    if not documents_within_scope(document_ids, candidate_ids):
        raise ValueError("Retrieved evidence includes a document outside document_ids.")
