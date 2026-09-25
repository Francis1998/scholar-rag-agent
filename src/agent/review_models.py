"""Bounded human judgments, not factual verification or agent state transitions."""

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

DEFAULT_REVIEW_LIMIT = 20
MAX_REVIEW_LIMIT = 100
MAX_REVIEW_SEQUENCE = 2**63 - 1
MAX_COMMENT_CHARACTERS = 4000
MAX_REVIEW_ID_CHARACTERS = 256
MAX_CITED_CHUNKS = 100


def _nonblank_text(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError("Text must be nonblank and cannot contain NUL characters.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Text must contain valid Unicode characters.") from exc
    return value


def _identity(value: str) -> str:
    _nonblank_text(value)
    if value != value.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError("IDs cannot have outer whitespace or control characters.")
    return value


def _ordered_ids(value: object) -> object:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Cited chunk IDs must be an array.")
    return value


def _unique_ids(value: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(value)) != len(value):
        raise ValueError("Cited chunk IDs must be unique.")
    return value


ReviewIdentity = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=MAX_REVIEW_ID_CHARACTERS),
    AfterValidator(_identity),
]
ReviewComment = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=MAX_COMMENT_CHARACTERS),
    AfterValidator(_nonblank_text),
]
CitedChunkIds = Annotated[
    tuple[ReviewIdentity, ...],
    Field(max_length=MAX_CITED_CHUNKS),
    BeforeValidator(_ordered_ids),
    AfterValidator(_unique_ids),
]
ReviewSequence = Annotated[int, Field(strict=True, ge=1, le=MAX_REVIEW_SEQUENCE)]


class ReviewDecision(StrEnum):
    """A researcher's explicit opinion; never a scientific or authenticated assertion."""

    ACCEPTED = "accepted"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


class ReviewSubmission(BaseModel):
    """Client-chosen UUID and exact content form the run-scoped retry identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: UUID
    decision: ReviewDecision
    comment: ReviewComment
    cited_chunk_ids: CitedChunkIds = ()


class SavedAnswerReview(ReviewSubmission):
    """An immutable, versioned record returned by the local review store."""

    schema_version: Literal["1.0"]
    sequence: ReviewSequence
    run_id: ReviewIdentity
    created_at: AwareDatetime

    @field_validator("created_at", mode="before")
    @classmethod
    def require_utc_text(cls, value: object) -> object:
        if (
            not isinstance(value, str)
            or len(value) > 40
            or "T" not in value
            or not value.endswith("Z")
        ):
            raise ValueError("Saved review timestamps must be UTC date-time text.")
        return value


class ReviewHistoryPage(BaseModel):
    """Newest-first records with an exclusive sequence cursor, not an offset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reviews: list[SavedAnswerReview] = Field(max_length=MAX_REVIEW_LIMIT)
    next_cursor: ReviewSequence | None


class ReviewPageOptions(BaseModel):
    """Apply HTTP pagination bounds to direct store callers as well."""

    limit: int = Field(default=DEFAULT_REVIEW_LIMIT, strict=True, ge=1, le=MAX_REVIEW_LIMIT)
    cursor: ReviewSequence | None = None


class AnswerReviewError(ValueError):
    """A sanitized review failure that can be exposed without saved text or paths."""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def invalid_review_record() -> AnswerReviewError:
    return AnswerReviewError(
        "invalid_review_record", "Saved review data is invalid, inconsistent, or unsupported.", 409
    )


def review_storage_unavailable() -> AnswerReviewError:
    return AnswerReviewError(
        "review_storage_unavailable", "The saved review or evidence store is unavailable.", 503
    )
