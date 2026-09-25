"""Human review boundary with no access to generation, retrieval, or the current corpus."""

import sqlite3

from agent.review_models import (
    DEFAULT_REVIEW_LIMIT,
    AnswerReviewError,
    ReviewHistoryPage,
    ReviewSubmission,
    SavedAnswerReview,
    review_storage_unavailable,
)
from storage.answer_reviews import SQLiteAnswerReviews
from storage.evidence_export import EvidenceExporter, EvidenceExportError


class AnswerReviewService:
    """Only completed, exportable evidence can receive or expose review history."""

    def __init__(self, exporter: EvidenceExporter, store: SQLiteAnswerReviews) -> None:
        self._exporter = exporter
        self._store = store

    def _frozen_chunk_ids(self, run_id: str) -> frozenset[str]:
        try:
            bundle = self._exporter.export(run_id)
        except EvidenceExportError as exc:
            raise AnswerReviewError(exc.code, str(exc), exc.status_code) from exc
        except sqlite3.Error as exc:
            raise review_storage_unavailable() from exc
        except (TypeError, UnicodeError, RecursionError) as exc:
            # SQLite can contain non-JSON blobs even in its TEXT payload column.
            raise AnswerReviewError(
                "invalid_run_record",
                "Saved evidence is inconsistent, invalid, or an unsupported version.",
                409,
            ) from exc
        return frozenset(source.chunk.chunk_id for source in bundle.snapshot.sources)

    def create(self, run_id: str, submission: ReviewSubmission) -> tuple[SavedAnswerReview, bool]:
        frozen_chunk_ids = self._frozen_chunk_ids(run_id)
        return self._store.append(run_id, submission, frozen_chunk_ids=frozen_chunk_ids)

    def list_reviews(
        self,
        run_id: str,
        *,
        limit: int = DEFAULT_REVIEW_LIMIT,
        cursor: int | None = None,
    ) -> ReviewHistoryPage:
        frozen_chunk_ids = self._frozen_chunk_ids(run_id)
        return self._store.list_reviews(
            run_id, limit=limit, cursor=cursor, frozen_chunk_ids=frozen_chunk_ids
        )
